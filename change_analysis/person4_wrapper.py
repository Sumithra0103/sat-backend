from pathlib import Path
from uuid import uuid4
from time import perf_counter
from typing import List, Dict, Any

import numpy as np
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from PIL import Image

from change_analysis.change_service import run_change_analysis


# ============================================================
# PERSON 4 MODEL METADATA
# ============================================================

MODEL_METADATA = {
    "model_name": "SiameseResNet18ChangeDetector-V2.1",
    "backbone": "Siamese ResNet18",
    "training_dataset": "SECOND",
    "capability": "CHANGE_DETECTION"
}


# ============================================================
# CONFIGURATION
# ============================================================

WRAPPER_INPUT_DIR = (
    Path("outputs")
    / "change_detection_v2"
    / "wrapper_inputs"
)

ALLOWED_TRENDS = {
    "increased",
    "decreased",
    "remained_unchanged",
    "urban_expansion",
    "unknown"
}


# ============================================================
# NUMPY IMAGE -> UINT8 RGB
# ============================================================

def numpy_to_uint8_image(
    array: np.ndarray
) -> np.ndarray:
    """
    Convert a NumPy image into H x W x 3 uint8 RGB.
    """

    array = np.asarray(array)

    if array.ndim == 2:

        array = np.repeat(
            array[:, :, None],
            3,
            axis=2
        )

    elif array.ndim == 3:

        # CHW -> HWC
        if array.shape[0] in (1, 3, 4):

            array = np.transpose(
                array,
                (1, 2, 0)
            )

        if array.shape[2] == 1:

            array = np.repeat(
                array,
                3,
                axis=2
            )

        elif array.shape[2] >= 3:

            array = array[:, :, :3]

        else:

            raise ValueError(
                f"Unsupported image shape: {array.shape}"
            )

    else:

        raise ValueError(
            "Each image must be a 2D or 3D NumPy array."
        )

    if not np.issubdtype(
        array.dtype,
        np.number
    ):

        raise ValueError(
            "Image array must contain numeric values."
        )

    array = array.astype(
        np.float32
    )

    finite_values = array[
        np.isfinite(array)
    ]

    if finite_values.size == 0:

        raise ValueError(
            "Image contains no valid pixel values."
        )

    minimum = float(
        finite_values.min()
    )

    maximum = float(
        finite_values.max()
    )

    if (
        minimum >= 0.0
        and
        maximum <= 1.0
    ):

        array = array * 255.0

    elif (
        minimum >= 0.0
        and
        maximum <= 255.0
    ):

        pass

    else:

        low = float(
            np.percentile(
                finite_values,
                1
            )
        )

        high = float(
            np.percentile(
                finite_values,
                99
            )
        )

        if high > low:

            array = np.clip(
                array,
                low,
                high
            )

            array = (
                (array - low)
                /
                (high - low)
                *
                255.0
            )

        else:

            array = np.zeros_like(
                array
            )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=255.0,
        neginf=0.0
    )

    array = np.clip(
        array,
        0,
        255
    ).astype(
        np.uint8
    )

    return array


# ============================================================
# SAVE NUMPY IMAGE
# ============================================================

def save_numpy_image(
    array: np.ndarray,
    output_path: Path
):
    """
    Save a NumPy image as RGB PNG.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image = numpy_to_uint8_image(
        array
    )

    Image.fromarray(
        image,
        mode="RGB"
    ).save(
        output_path
    )


# ============================================================
# AUTOMATIC GEOTIFF PIXEL SIZE
# ============================================================

def get_pixel_size_from_geotiff(
    image_path
):
    """
    Read pixel dimensions from a projected GeoTIFF.
    """

    if not HAS_RASTERIO or image_path is None:
        return None

    image_path = Path(
        image_path
    )

    if image_path.suffix.lower() not in {
        ".tif",
        ".tiff"
    }:

        return None

    try:

        with rasterio.open(
            image_path
        ) as src:

            crs = src.crs
            transform = src.transform

            if crs is None:

                return None

            if not crs.is_projected:

                return None

            pixel_width = abs(
                float(transform.a)
            )

            pixel_height = abs(
                float(transform.e)
            )

            if (
                pixel_width <= 0
                or
                pixel_height <= 0
            ):

                return None

            try:

                units, factor = (
                    crs.linear_units_factor
                )

                pixel_width_m = (
                    pixel_width * factor
                )

                pixel_height_m = (
                    pixel_height * factor
                )

            except Exception:

                units = (
                    crs.linear_units
                    or ""
                ).lower()

                if units in {
                    "metre",
                    "meter",
                    "metres",
                    "meters",
                    "m"
                }:

                    pixel_width_m = pixel_width
                    pixel_height_m = pixel_height

                elif units in {
                    "foot",
                    "feet",
                    "ft"
                }:

                    pixel_width_m = (
                        pixel_width * 0.3048
                    )

                    pixel_height_m = (
                        pixel_height * 0.3048
                    )

                else:

                    return None

            return [
                pixel_width_m,
                pixel_height_m
            ]

    except Exception:

        return None


# ============================================================
# CALCULATE AREA FROM PIXEL SIZE
# ============================================================

def calculate_area_from_pixel_size(
    changed_pixels: int,
    pixel_size_m
):
    """
    Calculate changed surface area in km².
    """

    if pixel_size_m is None:

        return None

    if isinstance(
        pixel_size_m,
        (int, float)
    ):

        pixel_width = float(
            pixel_size_m
        )

        pixel_height = float(
            pixel_size_m
        )

    elif (
        isinstance(
            pixel_size_m,
            (list, tuple)
        )
        and
        len(pixel_size_m) == 2
    ):

        pixel_width = float(
            pixel_size_m[0]
        )

        pixel_height = float(
            pixel_size_m[1]
        )

    else:

        return None

    if (
        pixel_width <= 0
        or
        pixel_height <= 0
    ):

        return None

    pixel_area_m2 = (
        pixel_width
        * pixel_height
    )

    changed_area_m2 = (
        int(changed_pixels)
        * pixel_area_m2
    )

    return float(
        changed_area_m2
        / 1_000_000.0
    )


# ============================================================
# DETERMINE CHANGE TREND
# ============================================================

def determine_change_trend(
    changed: bool,
    semantic_trend=None
):
    """
    Determine semantic change trend.
    """

    if not changed:

        return "remained_unchanged"

    if semantic_trend is not None:

        semantic_trend = str(
            semantic_trend
        ).strip().lower()

        if semantic_trend in (
            ALLOWED_TRENDS
            - {"unknown"}
        ):

            return semantic_trend

    return "unknown"


# ============================================================
# CHANGE DESCRIPTION
# ============================================================

def build_change_description(
    changed: bool,
    percentage: float,
    changed_regions: int,
    spatial_location: str,
    change_trend: str
):
    """
    Create an evidence-based change description.
    """

    if not changed:

        return (
            "No significant spatial change was "
            "detected between the two images."
        )

    if percentage < 1.0:

        magnitude = "small"

    elif percentage < 10.0:

        magnitude = "moderate"

    else:

        magnitude = "substantial"

    description = (
        f"A {magnitude} amount of spatial change "
        f"was detected between the two images, "
        f"affecting approximately {percentage:.2f}% "
        f"of the scene across {changed_regions} "
        f"detected change regions."
    )

    if (
        spatial_location
        and
        spatial_location != "none"
        and
        spatial_location != "unknown"
    ):

        description += (
            f" The changes are concentrated "
            f"primarily in the "
            f"{spatial_location}."
        )

    if change_trend == "increased":

        description += (
            " Semantic analysis indicates an "
            "increase in the relevant feature."
        )

    elif change_trend == "decreased":

        description += (
            " Semantic analysis indicates a "
            "decrease in the relevant feature."
        )

    elif change_trend == "urban_expansion":

        description += (
            " Semantic analysis indicates a "
            "pattern consistent with urban expansion."
        )

    return description


# ============================================================
# PERSON 1 COMPATIBILITY FUNCTION
# ============================================================

def predict(
    query: str,
    images: List[np.ndarray],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Person-1-compatible Change Analysis contract.
    """

    wrapper_start = perf_counter()

    if parameters is None:

        parameters = {}

    if not isinstance(
        parameters,
        dict
    ):

        raise TypeError(
            "parameters must be a dictionary."
        )

    if query is None:

        query = (
            "What changed between these two images?"
        )

    query = str(query)

    confidence_threshold = float(
        parameters.get(
            "confidence_threshold",
            0.5
        )
    )

    if not (
        0.0
        <= confidence_threshold
        <= 1.0
    ):

        raise ValueError(
            "confidence_threshold must be between "
            "0.0 and 1.0."
        )

    detail_level = str(
        parameters.get(
            "detail_level",
            "high"
        )
    ).lower()

    if detail_level not in {
        "low",
        "medium",
        "high"
    }:

        detail_level = "high"

    semantic_trend = (
        parameters.get(
            "semantic_trend"
        )
    )

    if semantic_trend is not None:

        semantic_trend = str(
            semantic_trend
        ).strip().lower()

        if semantic_trend not in ALLOWED_TRENDS:

            raise ValueError(
                "semantic_trend must be one of: "
                "increased, decreased, "
                "remained_unchanged, "
                "urban_expansion, unknown."
            )

    source_paths = parameters.get(
        "source_paths"
    )

    manual_pixel_size = parameters.get(
        "pixel_size_m"
    )

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    if not isinstance(
        images,
        (list, tuple)
    ):

        raise TypeError(
            "images must be a list containing "
            "[Image_T1, Image_T2]."
        )

    if len(images) != 2:

        raise ValueError(
            "Change analysis requires exactly two images: "
            "[Image_T1, Image_T2]."
        )

    if not isinstance(
        images[0],
        np.ndarray
    ):

        raise TypeError(
            "images[0] must be a NumPy array."
        )

    if not isinstance(
        images[1],
        np.ndarray
    ):

        raise TypeError(
            "images[1] must be a NumPy array."
        )

    if images[0].ndim < 2 or images[1].ndim < 2:

        raise ValueError(
            "Each image must have at least two spatial dimensions."
        )

    if images[0].shape[:2] != images[1].shape[:2]:

        raise ValueError(
            "Before and after images must have matching "
            "spatial dimensions."
        )

    # ========================================================
    # ORIGINAL GEOTIFF METADATA
    # ========================================================

    geotiff_pixel_size = None
    geotiff_metadata_checked = False

    if (
        isinstance(
            source_paths,
            (list, tuple)
        )
        and
        len(source_paths) == 2
    ):

        geotiff_metadata_checked = True

        t1_source = Path(
            source_paths[0]
        )

        t2_source = Path(
            source_paths[1]
        )

        # Prefer T1 metadata.
        geotiff_pixel_size = (
            get_pixel_size_from_geotiff(
                t1_source
            )
        )

        # Fall back to T2.
        if geotiff_pixel_size is None:

            geotiff_pixel_size = (
                get_pixel_size_from_geotiff(
                    t2_source
                )
            )

    # ========================================================
    # PIXEL SIZE PRIORITY
    # ========================================================

    if geotiff_pixel_size is not None:

        effective_pixel_size = (
            geotiff_pixel_size
        )

        pixel_size_source = (
            "GeoTIFF metadata"
        )

    elif manual_pixel_size is not None:

        effective_pixel_size = (
            manual_pixel_size
        )

        pixel_size_source = (
            "manual parameter"
        )

    else:

        effective_pixel_size = None

        pixel_size_source = (
            "unavailable"
        )

    # ========================================================
    # CREATE UNIQUE REQUEST
    # ========================================================

    request_id = (
        uuid4().hex[:12]
    )

    request_dir = (
        WRAPPER_INPUT_DIR
        / request_id
    )

    request_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    before_path = (
        request_dir
        / "image_T1_before.png"
    )

    after_path = (
        request_dir
        / "image_T2_after.png"
    )

    # ========================================================
    # SAVE ARRAYS FOR EXISTING V2.1 SERVICE
    # ========================================================

    save_numpy_image(
        images[0],
        before_path
    )

    save_numpy_image(
        images[1],
        after_path
    )

    # ========================================================
    # RUN EXISTING PERSON 4 SERVICE
    # ========================================================

    try:

        service_result = (
            run_change_analysis(
                image_before=before_path,
                image_after=after_path,
                query=query,
                semantic_trend=semantic_trend
            )
        )

    except Exception as error:

        return {

            "status":
                "error",

            "task":
                "change_analysis",

            "model":
                MODEL_METADATA["model_name"],

            "error":
                str(error),

            "evidence": {

                "request_id":
                    request_id,

                "input_images_received":
                    2
            },

            "execution_trace": {

                "task":
                    "change_analysis",

                "wrapper":
                    "person4_wrapper",

                "status":
                    "error"
            }
        }

    # ========================================================
    # SERVICE ERROR
    # ========================================================

    if service_result.get(
        "status"
    ) != "success":

        return {

            "status":
                "error",

            "task":
                "change_analysis",

            "model":
                MODEL_METADATA["model_name"],

            "error":
                service_result.get(
                    "error",
                    "Change analysis failed."
                ),

            "evidence":
                service_result.get(
                    "evidence",
                    {}
                ),

            "execution_trace":
                service_result.get(
                    "execution_trace",
                    {}
                )
        }

    # ========================================================
    # EXTRACT RESULTS
    # ========================================================

    changed = bool(
        service_result.get(
            "change_detected",
            False
        )
    )

    percentage = float(
        service_result.get(
            "change_percentage",
            0.0
        )
    )

    changed_pixels = int(
        service_result.get(
            "changed_pixels",
            0
        )
    )

    total_pixels = int(
        service_result.get(
            "total_pixels",
            0
        )
    )

    confidence = float(
        service_result.get(
            "confidence",
            0.0
        )
    )

    changed_regions = int(
        service_result.get(
            "number_of_regions",
            0
        )
    )

    spatial_location = str(
        service_result.get(
            "spatial_location",
            "unknown"
        )
    )

    # ========================================================
    # CHANGED AREA
    # ========================================================

    changed_area_km2 = (
        calculate_area_from_pixel_size(
            changed_pixels,
            effective_pixel_size
        )
    )

    # ========================================================
    # CHANGE TREND
    # ========================================================

    change_trend = (
        determine_change_trend(
            changed=changed,
            semantic_trend=semantic_trend
        )
    )

    # ========================================================
    # CHANGE DESCRIPTION
    # ========================================================

    change_description = (
        build_change_description(
            changed=changed,
            percentage=percentage,
            changed_regions=changed_regions,
            spatial_location=spatial_location,
            change_trend=change_trend
        )
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_threshold_met = (
        confidence
        >= confidence_threshold
    )

    # ========================================================
    # DETAIL LEVEL
    # ========================================================

    if detail_level == "low":

        regions = None
        probability_map = None

    else:

        regions = service_result.get(
            "regions",
            []
        )

        probability_map = (
            service_result.get(
                "probability_map"
            )
        )

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = dict(
        service_result.get(
            "evidence",
            {}
        )
    )

    evidence.update({

        "request_id":
            request_id,

        "input_type":
            "NumPy arrays",

        "input_order":
            "[T1_before, T2_after]",

        "before_image":
            str(before_path),

        "after_image":
            str(after_path),

        "original_source_paths":
            (
                list(source_paths)
                if isinstance(
                    source_paths,
                    (list, tuple)
                )
                else None
            ),

        "change_map":
            service_result.get(
                "change_map"
            ),

        "probability_map":
            service_result.get(
                "probability_map"
            ),

        "change_percentage":
            percentage,

        "changed_pixels":
            changed_pixels,

        "total_pixels":
            total_pixels,

        "region_count":
            changed_regions,

        "spatial_location":
            spatial_location,

        "confidence":
            confidence,

        "confidence_threshold":
            confidence_threshold,

        "confidence_threshold_met":
            confidence_threshold_met,

        "pixel_size_m":
            effective_pixel_size,

        "pixel_size_source":
            pixel_size_source,

        "geotiff_metadata_checked":
            geotiff_metadata_checked,

        "geospatial_area_available":
            changed_area_km2 is not None,

        "semantic_trend":
            change_trend,

        "semantic_trend_available":
            change_trend != "unknown"
    })

    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    execution_trace = dict(
        service_result.get(
            "execution_trace",
            {}
        )
    )

    execution_trace.update({

        "wrapper":
            "person4_wrapper",

        "wrapper_contract":
            "predict(query, images, parameters)",

        "model_name":
            MODEL_METADATA["model_name"],

        "backbone":
            MODEL_METADATA["backbone"],

        "training_dataset":
            MODEL_METADATA["training_dataset"],

        "capability":
            MODEL_METADATA["capability"],

        "input_type":
            "List[np.ndarray]",

        "input_order":
            "[T1_before, T2_after]",

        "detail_level":
            detail_level,

        "confidence_threshold":
            confidence_threshold,

        "confidence_threshold_met":
            confidence_threshold_met,

        "geotiff_metadata_checked":
            geotiff_metadata_checked,

        "geotiff_pixel_size_detected":
            geotiff_pixel_size is not None,

        "pixel_size_source":
            pixel_size_source,

        "changed_area_calculated":
            changed_area_km2 is not None,

        "semantic_trend_supplied":
            semantic_trend is not None,

        "request_id":
            request_id
    })

    # ========================================================
    # PROCESSING TIME
    # ========================================================

    wrapper_processing_time = (
        perf_counter()
        - wrapper_start
    )

    # ========================================================
    # FINAL PERSON-1 CONTRACT
    # ========================================================

    result = {

        "changed_regions":
            changed_regions,

        "changed_area_km2":
            (
                round(
                    changed_area_km2,
                    8
                )
                if changed_area_km2 is not None
                else None
            ),

        "change_trend":
            change_trend,

        "change_description":
            change_description,

        "spatial_change_mask":
            service_result.get(
                "change_map"
            ),

        "confidence":
            round(
                confidence,
                4
            ),

        "status":
            "success",

        "task":
            "change_analysis",

        "query":
            query,

        "model":
            MODEL_METADATA["model_name"],

        "backbone":
            MODEL_METADATA["backbone"],

        "training_dataset":
            MODEL_METADATA["training_dataset"],

        "capability":
            MODEL_METADATA["capability"],

        "change_detected":
            changed,

        "change_percentage":
            round(
                percentage,
                4
            ),

        "changed_pixels":
            changed_pixels,

        "total_pixels":
            total_pixels,

        "original_size":
            service_result.get(
                "original_size"
            ),

        "spatial_location":
            spatial_location,

        "regions":
            regions,

        "number_of_regions":
            changed_regions,

        "probability_map":
            probability_map,

        "semantic_class":
            service_result.get(
                "semantic_class",
                "unknown"
            ),

        "confidence_threshold":
            confidence_threshold,

        "confidence_threshold_met":
            confidence_threshold_met,

        "pixel_size_m":
            effective_pixel_size,

        "pixel_size_source":
            pixel_size_source,

        "evidence":
            evidence,

        "execution_trace":
            execution_trace,

        "processing_time_seconds":
            round(
                wrapper_processing_time,
                4
            ),

        "result_file":
            service_result.get(
                "result_file"
            )
    }

    return result
