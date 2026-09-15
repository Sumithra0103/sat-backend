from pathlib import Path
import numpy as np
from PIL import Image

from change_analysis.v2_adapter import V2DetectorAdapter


def analyze_regions_and_location(change_map: np.ndarray):
    """
    Analyzes binary change map to compute discrete region counts,
    bounding box coordinates, and spatial location description.
    """
    if change_map is None or not isinstance(change_map, np.ndarray):
        return 0, "unknown", []

    h, w = change_map.shape[:2]
    y_indices, x_indices = np.where(change_map > 0)

    if len(y_indices) == 0:
        return 0, "none", []

    # Calculate spatial centroid
    mean_y = float(np.mean(y_indices))
    mean_x = float(np.mean(x_indices))

    rel_y = mean_y / h
    rel_x = mean_x / w

    loc_y = "central"
    if rel_y < 0.33:
        loc_y = "northern"
    elif rel_y > 0.66:
        loc_y = "southern"

    loc_x = ""
    if rel_x < 0.33:
        loc_x = "-western"
    elif rel_x > 0.66:
        loc_x = "-eastern"

    if loc_y == "central" and not loc_x:
        spatial_location = "central sector"
    else:
        spatial_location = f"{loc_y}{loc_x} sector"

    # Region segmentation via grid-connected components
    regions = []
    try:
        from scipy.ndimage import label, find_objects
        labeled_array, num_features = label(change_map > 0)
        slices = find_objects(labeled_array)
        for idx, slc in enumerate(slices[:20]): # Limit to top 20 regions
            if slc is None:
                continue
            ymin, ymax = slc[0].start, slc[0].stop
            xmin, xmax = slc[1].start, slc[1].stop
            reg_area = int(np.sum(labeled_array[slc] == (idx + 1)))
            regions.append({
                "region_id": idx + 1,
                "bbox": [ymin, xmin, ymax, xmax],
                "area_pixels": reg_area
            })
        number_of_regions = int(num_features)
    except Exception:
        # Fallback grid partitioning
        ymin, ymax = int(y_indices.min()), int(y_indices.max())
        xmin, xmax = int(x_indices.min()), int(x_indices.max())
        number_of_regions = 1 if len(y_indices) > 0 else 0
        regions.append({
            "region_id": 1,
            "bbox": [ymin, xmin, ymax, xmax],
            "area_pixels": int(len(y_indices))
        })

    return number_of_regions, spatial_location, regions


_ADAPTER_INSTANCE = None


def get_adapter():
    global _ADAPTER_INSTANCE
    if _ADAPTER_INSTANCE is None:
        _ADAPTER_INSTANCE = V2DetectorAdapter()
    return _ADAPTER_INSTANCE


def run_change_analysis(
    image_before,
    image_after,
    query=None,
    semantic_trend=None
):
    """
    Executes bi-temporal change detection using Person 4 V2 Change Detector Adapter.
    """
    adapter = get_adapter()
    result = adapter(image_before, image_after)

    change_map = result.get("change_map")
    num_regions, spatial_loc, regions = analyze_regions_and_location(change_map)

    output_dir = Path("outputs") / "change_detection_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "latest_change_result.png"

    if change_map is not None:
        Image.fromarray((change_map * 255).astype(np.uint8), mode="L").save(result_file)

    return {
        "status": "success",
        "model": result.get("model", "SiameseResNet18ChangeDetector-V2.1"),
        "change_detected": result.get("change_detected", False),
        "change_percentage": result.get("change_percentage", 0.0),
        "changed_pixels": result.get("changed_pixels", 0),
        "total_pixels": result.get("total_pixels", 0),
        "confidence": result.get("confidence", 0.90),
        "change_map": change_map,
        "probability_map": result.get("probability_map"),
        "original_size": result.get("original_size"),
        "number_of_regions": num_regions,
        "spatial_location": spatial_loc,
        "regions": regions,
        "result_file": str(result_file),
        "evidence": {
            "query": query,
            "semantic_trend": semantic_trend,
            "detector": "SiameseResNet18ChangeDetector-V2.1"
        },
        "execution_trace": {
            "module": "change_analysis.change_service",
            "status": "success"
        }
    }


if __name__ == "__main__":
    print("change_service module loaded successfully.")
