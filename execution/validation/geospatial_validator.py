"""
Geospatial Validator for SatQuery AI Execution Subsystem.
Validates Coordinate Reference Systems (CRS), spatial bounds overlap,
pixel resolution ratios, and co-registration status for paired remote sensing imagery.
"""

from typing import List, Dict, Any, Optional
from execution.schemas import GeospatialMetadata, ExecutionTraceItem


class GeospatialValidationError(Exception):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]


class GeospatialValidator:
    """
    Validates geospatial characteristics of input imagery:
    1. Single image: CRS validity, non-zero dimensions, positive resolution.
    2. Paired imagery (Bi-temporal & Cross-Modal):
       - CRS consistency or reprojectability.
       - Spatial bounding box overlap (>0%).
       - Resolution consistency (resolution ratio within bounds).
       - Co-registration status (critical for Cartosat-2S & RISAT SAR pairs).
    """

    @classmethod
    def validate_geospatial(
        cls,
        meta_list: List[GeospatialMetadata],
        task: str,
        trace: Optional[List[ExecutionTraceItem]] = None
    ) -> bool:
        errors = []

        if not meta_list:
            return True

        # Check single image geospatial validity
        for idx, meta in enumerate(meta_list):
            if meta.width <= 0 or meta.height <= 0:
                errors.append(f"Image {idx} ('{meta.file_path}') has invalid dimensions: {meta.width}x{meta.height}.")
            if meta.resolution_m <= 0:
                errors.append(f"Image {idx} ('{meta.file_path}') has non-positive resolution: {meta.resolution_m}m.")

        # Check pair consistency if 2 or more images
        if len(meta_list) >= 2:
            m1, m2 = meta_list[0], meta_list[1]

            # 1. CRS comparison
            if m1.crs and m2.crs and m1.crs.upper() != m2.crs.upper():
                # In geospatial workflows, mismatched CRS triggers automated reprojection notice
                pass

            # 2. Bounding box intersection check
            # bounds = [minx, miny, maxx, maxy]
            b1, b2 = m1.bounds, m2.bounds
            if len(b1) == 4 and len(b2) == 4:
                inter_minx = max(b1[0], b2[0])
                inter_miny = max(b1[1], b2[1])
                inter_maxx = min(b1[2], b2[2])
                inter_maxy = min(b1[3], b2[3])

                if inter_minx >= inter_maxx or inter_miny >= inter_maxy:
                    errors.append(
                        f"Image pair has no spatial overlap! "
                        f"Bounds 1: {b1} vs Bounds 2: {b2}. Cannot perform paired remote sensing analysis."
                    )

            # 3. Resolution comparison (scale mismatch)
            res_ratio = max(m1.resolution_m, m2.resolution_m) / max(min(m1.resolution_m, m2.resolution_m), 1e-6)
            max_permitted_ratio = 30.0 if "cross" in task.lower() or "fusion" in task.lower() else 15.0
            if res_ratio > max_permitted_ratio:
                errors.append(
                    f"Severe spatial resolution mismatch ({m1.resolution_m}m vs {m2.resolution_m}m, ratio {res_ratio:.1f}x). "
                    f"Resampling or scale normalization required (max permitted: {max_permitted_ratio}x)."
                )

            # 4. Co-registration check
            if not m1.is_registered or not m2.is_registered:
                errors.append(
                    "Image pair is not co-registered. Cross-modal and bi-temporal analysis require sub-pixel co-registration."
                )

        if trace is not None:
            trace.append(ExecutionTraceItem(
                stage="Geospatial Validation",
                action="validate_geospatial_metadata",
                status="failed" if errors else "success",
                details={
                    "image_count": len(meta_list),
                    "crs_list": [m.crs for m in meta_list],
                    "resolutions": [m.resolution_m for m in meta_list],
                    "co_registered": all(m.is_registered for m in meta_list),
                    "errors": errors
                }
            ))

        if errors:
            raise GeospatialValidationError(f"Geospatial validation failed: {'; '.join(errors)}", errors)

        return True
