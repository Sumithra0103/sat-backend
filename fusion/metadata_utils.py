"""
Metadata Parsing and Preservation Utilities for Sentinel-1 Preprocessing.

Records geospatial and acquisition metadata (CRS, affine transform, spatial resolution,
source paths, patch ID, acquisition ID, MGRS tile) and generates standardized CSV logs.
"""

import csv
import os
from typing import Any, Dict, List


def save_processed_metadata_csv(
    samples_metadata: List[Dict[str, Any]],
    output_csv_path: str,
) -> None:
    """
    Saves complete preprocessing metadata to CSV inside data/processed/sentinel1/metadata/.

    Args:
        samples_metadata: List of dictionaries containing patch metadata and statistics.
        output_csv_path: Target CSV file path.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    fieldnames = [
        "patch_id",
        "acquisition_id",
        "mgrs_tile",
        "vv_source_path",
        "vh_source_path",
        "processed_file_path",
        "height",
        "width",
        "crs",
        "resolution_x",
        "resolution_y",
        "vv_pre_min",
        "vv_pre_max",
        "vv_pre_mean",
        "vv_pre_std",
        "vh_pre_min",
        "vh_pre_max",
        "vh_pre_mean",
        "vh_pre_std",
        "vv_pixels_below_min_db",
        "vh_pixels_below_min_db",
        "vv_pixels_above_max_db",
        "vh_pixels_above_max_db",
    ]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples_metadata:
            writer.writerow({
                "patch_id": sample.get("patch_id", ""),
                "acquisition_id": sample.get("acquisition_id", ""),
                "mgrs_tile": sample.get("mgrs_tile", ""),
                "vv_source_path": sample.get("vv_path", ""),
                "vh_source_path": sample.get("vh_path", ""),
                "processed_file_path": sample.get("processed_path", ""),
                "height": sample.get("height", 120),
                "width": sample.get("width", 120),
                "crs": sample.get("crs", ""),
                "resolution_x": sample.get("resolution", (10.0, 10.0))[0],
                "resolution_y": sample.get("resolution", (10.0, 10.0))[1],
                "vv_pre_min": f"{sample.get('vv_min', 0.0):.6f}",
                "vv_pre_max": f"{sample.get('vv_max', 0.0):.6f}",
                "vv_pre_mean": f"{sample.get('vv_mean', 0.0):.6f}",
                "vv_pre_std": f"{sample.get('vv_std', 0.0):.6f}",
                "vh_pre_min": f"{sample.get('vh_min', 0.0):.6f}",
                "vh_pre_max": f"{sample.get('vh_max', 0.0):.6f}",
                "vh_pre_mean": f"{sample.get('vh_mean', 0.0):.6f}",
                "vh_pre_std": f"{sample.get('vh_std', 0.0):.6f}",
                "vv_pixels_below_min_db": sample.get("vv_below_min", 0),
                "vh_pixels_below_min_db": sample.get("vh_below_min", 0),
                "vv_pixels_above_max_db": sample.get("vv_above_max", 0),
                "vh_pixels_above_max_db": sample.get("vh_above_max", 0),
            })


def save_preprocessing_samples_csv(
    samples_metadata: List[Dict[str, Any]],
    output_csv_path: str,
) -> None:
    """
    Creates outputs/preprocessing_samples.csv containing:
    patch_id, VV path, VH path, height, width, CRS, resolution,
    VV min, VV max, VV mean, VH min, VH max, VH mean.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    fieldnames = [
        "patch_id",
        "VV path",
        "VH path",
        "height",
        "width",
        "CRS",
        "resolution",
        "VV min",
        "VV max",
        "VV mean",
        "VH min",
        "VH max",
        "VH mean",
    ]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples_metadata:
            res = s.get("resolution", (10.0, 10.0))
            res_str = f"{res[0]}m x {res[1]}m"
            writer.writerow({
                "patch_id": s.get("patch_id", ""),
                "VV path": s.get("vv_path", ""),
                "VH path": s.get("vh_path", ""),
                "height": s.get("height", 120),
                "width": s.get("width", 120),
                "CRS": s.get("crs", ""),
                "resolution": res_str,
                "VV min": f"{s.get('vv_min', 0.0):.6f}",
                "VV max": f"{s.get('vv_max', 0.0):.6f}",
                "VV mean": f"{s.get('vv_mean', 0.0):.6f}",
                "VH min": f"{s.get('vh_min', 0.0):.6f}",
                "VH max": f"{s.get('vh_max', 0.0):.6f}",
                "VH mean": f"{s.get('vh_mean', 0.0):.6f}",
            })
