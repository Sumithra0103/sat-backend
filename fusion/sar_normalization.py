"""
SAR Normalization Utilities.

Specifically tailored for BigEarthNet-S1 pre-calibrated backscatter in decibels (dB).
CRITICAL: The input data are ALREADY in dB (negative float values).
DO NOT apply log10 transformation.

This module provides:
1. Pre-clipping statistical analysis (min, max, mean, std, outlier pixel counts).
2. Configurable dB clipping (default: min_db = -30.0, max_db = 0.0).
3. Independent min-max linear normalization mapping [-30.0 dB, 0.0 dB] to [0.0, 1.0].
"""

from typing import Any, Dict, Tuple
import numpy as np


def compute_band_stats(
    arr: np.ndarray,
    min_db: float = -30.0,
    max_db: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes rigorous distribution statistics and outlier counts for a SAR band before clipping.

    Args:
        arr: 2D numpy array of SAR backscatter in dB.
        min_db: Lower threshold in dB (default: -30.0).
        max_db: Upper threshold in dB (default: 0.0).

    Returns:
        Dictionary containing statistical metrics.
    """
    total_pixels = int(arr.size)
    valid_mask = np.isfinite(arr)
    invalid_count = int(total_pixels - np.count_nonzero(valid_mask))

    if invalid_count > 0:
        valid_data = arr[valid_mask]
    else:
        valid_data = arr

    if valid_data.size == 0:
        return {
            "total_pixels": total_pixels,
            "invalid_pixels": invalid_count,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "below_min_count": 0,
            "above_max_count": 0,
        }

    min_val = float(np.min(valid_data))
    max_val = float(np.max(valid_data))
    mean_val = float(np.mean(valid_data))
    std_val = float(np.std(valid_data))

    below_min = int(np.count_nonzero(valid_data < min_db))
    above_max = int(np.count_nonzero(valid_data > max_db))

    return {
        "total_pixels": total_pixels,
        "invalid_pixels": invalid_count,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "below_min_count": below_min,
        "above_max_count": above_max,
    }


def normalize_band(
    arr: np.ndarray,
    min_db: float = -30.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """
    Clips and linearly normalizes a single SAR channel (already in dB) to [0.0, 1.0].

    Formula:
        clipped = np.clip(arr, min_db, max_db)
        normalized = (clipped - min_db) / (max_db - min_db)

    Mapping:
        -30.0 dB -> 0.0
          0.0 dB -> 1.0

    Args:
        arr: 2D numpy array in dB.
        min_db: Lower clipping boundary.
        max_db: Upper clipping boundary.

    Returns:
        Normalized array with float32 dtype and values bounded in [0.0, 1.0].
    """
    if max_db <= min_db:
        raise ValueError(f"max_db ({max_db}) must be strictly greater than min_db ({min_db})")

    clipped = np.clip(arr.astype(np.float32), min_db, max_db)
    normalized = (clipped - min_db) / (max_db - min_db)
    return normalized.astype(np.float32)


def clip_and_normalize_sar(
    vv_arr: np.ndarray,
    vh_arr: np.ndarray,
    min_db: float = -30.0,
    max_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Calculates statistics before clipping, clips values, and normalizes VV and VH
    independently to [0.0, 1.0].

    Args:
        vv_arr: 2D numpy array of VV backscatter in dB.
        vh_arr: 2D numpy array of VH backscatter in dB.
        min_db: Lower clip threshold in dB.
        max_db: Upper clip threshold in dB.

    Returns:
        Tuple of:
            - norm_vv: Normalized VV array in [0.0, 1.0]
            - norm_vh: Normalized VH array in [0.0, 1.0]
            - stats_report: Dictionary with pre-clipping and post-clipping statistics
    """
    vv_pre_stats = compute_band_stats(vv_arr, min_db=min_db, max_db=max_db)
    vh_pre_stats = compute_band_stats(vh_arr, min_db=min_db, max_db=max_db)

    norm_vv = normalize_band(vv_arr, min_db=min_db, max_db=max_db)
    norm_vh = normalize_band(vh_arr, min_db=min_db, max_db=max_db)

    vv_clipped = np.clip(vv_arr, min_db, max_db)
    vh_clipped = np.clip(vh_arr, min_db, max_db)

    vv_post_stats = {
        "min": float(np.min(vv_clipped)),
        "max": float(np.max(vv_clipped)),
        "mean": float(np.mean(vv_clipped)),
        "std": float(np.std(vv_clipped)),
    }
    vh_post_stats = {
        "min": float(np.min(vh_clipped)),
        "max": float(np.max(vh_clipped)),
        "mean": float(np.mean(vh_clipped)),
        "std": float(np.std(vh_clipped)),
    }

    vv_norm_range = (float(np.min(norm_vv)), float(np.max(norm_vv)))
    vh_norm_range = (float(np.min(norm_vh)), float(np.max(norm_vh)))

    stats_report = {
        "vv_pre": vv_pre_stats,
        "vh_pre": vh_pre_stats,
        "vv_post": vv_post_stats,
        "vh_post": vh_post_stats,
        "vv_norm_range": vv_norm_range,
        "vh_norm_range": vh_norm_range,
        "total_invalid_pixels": vv_pre_stats["invalid_pixels"] + vh_pre_stats["invalid_pixels"],
        "total_below_min_db": vv_pre_stats["below_min_count"] + vh_pre_stats["below_min_count"],
        "total_above_max_db": vv_pre_stats["above_max_count"] + vh_pre_stats["above_max_count"],
    }

    return norm_vv, norm_vh, stats_report
