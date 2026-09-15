"""
Sentinel-2 Optical Reflectance Normalization Utilities.

Specifically tailored for BigEarthNet-S2 Level-2A surface reflectance imagery.
Level-2A values represent Bottom-Of-Atmosphere (BOA) surface reflectance
quantized by an ESA scaling factor of 10,000.

Normalization formula:
    reflectance = pixel_value / 10000.0
    reflectance = np.clip(reflectance, 0.0, 1.0)
"""

from typing import Any, Dict, List, Optional
import numpy as np


def normalize_sentinel2_reflectance(
    raw_bands: np.ndarray,
    scale_factor: float = 10000.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> np.ndarray:
    """
    Converts quantized uint16/float32 Sentinel-2 Level-2A reflectance to [0.0, 1.0].

    Applies linear division by scale_factor (10000.0) followed by clamping to [0.0, 1.0].

    Args:
        raw_bands: Array of shape (12, H, W) or (H, W) with raw reflectance DN values.
        scale_factor: ESA quantization scaling factor (default: 10000.0).
        clamp_min: Lower bound for reflectance values (default: 0.0).
        clamp_max: Upper bound for reflectance values (default: 1.0).

    Returns:
        Float32 numpy array with normalized reflectance values clamped to [0.0, 1.0].
    """
    scaled = raw_bands.astype(np.float32) / float(scale_factor)
    return np.clip(scaled, clamp_min, clamp_max).astype(np.float32)


def compute_sentinel2_stats(
    arr: np.ndarray,
    band_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes overall and per-band statistics for a normalized Sentinel-2 array.

    Args:
        arr: Array of shape (12, H, W) with normalized reflectance.
        band_names: Optional list of band identifiers.

    Returns:
        Dictionary containing overall min, max, mean, std and per-band breakdowns.
    """
    stats: Dict[str, Any] = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "bands": {},
    }

    if arr.ndim == 3:
        num_channels = arr.shape[0]
        for c in range(num_channels):
            b_name = band_names[c] if band_names and c < len(band_names) else f"channel_{c}"
            channel_data = arr[c]
            stats["bands"][b_name] = {
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
            }

    return stats
