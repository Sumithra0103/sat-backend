"""
SatQueryAI Person 5 — Final End-to-End Integration Pipeline
Step 5F Implementation

Entrypoint for Person 1 (Agent / Platform Lead).
Provides run_pipeline() to automatically route, execute, and format JSON-serializable
results for multimodal classification and bi-temporal change analysis queries.
"""

import sys
import json
import math
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing person5_interface layer
from fusion.person5_interface import run_person5


def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts NumPy types, float32, int64, PyTorch Tensors, NaNs, and Path objects
    into standard JSON-serializable Python data structures (dict, list, str, int, float, bool, None).
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif HAS_TORCH and torch.is_tensor(obj):
        return make_json_safe(obj.detach().cpu().numpy().tolist())
    elif hasattr(obj, 'as_posix'):  # Path object
        return str(obj)
    else:
        return str(obj)



def format_error_response(
    query: str,
    image: Optional[str],
    t1: Optional[str],
    t2: Optional[str],
    mode: Optional[str],
    error_type: str,
    error_message: str
) -> Dict[str, Any]:
    """Helper to build a standardized, JSON-safe error response."""
    res = {
        "person5_pipeline": {
            "version": "1.0",
            "status": "error",
            "mode": mode
        },
        "request": {
            "query": query if query else "",
            "inputs": {
                "image": image,
                "t1": t1,
                "t2": t2
            }
        },
        "result": None,
        "evidence": None,
        "execution": {},
        "scientific_disclaimer": {
            "ground_truth_available": False,
            "change_labels_available": False,
            "confidence_is_calibrated": False
        },
        "error": {
            "type": error_type,
            "message": error_message
        }
    }
    return make_json_safe(res)


def run_pipeline(
    query: str,
    image: Optional[str] = None,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
    mode: Optional[str] = None,
    save_evidence: bool = False
) -> Dict[str, Any]:
    """
    Main End-to-End Integration Entrypoint for Person 1 (Agent / Platform Lead).

    Args:
        query: Natural language query string.
        image: Sentinel-2 image filename for multimodal mode.
        t1: Sentinel-2 T1 image filename for temporal mode.
        t2: Sentinel-2 T2 image filename for temporal mode.
        mode: Explicit mode ('multimodal' or 'temporal'). Optional.
        save_evidence: Whether to generate visual evidence files.

    Returns:
        100% JSON-serializable Python dictionary adhering to Person 5 Output Contract.
    """
    start_time = time.time()

    # 1. Validation & Automatic Mode Routing
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        return format_error_response(
            query=query, image=image, t1=t1, t2=t2, mode=mode,
            error_type="InputValidationError",
            error_message="A non-empty 'query' string parameter is required."
        )

    determined_mode = None

    if mode is not None:
        mode_clean = str(mode).strip().lower()
        if mode_clean not in ["multimodal", "temporal"]:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode=mode,
                error_type="InputValidationError",
                error_message=f"Invalid mode '{mode}'. Supported modes are 'multimodal' and 'temporal'."
            )
        determined_mode = mode_clean
    else:
        # Automatic mode determination from input arguments
        if t1 is not None and t2 is not None:
            determined_mode = "temporal"
        elif image is not None and t1 is None and t2 is None:
            determined_mode = "multimodal"
        elif t1 is not None and t2 is None:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode="temporal",
                error_type="InputValidationError",
                error_message="Parameter 't2' is missing. Both 't1' and 't2' are required for temporal change analysis."
            )
        elif t2 is not None and t1 is None:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode="temporal",
                error_type="InputValidationError",
                error_message="Parameter 't1' is missing. Both 't1' and 't2' are required for temporal change analysis."
            )
        else:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode=None,
                error_type="InputValidationError",
                error_message="Insufficient inputs provided. Specify either 'image' for multimodal mode or 't1' and 't2' for temporal mode."
            )

    # 2. Input Sanity Check for Temporal Mode
    if determined_mode == "temporal":
        if not t1 or not t2:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode="temporal",
                error_type="InputValidationError",
                error_message="Both 't1' and 't2' image parameters are required for temporal mode."
            )
        if Path(t1).name == Path(t2).name:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode="temporal",
                error_type="InputValidationError",
                error_message="t1 and t2 must represent different acquisition dates. Identical filenames supplied."
            )

    # 3. Input Sanity Check for Multimodal Mode
    if determined_mode == "multimodal":
        if not image:
            return format_error_response(
                query=query, image=image, t1=t1, t2=t2, mode="multimodal",
                error_type="InputValidationError",
                error_message="Parameter 'image' is required for multimodal mode."
            )

    # 4. Dispatch to Person-5 Unified Interface
    raw_response = run_person5(
        mode=determined_mode,
        query=query,
        image=image,
        t1=t1,
        t2=t2,
        save_evidence=save_evidence
    )

    # 5. Handle Dispatch Errors
    if raw_response.get("person5_interface", {}).get("status") == "error":
        err_info = raw_response.get("error", {})
        return format_error_response(
            query=query, image=image, t1=t1, t2=t2, mode=determined_mode,
            error_type=err_info.get("type", "ExecutionError"),
            error_message=err_info.get("message", "An error occurred during query execution.")
        )

    raw_result = raw_response.get("result", {})
    raw_evidence = raw_response.get("evidence", {})

    # 6. Normalize Multimodal vs Temporal Output Structures
    normalized_result = {}
    normalized_evidence = {}

    if determined_mode == "multimodal":
        query_analysis_data = raw_result.get("query_analysis", {})
        detections = {}

        if isinstance(query_analysis_data, dict):
            for k, v in query_analysis_data.items():
                if isinstance(v, dict) and "detected" in v and "score" in v:
                    detections[k] = {
                        "detected": bool(v.get("detected", False)),
                        "score": float(v.get("score", 0.0)),
                        "confidence_percent": float(v.get("confidence_percent", 0.0)),
                        "strongest_class": v.get("strongest_class")
                    }

        normalized_result = {
            "task": "multimodal_query",
            "patch_id": raw_result.get("patch_id"),
            "s1_name": raw_result.get("s1_name"),
            "s2_name": raw_result.get("s2_name"),
            "s1_filename": raw_result.get("s1_filename"),
            "s2_filename": raw_result.get("s2_filename"),
            "detections": detections if detections else query_analysis_data,
            "classification": {
                "top_predictions": [c for c in raw_result.get("all_class_results", []) if isinstance(c, dict) and c.get("detected")][:5],
                "requested_results": raw_result.get("requested_results", []),
                "all_class_results": raw_result.get("all_class_results", [])
            },
            "matched_query_groups": raw_result.get("matched_query_groups", []),
            "input_files": raw_result.get("input_files"),
            "model_information": raw_result.get("model_information")
        }

        if save_evidence:
            normalized_evidence = {
                "input_evidence": raw_evidence.get("input_evidence"),
                "visual_evidence_path": raw_evidence.get("visual_evidence_path"),
                "query_analysis_path": raw_evidence.get("query_analysis_path"),
                "gradcam_evidence_paths": raw_evidence.get("gradcam_evidence_paths")
            }
        else:
            normalized_evidence = {
                "input_evidence": raw_evidence.get("input_evidence"),
                "visual_evidence_path": None,
                "query_analysis_path": None,
                "gradcam_evidence_paths": None
            }

    elif determined_mode == "temporal":
        hotspot_raw = raw_result.get("hotspot", {})
        hotspot_norm = {
            "detected": bool(hotspot_raw.get("hotspot_detected", False)),
            "area_pixels": int(hotspot_raw.get("hotspot_area_pixels", 0)),
            "area_percent": float(hotspot_raw.get("hotspot_area_percent", 0.0)),
            "centroid_x": hotspot_raw.get("centroid_x"),
            "centroid_y": hotspot_raw.get("centroid_y"),
            "bbox": hotspot_raw.get("bbox")
        }

        normalized_result = {
            "task": "temporal_change_analysis",
            "spatial_id": raw_result.get("spatial_id"),
            "temporal_gap_days": raw_result.get("temporal_gap_days"),
            "t1_s2": raw_result.get("t1_s2"),
            "t2_s2": raw_result.get("t2_s2"),
            "t1_s1": raw_result.get("t1_s1"),
            "t2_s1": raw_result.get("t2_s1"),
            "change_evidence": {
                "mean_score": raw_result.get("change_evidence_score"),
                "adaptive_threshold_percentile": raw_result.get("adaptive_threshold_percentile", 90),
                "adaptive_threshold": raw_result.get("adaptive_threshold"),
                "raw_changed_area_percent": raw_result.get("raw_changed_area_percent"),
                "filtered_changed_area_percent": raw_result.get("filtered_changed_area_percent"),
                "optical_contribution_percent": raw_result.get("optical_contribution_percent"),
                "sar_contribution_percent": raw_result.get("sar_contribution_percent"),
                "modality_spatial_agreement": raw_result.get("modality_spatial_agreement")
            },
            "hotspot": hotspot_norm,
            "input_validation": raw_result.get("input_validation")
        }

        if save_evidence:
            normalized_evidence = {
                "visual_evidence_path": raw_evidence.get("visual_evidence_path")
            }
        else:
            normalized_evidence = {
                "visual_evidence_path": None
            }

    elapsed_time = round(time.time() - start_time, 4)
    device_str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"


    pipeline_response = {
        "person5_pipeline": {
            "version": "1.0",
            "status": "success",
            "mode": determined_mode
        },
        "request": {
            "query": query,
            "inputs": {
                "image": image,
                "t1": t1,
                "t2": t2
            }
        },
        "result": normalized_result,
        "evidence": normalized_evidence,
        "execution": {
            "inference_time_seconds": elapsed_time,
            "device": device_str
        },
        "scientific_disclaimer": {
            "ground_truth_available": False,
            "change_labels_available": False,
            "confidence_is_calibrated": False
        },
        "error": None
    }

    return make_json_safe(pipeline_response)


def main():
    parser = argparse.ArgumentParser(
        description="SatQueryAI Person 5 — Final End-to-End Integration Pipeline CLI."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["multimodal", "temporal"],
        help="Optional explicit mode: 'multimodal' or 'temporal'"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query string"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Sentinel-2 image filename (for multimodal mode)"
    )
    parser.add_argument(
        "--t1",
        type=str,
        default=None,
        help="Sentinel-2 T1 image filename (for temporal mode)"
    )
    parser.add_argument(
        "--t2",
        type=str,
        default=None,
        help="Sentinel-2 T2 image filename (for temporal mode)"
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save visual evidence files"
    )

    args = parser.parse_args()

    res = run_pipeline(
        query=args.query,
        image=args.image,
        t1=args.t1,
        t2=args.t2,
        mode=args.mode,
        save_evidence=args.save_evidence
    )

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
