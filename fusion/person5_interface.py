"""
Unified SatQueryAI Person-5 Multimodal Backend Interface
Step 5E Implementation

Acts as a clean, unified orchestration interface over:
  - fusion.multimodal_query (Mode: "multimodal")
  - fusion.temporal_query   (Mode: "temporal")

Exposes Python API functions:
  - run_multimodal(image, query, save_evidence=False)
  - run_temporal(t1, t2, query, save_evidence=False)
  - run_person5(mode, query, image=None, t1=None, t2=None, save_evidence=False)

CLI execution supported:
  python -m fusion.person5_interface --mode multimodal --image <S2_NPY> --query "..." [--save-evidence]
  python -m fusion.person5_interface --mode temporal --t1 <S2_T1_NPY> --t2 <S2_T2_NPY> --query "..." [--save-evidence]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing implementations without duplicating logic
import fusion.multimodal_query as mq
import fusion.temporal_query as tq


def run_multimodal(
    image: str,
    query: str,
    save_evidence: bool = False
) -> Dict[str, Any]:
    """
    Executes multimodal SAR + Optical inference for a single Sentinel-2 image.
    Automatically resolves the corresponding Sentinel-1 image via metadata.
    """
    if not image or not query:
        return {
            "person5_interface": {
                "mode": "multimodal",
                "status": "error"
            },
            "query": query if query else "",
            "error": {
                "type": "InputValidationError",
                "message": "Both 'image' and 'query' parameters are required for multimodal mode."
            }
        }

    try:
        device = mq.get_device()
        raw_res = mq.run_normal_inference(
            image_name=image,
            query=query,
            threshold=0.50,
            device=device,
            save_evidence=save_evidence
        )

        if not raw_res:
            out_json = PROJECT_ROOT / "outputs/fusion/multimodal_query_result.json"
            if out_json.exists():
                with open(out_json, "r", encoding="utf-8") as f:
                    raw_res = json.load(f)

        if not raw_res or not isinstance(raw_res, dict):
            return {
                "person5_interface": {
                    "mode": "multimodal",
                    "status": "error"
                },
                "query": query,
                "error": {
                    "type": "MultimodalExecutionError",
                    "message": "Failed to retrieve multimodal inference results."
                }
            }

        s1_name = raw_res.get("s1_name")
        s2_name = raw_res.get("s2_name")
        s1_filename = f"{s1_name}.npy" if s1_name and not s1_name.endswith(".npy") else s1_name
        s2_filename = Path(image).name

        evidence_sec = {
            "input_evidence": raw_res.get("input_evidence", {}),
            "visual_evidence_path": raw_res.get("evidence_image") or raw_res.get("visual_evidence", {}).get("evidence_image"),
            "query_analysis_path": raw_res.get("query_analysis_image"),
            "gradcam_evidence_paths": raw_res.get("gradcam_evidence_paths")
        }

        result_sec = {
            "patch_id": raw_res.get("patch_id"),
            "s1_name": s1_name,
            "s2_name": s2_name,
            "s1_filename": s1_filename,
            "s2_filename": s2_filename,
            "requested_results": raw_res.get("requested_results"),
            "all_class_results": raw_res.get("all_class_results"),
            "matched_query_groups": raw_res.get("matched_query_groups"),
            "query_analysis": raw_res.get("query_analysis"),
            "inference_time_seconds": raw_res.get("inference_time_seconds"),
            "model_information": raw_res.get("model"),
            "input_files": raw_res.get("input_files")
        }

        return {
            "person5_interface": {
                "mode": "multimodal",
                "status": "success"
            },
            "query": query,
            "result": result_sec,
            "evidence": evidence_sec
        }

    except Exception as e:
        return {
            "person5_interface": {
                "mode": "multimodal",
                "status": "error"
            },
            "query": query,
            "error": {
                "type": e.__class__.__name__,
                "message": str(e)
            }
        }


def run_temporal(
    t1: str,
    t2: str,
    query: str,
    save_evidence: bool = False
) -> Dict[str, Any]:
    """
    Executes bi-temporal S1+S2 change evidence query for two Sentinel-2 images (T1 & T2).
    Automatically resolves corresponding Sentinel-1 images.
    """
    if not t1 or not t2 or not query:
        return {
            "person5_interface": {
                "mode": "temporal",
                "status": "error"
            },
            "query": query if query else "",
            "error": {
                "type": "InputValidationError",
                "message": "Parameters 't1', 't2', and 'query' are all required for temporal mode."
            }
        }

    try:
        raw_res = tq.run_temporal_query(
            s2_t1_name=t1,
            s2_t2_name=t2,
            query_text=query,
            save_evidence=save_evidence
        )

        if "error" in raw_res:
            return {
                "person5_interface": {
                    "mode": "temporal",
                    "status": "error"
                },
                "query": query,
                "error": {
                    "type": "InputValidationError",
                    "message": raw_res["error"]
                }
            }

        t_info = raw_res.get("temporal_analysis", {})
        c_info = raw_res.get("change_evidence", {})
        h_info = raw_res.get("hotspot", {})
        e_info = raw_res.get("evidence", {})

        result_sec = {
            "t1_s2": t_info.get("t1_s2"),
            "t2_s2": t_info.get("t2_s2"),
            "t1_s1": t_info.get("t1_s1"),
            "t2_s1": t_info.get("t2_s1"),
            "spatial_id": t_info.get("spatial_id"),
            "temporal_gap_days": t_info.get("temporal_gap_days"),
            "adaptive_threshold_percentile": c_info.get("adaptive_threshold_percentile", 90),
            "adaptive_threshold": c_info.get("adaptive_threshold"),
            "raw_changed_area_percent": c_info.get("raw_changed_area_percent"),
            "filtered_changed_area_percent": c_info.get("filtered_changed_area_percent"),
            "change_evidence_score": c_info.get("change_evidence_score"),
            "optical_contribution_percent": c_info.get("optical_contribution_percent"),
            "sar_contribution_percent": c_info.get("sar_contribution_percent"),
            "modality_spatial_agreement": c_info.get("modality_spatial_agreement"),
            "hotspot": h_info,
            "inference_time_seconds": raw_res.get("inference_time_seconds"),
            "input_validation": raw_res.get("input_validation")
        }

        evidence_sec = {
            "visual_evidence_path": raw_res.get("visual_evidence_path"),
            "ground_truth_available": e_info.get("ground_truth_available", False),
            "change_labels_available": e_info.get("change_labels_available", False),
            "confidence_is_calibrated": e_info.get("confidence_is_calibrated", False)
        }

        return {
            "person5_interface": {
                "mode": "temporal",
                "status": "success"
            },
            "query": query,
            "result": result_sec,
            "evidence": evidence_sec
        }

    except Exception as e:
        return {
            "person5_interface": {
                "mode": "temporal",
                "status": "error"
            },
            "query": query,
            "error": {
                "type": e.__class__.__name__,
                "message": str(e)
            }
        }


def run_person5(
    mode: str,
    query: str,
    image: Optional[str] = None,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
    save_evidence: bool = False
) -> Dict[str, Any]:
    """
    Unified dispatch function for Person-5 multimodal backend interface.
    """
    if not mode:
        return {
            "person5_interface": {
                "mode": "unknown",
                "status": "error"
            },
            "query": query if query else "",
            "error": {
                "type": "InputValidationError",
                "message": "Parameter 'mode' is required. Supported modes: 'multimodal', 'temporal'."
            }
        }

    mode_clean = str(mode).strip().lower()

    if mode_clean == "multimodal":
        if not image:
            return {
                "person5_interface": {
                    "mode": "multimodal",
                    "status": "error"
                },
                "query": query if query else "",
                "error": {
                    "type": "InputValidationError",
                    "message": "Parameter 'image' is required for multimodal mode."
                }
            }
        return run_multimodal(image=image, query=query, save_evidence=save_evidence)

    elif mode_clean == "temporal":
        if not t1 or not t2:
            return {
                "person5_interface": {
                    "mode": "temporal",
                    "status": "error"
                },
                "query": query if query else "",
                "error": {
                    "type": "InputValidationError",
                    "message": "Both 't1' and 't2' parameters are required for temporal mode."
                }
            }
        return run_temporal(t1=t1, t2=t2, query=query, save_evidence=save_evidence)

    else:
        return {
            "person5_interface": {
                "mode": mode,
                "status": "error"
            },
            "query": query if query else "",
            "error": {
                "type": "InputValidationError",
                "message": f"Unsupported mode '{mode}'. Supported modes are 'multimodal' and 'temporal'."
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Unified SatQueryAI Person-5 Multimodal Backend Interface."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["multimodal", "temporal"],
        help="Analysis mode: 'multimodal' or 'temporal'"
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
        help="Sentinel-2 image filename (Required for mode='multimodal')"
    )
    parser.add_argument(
        "--t1",
        type=str,
        default=None,
        help="Sentinel-2 T1 image filename (Required for mode='temporal')"
    )
    parser.add_argument(
        "--t2",
        type=str,
        default=None,
        help="Sentinel-2 T2 image filename (Required for mode='temporal')"
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save visual evidence images"
    )

    args = parser.parse_args()

    res = run_person5(
        mode=args.mode,
        query=args.query,
        image=args.image,
        t1=args.t1,
        t2=args.t2,
        save_evidence=args.save_evidence
    )

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
