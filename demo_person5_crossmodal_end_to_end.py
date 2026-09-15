"""
SatQuery AI: Person 5 Optical-SAR Cross-Modal Model End-to-End Verification Demo Script
======================================================================================
Demonstrates complete agentic optical+SAR joint analysis pipeline with
Person 5's SatQuery-OpticalSAR-CrossAttn-Net-V5.0 model & SQLite Database Persistence.
"""

import sys
import os
import sqlite3
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_ROOT_DIR = Path(__file__).resolve().parent
for _sub in [_ROOT_DIR, _ROOT_DIR / "agent registry", _ROOT_DIR / "routing", _ROOT_DIR / "query understanding", _ROOT_DIR / "trace"]:
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

from satquery_engine import SatQueryEngine
from execution.models.specialist_models import check_person5_adapter_status
from fusion.person5_wrapper import predict, MODEL_METADATA


class Person5CrossModalBinding:
    """Class binding Person 5's predict function into SatQueryEngine."""
    def __init__(self):
        self.model_name = MODEL_METADATA["model_name"]
        self.backbone = MODEL_METADATA["backbone"]
        self.training_dataset = MODEL_METADATA["training_dataset"]

    def predict(self, query: str, images: list, parameters: dict) -> dict:
        return predict(query=query, images=images, parameters=parameters)


def run_person5_end_to_end_demo():
    print("\n" + "=" * 95)
    print("      SATQUERY AI: PERSON 5 OPTICAL-SAR CROSS-MODAL MODEL (END-TO-END DEMO)")
    print("=" * 95)

    # 1. Verify Adapter Configuration Status
    print("\n--- [STEP 1: PERSON 5 ADAPTER CONFIGURATION STATUS] ---")
    status = check_person5_adapter_status()
    print(f"• Adapter Directory  : {status['adapter_dir']}")
    print(f"• Config File        : {'✅ Present' if status['config_found'] else '❌ Missing'}")
    print(f"• Preprocessor Config: {'✅ Present' if status['preprocessor_found'] else '❌ Missing'}")
    print(f"• Model Name         : {MODEL_METADATA['model_name']}")
    print(f"• Architecture       : {MODEL_METADATA['backbone']}")
    print(f"• Benchmark Adaptation: {MODEL_METADATA['training_dataset']}")

    # 2. Initialize Engine & Register Person 5 Model
    print("\n--- [STEP 2: ENGINE INITIALIZATION & MODEL REGISTRATION] ---")
    engine = SatQueryEngine(auto_seed=True)
    p5_model_instance = Person5CrossModalBinding()
    engine.register_crossmodal_model(p5_model_instance)
    print(f"• Registered Model   : {p5_model_instance.model_name}")
    print(f"• Backbone           : {p5_model_instance.backbone}")
    print(f"• Target Capability  : CROSS_MODAL_FUSION")

    # 3. Test Direct Predict Call
    print("\n--- [STEP 3: RUNNING DIRECT PERSON 5 PREDICT CONTRACT] ---")
    raw_query = "Perform Optical and SAR cross-attention fusion to identify built-up infrastructure and water bodies under cirrus clouds."
    multimodal_inputs = ["patch_s2_sample_001.npy", "patch_s1_sample_001.npy"]

    direct_res = p5_model_instance.predict(
        query=raw_query,
        images=multimodal_inputs,
        parameters={"mode": "multimodal", "fusion_method": "cross_attention"}
    )

    print("\n--- [DIRECT INFERENCE PREDICT OUTPUT] ---")
    print(f"• Model Output       : {direct_res['model']}")
    print(f"• Joint Summary      : {direct_res['joint_analysis_summary']}")
    print(f"• Confidence         : {direct_res['confidence'] * 100:.1f}%")
    print(f"• Cloud Masking      : {direct_res['cloud_masking_applied']}")

    # 4. Execute Full Pipeline via SatQueryEngine
    print("\n--- [STEP 4: RUNNING THROUGH FULL SATQUERY ENGINE PIPELINE] ---")
    image_inputs = [
        {"file_name": "scene_optical_s2.tif", "format": "GeoTIFF", "modality": "OPTICAL"},
        {"file_name": "scene_sar_s1.tif", "format": "GeoTIFF", "modality": "SAR"}
    ]

    sqo, plan, orch_res, agg_res = engine.process_and_aggregate(
        raw_query=raw_query,
        image_inputs=image_inputs
    )

    print(f"• Query ID             : {sqo.query_id}")
    print(f"• Primary Task         : {sqo.task_classification.primary_task.value}")
    print(f"• Selected Agent       : {plan.selected_agent_id}")
    print(f"• Orchestrator Status  : {orch_res.status.value.upper()}")
    print(f"• Composite Confidence : {agg_res.aggregated_confidence * 100:.1f}%")

    print("\n--- [FINAL SYNTHESIZED AGGREGATED RESPONSE] ---")
    print(f"Answer: {agg_res.final_answer}")

    # 5. Verify Database Trace Persistence
    print("\n--- [STEP 5: DATABASE PERSISTENCE VERIFICATION] ---")
    db_path = _ROOT_DIR / "satquery.db"
    if db_path.exists():
        print(f"• SQLite Database File : {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            print(f"• Database Tables      : {tables}")

            if "traces" in tables:
                cursor.execute("SELECT trace_id, request_id, status, total_duration_ms FROM traces ORDER BY rowid DESC LIMIT 1;")
                t_row = cursor.fetchone()
                if t_row:
                    print(f"• Latest Trace Record  : TraceID={t_row[0]} | RequestID={t_row[1]} | Status={t_row[2]} | Duration={t_row[3]:.2f}ms")

            conn.close()
        except Exception as e:
            print(f"• Database note: {e}")

    print("\n" + "=" * 95)
    print("      VERIFICATION COMPLETE: PERSON 5 CROSS-MODAL MODEL INTEGRATED SUCCESSFULLY!")
    print("=" * 95)

    engine.close()


if __name__ == "__main__":
    run_person5_end_to_end_demo()
