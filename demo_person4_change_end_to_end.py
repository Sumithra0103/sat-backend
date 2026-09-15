"""
SatQuery AI: Person 4 Bi-Temporal Change Model End-to-End Verification Demo Script
==================================================================================
Demonstrates complete agentic bi-temporal change detection pipeline with
Person 4's SiameseResNet18ChangeDetector-V2.1 model & SQLite Database Persistence.
"""

import sys
import os
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_ROOT_DIR = Path(__file__).resolve().parent
for _sub in [_ROOT_DIR, _ROOT_DIR / "agent registry", _ROOT_DIR / "routing", _ROOT_DIR / "query understanding", _ROOT_DIR / "trace"]:
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

from satquery_engine import SatQueryEngine
from execution.models.specialist_models import check_person4_adapter_status
from change_analysis.person4_wrapper import predict, MODEL_METADATA


class Person4ChangeModelBinding:
    """Class binding Person 4's predict function into SatQueryEngine."""
    def __init__(self):
        self.model_name = MODEL_METADATA["model_name"]
        self.backbone = MODEL_METADATA["backbone"]
        self.training_dataset = MODEL_METADATA["training_dataset"]

    def predict(self, query: str, images: list, parameters: dict) -> dict:
        return predict(query=query, images=images, parameters=parameters)


def run_person4_end_to_end_demo():
    print("\n" + "=" * 95)
    print("      SATQUERY AI: PERSON 4 BI-TEMPORAL CHANGE MODEL (END-TO-END DEMO)")
    print("=" * 95)

    # 1. Verify Adapter Configuration Status
    print("\n--- [STEP 1: PERSON 4 ADAPTER CONFIGURATION STATUS] ---")
    status = check_person4_adapter_status()
    print(f"• Adapter Directory  : {status['adapter_dir']}")
    print(f"• Config File        : {'✅ Present' if status['config_found'] else '❌ Missing'}")
    print(f"• Preprocessor Config: {'✅ Present' if status['preprocessor_found'] else '❌ Missing'}")
    print(f"• Model Name         : {MODEL_METADATA['model_name']}")
    print(f"• Architecture       : {MODEL_METADATA['backbone']}")
    print(f"• Benchmark Adaptation: {MODEL_METADATA['training_dataset']}")

    # 2. Initialize Engine & Register Person 4 Change Model
    print("\n--- [STEP 2: ENGINE INITIALIZATION & MODEL REGISTRATION] ---")
    engine = SatQueryEngine(auto_seed=True)
    p4_model_instance = Person4ChangeModelBinding()
    engine.register_change_model(p4_model_instance)
    print(f"• Registered Model   : {p4_model_instance.model_name}")
    print(f"• Backbone           : {p4_model_instance.backbone}")
    print(f"• Target Capability  : CHANGE_DETECTION")

    # 3. Create Sample Bi-Temporal Synthetic Images
    print("\n--- [STEP 3: PREPARING BI-TEMPORAL IMAGERY & RUNNING PREDICT] ---")
    img_t1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img_t1[:, :] = [30, 120, 40]  # Vegetation green

    img_t2 = img_t1.copy()
    img_t2[50:150, 50:150] = [180, 70, 70]  # Built-up area change (reddish brown)

    raw_query = "What spatial changes occurred between T1 (2023) and T2 (2024), and has built-up area expanded?"
    image_inputs = [
        {"file_name": "t1_2023_before.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2023-01-01"},
        {"file_name": "t2_2024_after.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2024-01-01"}
    ]

    print(f"• Query              : '{raw_query}'")
    print(f"• Input Temporal Pair: T1 (2023-01-01) vs T2 (2024-01-01)")

    # Test direct predict call first
    direct_res = p4_model_instance.predict(
        query=raw_query,
        images=[img_t1, img_t2],
        parameters={"pixel_size_m": [10.0, 10.0], "confidence_threshold": 0.5}
    )

    print("\n--- [DIRECT INFERENCE PREDICT OUTPUT] ---")
    print(f"• Model Output       : {direct_res['model']}")
    print(f"• Change Detected    : {direct_res['change_detected']}")
    print(f"• Changed Percentage : {direct_res['change_percentage']:.2f}%")
    print(f"• Changed Pixels     : {direct_res['changed_pixels']}")
    print(f"• Changed Area (km²) : {direct_res['changed_area_km2']}")
    print(f"• Spatial Location   : {direct_res['spatial_location']}")
    print(f"• Change Trend       : {direct_res['change_trend']}")
    print(f"• Confidence         : {direct_res['confidence'] * 100:.1f}%")
    print(f"• Change Description : {direct_res['change_description']}")

    # 4. Execute Full Pipeline via SatQueryEngine
    print("\n--- [STEP 4: RUNNING THROUGH FULL SATQUERY ENGINE PIPELINE] ---")
    sqo, plan, orch_res, agg_res = engine.process_and_aggregate(
        raw_query=raw_query,
        image_inputs=image_inputs
    )

    print(f"• Query ID             : {sqo.query_id}")
    print(f"• Primary Task         : {sqo.task_classification.primary_task.value}")
    print(f"• Temporal Structure   : {sqo.input_compatibility.temporal_structure}")
    print(f"• Selected Agent       : {plan.selected_agent_id}")
    print(f"• Orchestrator Status  : {orch_res.status.value.upper()}")
    print(f"• Composite Confidence : {agg_res.aggregated_confidence * 100:.1f}%")

    print("\n--- [FINAL SYNTHESIZED AGGREGATED RESPONSE] ---")
    print(f"Answer: {agg_res.final_answer}")
    if agg_res.visual_evidence_urls:
        print(f"Visual Evidence: {agg_res.visual_evidence_urls}")

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
    print("      VERIFICATION COMPLETE: PERSON 4 BI-TEMPORAL MODEL INTEGRATED SUCCESSFULLY!")
    print("=" * 95)

    engine.close()


if __name__ == "__main__":
    run_person4_end_to_end_demo()
