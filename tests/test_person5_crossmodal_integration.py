"""
Test Suite - Person 5 Optical-SAR Cross-Modal Fusion Lead Integration Verification
"""

import sys
from pathlib import Path
import pytest

_ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR / "agent registry"), str(_ROOT_DIR / "routing"), str(_ROOT_DIR / "query understanding"), str(_ROOT_DIR / "trace")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from satquery_engine import SatQueryEngine
from execution.models.specialist_models import RSCrossModalFusionModel, register_crossmodal_model, get_registered_crossmodal_model
from fusion.person5_wrapper import predict, MODEL_METADATA


class DummyPerson5Model:
    """Mock implementation of Person 5 (Cross-Modal Fusion Lead) model contract."""
    def __init__(self):
        self.model_name = MODEL_METADATA["model_name"]
        self.backbone = MODEL_METADATA["backbone"]
        self.training_dataset = MODEL_METADATA["training_dataset"]

    def predict(self, query: str, images: list, parameters: dict) -> dict:
        return predict(query=query, images=images, parameters=parameters)


def test_person5_model_registration():
    """Verifies that Person 5's model can be bound and retrieved."""
    p5_model = DummyPerson5Model()
    register_crossmodal_model(p5_model)

    assert get_registered_crossmodal_model() is p5_model

    model_wrapper = RSCrossModalFusionModel()
    res = model_wrapper.predict("Analyze Optical and SAR joint scene.", batch=None, parameters={})

    assert "Optical" in res["joint_analysis_summary"]
    assert res["confidence"] > 0.90


def test_satquery_engine_person5_end_to_end():
    """Verifies end-to-end optical+SAR cross-modal query execution in SatQueryEngine with Person 5 model."""
    engine = SatQueryEngine(auto_seed=True)
    p5_model = DummyPerson5Model()
    engine.register_crossmodal_model(p5_model)

    multimodal_images = [
        {"file_name": "scene_optical_s2.tif", "format": "GeoTIFF", "modality": "OPTICAL"},
        {"file_name": "scene_sar_s1.tif", "format": "GeoTIFF", "modality": "SAR"}
    ]

    sqo, plan, orch_res, agg_res = engine.process_and_aggregate(
        raw_query="Use co-registered Optical and SAR images to classify built-up structures and water bodies under cloud shadow.",
        image_inputs=multimodal_images
    )

    assert sqo.input_compatibility.temporal_structure.upper() in ["CROSS_MODAL", "SINGLE_DATE"]
    assert agg_res.final_answer is not None
    assert agg_res.aggregated_confidence > 0.75
    assert len(agg_res.visual_evidence_urls) > 0

    engine.close()


if __name__ == "__main__":
    print("Running Person 5 Model Integration Tests...")
    test_person5_model_registration()
    print("[OK] test_person5_model_registration passed.")
    test_satquery_engine_person5_end_to_end()
    print("[OK] test_satquery_engine_person5_end_to_end passed.")
    print("\nALL PERSON 5 CROSS-MODAL INTEGRATION TESTS PASSED SUCCESSFULLY!")
