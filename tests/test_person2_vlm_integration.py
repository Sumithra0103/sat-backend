"""
Test Suite - Person 2 VLM / BigEarthNet Lead Model Integration Verification
"""

import sys
from pathlib import Path
import pytest

_ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR / "agent registry"), str(_ROOT_DIR / "routing"), str(_ROOT_DIR / "query understanding"), str(_ROOT_DIR / "trace")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from satquery_engine import SatQueryEngine
from execution.models.specialist_models import RSVQAModel, register_vlm_model, get_registered_vlm_model


class DummyPerson2Model:
    """Mock implementation of Person 2 (VLM / BigEarthNet Lead) model contract."""
    def __init__(self):
        self.model_name = "Person2-RS-VLM-BigEarthNet"
        self.backbone = "Qwen2VL-RS-LoRA"
        self.training_dataset = "BigEarthNet.txt, BigEarthNet-MM"

    def predict(self, query: str, images: list, parameters: dict) -> dict:
        return {
            "answer": "Person 2 VLM analysis: The optical scene captures semi-urban terrain with 45.2% agricultural fields.",
            "confidence": 0.95,
            "landcover_stats": {
                "agricultural_fields": 45.2,
                "forest_canopy": 25.1,
                "water_bodies": 15.3,
                "built_up": 14.4
            },
            "domain_adaptation": self.training_dataset
        }


def test_person2_model_registration():
    """Verifies that Person 2's model can be bound and retrieved."""
    p2_model = DummyPerson2Model()
    register_vlm_model(p2_model)

    assert get_registered_vlm_model() is p2_model

    model_wrapper = RSVQAModel()
    res = model_wrapper.predict("Describe the land-cover in this image", batch=None, parameters={})

    assert res["confidence"] == 0.95
    assert "Person 2 VLM" in res["answer"]
    assert res["landcover_stats"]["agricultural_fields"] == 45.2


def test_satquery_engine_person2_end_to_end():
    """Verifies end-to-end VQA query execution in SatQueryEngine with Person 2 VLM model."""
    engine = SatQueryEngine(auto_seed=True)
    p2_model = DummyPerson2Model()
    engine.register_vlm_model(p2_model)

    single_image = [
        {"file_name": "scene_optical.tif", "format": "GeoTIFF", "modality": "OPTICAL"}
    ]

    sqo, plan, orch_res, agg_res = engine.process_and_aggregate(
        raw_query="Describe the land-cover and major objects visible in this image.",
        image_inputs=single_image
    )

    assert agg_res.final_answer is not None
    assert agg_res.aggregated_confidence > 0.5


    engine.close()


if __name__ == "__main__":
    print("Running Person 2 VLM Model Integration Tests...")
    test_person2_model_registration()
    print("[OK] test_person2_model_registration passed.")
    test_satquery_engine_person2_end_to_end()
    print("[OK] test_satquery_engine_person2_end_to_end passed.")
    print("\nALL PERSON 2 VLM INTEGRATION TESTS PASSED SUCCESSFULLY!")
