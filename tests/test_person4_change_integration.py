"""
Test Suite - Person 4 Change Analysis Lead Model Integration Verification
"""

import sys
from pathlib import Path
import pytest

_ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR / "agent registry"), str(_ROOT_DIR / "routing"), str(_ROOT_DIR / "query understanding"), str(_ROOT_DIR / "trace")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from satquery_engine import SatQueryEngine
from execution.models.specialist_models import RSChangeDetectionModel, register_change_model, get_registered_change_model


class DummyPerson4Model:
    """Mock implementation of Person 4 (Change Analysis Lead) model contract."""
    def __init__(self):
        self.model_name = "Person4-SiamUnet-CDVQA"
        self.backbone = "SiamUnet-ResNet50"
        self.training_dataset = "CDVQA, LEVIR-CD"

    def predict(self, query: str, images: list, parameters: dict) -> dict:
        return {
            "changed_regions": 22,
            "change_map": "p4_bitemporal_change_mask.png",
            "change_trend": "increased",
            "change_description": "Person 4 Siamese U-Net detected 22 building construction changes covering 3.12 sq km.",
            "changed_area_km2": 3.12,
            "confidence": 0.96
        }


def test_person4_model_registration():
    """Verifies that Person 4's model can be bound and retrieved."""
    p4_model = DummyPerson4Model()
    register_change_model(p4_model)
    
    assert get_registered_change_model() is p4_model
    
    model_wrapper = RSChangeDetectionModel()
    res = model_wrapper.predict("What changed between these dates?", batch=None, parameters={})
    
    assert res["changed_regions"] == 22
    assert res["changed_area_km2"] == 3.12
    assert res["confidence"] == 0.96
    assert "Person 4 Siamese U-Net" in res["change_description"]


def test_satquery_engine_person4_end_to_end():
    """Verifies end-to-end bi-temporal change query execution in SatQueryEngine with Person 4 model."""
    engine = SatQueryEngine(auto_seed=True)
    p4_model = DummyPerson4Model()
    engine.register_change_model(p4_model)
    
    bitemporal_images = [
        {"file_name": "t1_before.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2023-01-01"},
        {"file_name": "t2_after.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2024-01-01"}
    ]
    
    sqo, plan, orch_res, agg_res = engine.process_and_aggregate(
        raw_query="What changed between T1 and T2, and has the built-up area increased?",
        image_inputs=bitemporal_images
    )
    
    assert sqo.input_compatibility.temporal_structure.upper() in ["BI_TEMPORAL", "MULTI_TEMPORAL"]
    assert agg_res.final_answer is not None
    assert agg_res.aggregated_confidence > 0.8
    assert len(agg_res.visual_evidence_urls) > 0


    engine.close()


if __name__ == "__main__":
    print("Running Person 4 Model Integration Tests...")
    test_person4_model_registration()
    print("[OK] test_person4_model_registration passed.")
    test_satquery_engine_person4_end_to_end()
    print("[OK] test_satquery_engine_person4_end_to_end passed.")
    print("\nALL PERSON 4 CHANGE ANALYSIS INTEGRATION TESTS PASSED SUCCESSFULLY!")


