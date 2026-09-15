"""
Unit & Integration Tests for SatQuery AI Query Understanding System.
Can be executed with pytest or directly with python.
"""

import sys
from pathlib import Path
import pytest

# Ensure parent query_understanding directory is in sys.path
_PKG_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from query_understanding import (
    QueryUnderstandingPipeline,
    TaskType,
    Modality,
    ImageFormat
)


@pytest.fixture
def pipeline():
    return QueryUnderstandingPipeline()


def test_single_image_captioning(pipeline):
    query = "Describe the land-cover and major objects visible in this image."
    images = [{
        "image_id": "img_1",
        "file_name": "scene.tif",
        "format": "GeoTIFF",
        "modality": "OPTICAL"
    }]
    sqo = pipeline.process_query(query, images)
    assert sqo.task_classification.primary_task == TaskType.SINGLE_IMAGE_CAPTIONING
    assert sqo.input_compatibility.is_compatible is True
    assert sqo.ambiguity_report.is_ambiguous is False
    assert sqo.routing_metadata.target_agent_family == "CaptioningSpecialist"
    assert len(sqo.image_inputs) == 1
    assert sqo.image_inputs[0]["format"] == "GeoTIFF"


def test_text_guided_grounding(pipeline):
    query = "Highlight the water body referred to in the query."
    images = [{
        "image_id": "img_1",
        "file_name": "lake.png",
        "format": "PNG",
        "modality": "OPTICAL"
    }]
    sqo = pipeline.process_query(query, images)
    assert sqo.task_classification.primary_task == TaskType.TEXT_GUIDED_GROUNDING
    assert "water body" in sqo.extracted_entities.target_classes
    assert sqo.routing_metadata.target_agent_family == "GroundingSpecialist"
    assert len(sqo.image_inputs) == 1


def test_bitemporal_change_understanding(pipeline):
    query = "What changed between these two dates, and where did the change occur?"
    images = [
        {"image_id": "t1", "file_name": "t1.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2024-01-01"},
        {"image_id": "t2", "file_name": "t2.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2025-01-01"}
    ]
    sqo = pipeline.process_query(query, images)
    assert sqo.task_classification.primary_task == TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING
    assert sqo.input_compatibility.temporal_structure == "BI_TEMPORAL"
    assert sqo.input_compatibility.is_compatible is True
    assert len(sqo.image_inputs) == 2


def test_cross_modal_fusion(pipeline):
    query = "Use the optical and SAR images together to identify built-up and water-covered regions."
    images = [
        {"image_id": "opt", "file_name": "opt.tif", "format": "GeoTIFF", "modality": "OPTICAL"},
        {"image_id": "sar", "file_name": "sar.tif", "format": "GeoTIFF", "modality": "SAR"}
    ]
    sqo = pipeline.process_query(query, images)
    assert sqo.task_classification.primary_task == TaskType.CROSS_MODAL_OPTICAL_SAR_FUSION
    assert sqo.input_compatibility.temporal_structure == "CROSS_MODAL"
    assert sqo.input_compatibility.is_compatible is True
    assert len(sqo.image_inputs) == 2


def test_multi_turn_history_and_coreference(pipeline):
    session_id = "test_sess_001"
    query1 = "Highlight the built-up area in these images."
    images1 = [
        {"image_id": "img1", "file_name": "img1.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2020-01-01"},
        {"image_id": "img2", "file_name": "img2.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2026-01-01"}
    ]
    sqo1 = pipeline.process_query(query1, images1, session_id=session_id)
    assert "built-up area" in sqo1.extracted_entities.target_classes

    query2 = "Has it increased, decreased, or remained unchanged?"
    sqo2 = pipeline.process_query(query2, image_inputs=[], session_id=session_id)

    assert sqo2.extracted_entities.is_coreference_resolved is True
    assert sqo2.extracted_entities.inherited_from_turn == 1
    assert sqo2.task_classification.primary_task == TaskType.BI_TEMPORAL_CHANGE_QUANTIFICATION
    assert sqo2.input_compatibility.image_count == 2
    assert len(sqo2.image_inputs) == 2


def test_ambiguity_and_incompatibility_detection(pipeline):
    query = "What changed between these two dates, and where did the change occur?"
    images = [
        {"image_id": "only_one", "file_name": "solo.tif", "format": "GeoTIFF", "modality": "OPTICAL"}
    ]
    sqo = pipeline.process_query(query, images)
    assert sqo.input_compatibility.is_compatible is False
    assert sqo.input_compatibility.status == "INCOMPATIBLE"
    assert sqo.ambiguity_report.is_ambiguous is True
    assert len(sqo.ambiguity_report.issues) > 0


if __name__ == "__main__":
    p = QueryUnderstandingPipeline()
    test_single_image_captioning(p)
    test_text_guided_grounding(p)
    test_bitemporal_change_understanding(p)
    test_cross_modal_fusion(p)
    test_multi_turn_history_and_coreference(p)
    test_ambiguity_and_incompatibility_detection(p)
    print("\nALL 6 UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
