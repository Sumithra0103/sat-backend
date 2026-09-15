"""
End-to-End Integration Tests for SatQuery AI.
Verifies the complete flow:
Natural Language Query + Geospatial Imagery
          ↓
Query Understanding (SQO)
          ↓
Agent Registry (Specialist Candidates)
          ↓
SatQuery Router (Multi-criteria Scoring)
          ↓
Auditable Execution Plan (Preprocessing -> Specialist -> Evidence Grounding -> Fallbacks)
"""

import sys
from pathlib import Path
import pytest

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from satquery_engine import SatQueryEngine
from satquery_routing import RoutingStatus
from query_understanding import TaskType


@pytest.fixture(scope="module")
def engine():
    eng = SatQueryEngine(cache_ttl_seconds=5.0)
    yield eng
    eng.close()


def test_scenario_1_single_image_captioning(engine):
    """
    Representative Query 1: 'Describe the land-cover and major objects visible in this image.'
    Adapted on BigEarthNet.txt / BigEarthNet-MM.
    """
    query = "Describe the land-cover and major objects visible in this image."
    images = [{
        "image_id": "img_cartosat_01",
        "file_name": "scene_cartosat2s_optical.tif",
        "format": "GeoTIFF",
        "modality": "OPTICAL",
        "sensor_type": "Cartosat-2S",
        "timestamp": "2026-01-10T10:30:00Z"
    }]

    sqo, plan = engine.process_query(query, images)

    # Verify Query Understanding
    assert sqo.task_classification.primary_task == TaskType.SINGLE_IMAGE_CAPTIONING
    assert sqo.input_compatibility.is_compatible is True
    assert sqo.ambiguity_report.is_ambiguous is False
    assert len(sqo.image_inputs) == 1

    # Verify Router Selection
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-caption-agent"
    assert plan.auditable_summary.selected_task == "scene_captioning"
    assert "bigearthnet" in plan.auditable_summary.primary_agent_name.lower() or plan.auditable_summary.routing_confidence > 0.8
    assert len(plan.steps) >= 3


def test_scenario_2_text_guided_grounding(engine):
    """
    Representative Query 2: 'Highlight the water body referred to in the query.'
    Benchmark: VRSBench.
    """
    query = "Highlight the water body referred to in the query."
    images = [{
        "image_id": "img_vrsbench_02",
        "file_name": "reservoir_patch.png",
        "format": "PNG",
        "modality": "OPTICAL",
        "sensor_type": "Public_Benchmark",
        "timestamp": "2025-11-05T14:15:00Z"
    }]

    sqo, plan = engine.process_query(query, images)

    # Verify Query Understanding
    assert sqo.task_classification.primary_task == TaskType.TEXT_GUIDED_GROUNDING
    assert "water body" in sqo.extracted_entities.target_classes
    assert sqo.input_compatibility.is_compatible is True

    # Verify Router Selection
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-grounding-agent"
    assert plan.auditable_summary.selected_task == "region_grounding"


def test_scenario_3_bitemporal_change_detection(engine):
    """
    Representative Query 3: 'What changed between these two dates, and where did the change occur?'
    Benchmark: CDVQA / LEVIR-CD.
    """
    query = "What changed between these two dates, and where did the change occur?"
    images = [
        {
            "image_id": "img_t1",
            "file_name": "flood_pre_2024.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "sensor_type": "Cartosat-2S",
            "timestamp": "2024-06-15T09:00:00Z"
        },
        {
            "image_id": "img_t2",
            "file_name": "flood_post_2025.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "sensor_type": "Cartosat-2S",
            "timestamp": "2025-06-15T09:00:00Z"
        }
    ]

    sqo, plan = engine.process_query(query, images)

    # Verify Query Understanding
    assert sqo.task_classification.primary_task == TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING
    assert sqo.input_compatibility.is_compatible is True
    assert sqo.input_compatibility.image_count == 2

    # Verify Router Selection
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-change-agent"
    assert plan.requirements.temporal_structure == "BITEMPORAL_PAIR"
    assert plan.requirements.required_modality == "bitemporal_optical"


def test_scenario_4_cross_modal_optical_sar(engine):
    """
    Representative Query 4: 'Use the optical and SAR images together to identify built-up and water-covered regions.'
    Dataset: BigEarthNet-MM & ISRO/SAC Cartosat-2S + RISAT SAR pair.
    """
    query = "Use the optical and SAR images together to identify built-up and water-covered regions."
    images = [
        {
            "image_id": "img_opt",
            "file_name": "cartosat2s_optical.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "sensor_type": "Cartosat-2S",
            "is_co_registered": True
        },
        {
            "image_id": "img_sar",
            "file_name": "risat_sar_cband.tif",
            "format": "GeoTIFF",
            "modality": "SAR",
            "sensor_type": "RISAT",
            "is_co_registered": True
        }
    ]

    sqo, plan = engine.process_query(query, images)

    # Verify Query Understanding
    assert sqo.task_classification.primary_task == TaskType.CROSS_MODAL_OPTICAL_SAR_FUSION
    assert sqo.input_compatibility.is_compatible is True
    assert sqo.input_compatibility.temporal_structure == "CROSS_MODAL"

    # Verify Router Selection
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-crossmodal-agent"
    assert plan.requirements.required_modality == "cross_modal_optical_sar"
    assert plan.requirements.requires_cloud_resilience is True


def test_scenario_5_multi_turn_coreference(engine):
    """
    Representative Query 5 (Multi-Turn):
    Turn 1: 'Highlight the built-up area in these images.'
    Turn 2: 'Has it increased, decreased, or remained unchanged?'
    """
    session_id = "sess_integration_005"
    turn1_query = "Highlight the built-up area in these images."
    turn1_images = [
        {"image_id": "t1", "file_name": "urban_t1.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2020-01-01"},
        {"image_id": "t2", "file_name": "urban_t2.tif", "format": "GeoTIFF", "modality": "OPTICAL", "timestamp": "2026-01-01"}
    ]

    sqo1, plan1 = engine.process_query(turn1_query, turn1_images, session_id=session_id)
    assert "built-up area" in sqo1.extracted_entities.target_classes

    turn2_query = "Has it increased, decreased, or remained unchanged?"
    sqo2, plan2 = engine.process_query(turn2_query, image_inputs=[], session_id=session_id)

    # Verify context inheritance & coreference resolution
    assert sqo2.extracted_entities.is_coreference_resolved is True
    assert sqo2.extracted_entities.inherited_from_turn == 1
    assert sqo2.input_compatibility.image_count == 2
    assert plan2.routing_status == RoutingStatus.SUCCESS
    assert plan2.selected_agent_id == "rs-change-agent"


def test_scenario_6_ambiguity_rejection(engine):
    """
    Rejection test: Bi-temporal change query submitted with only 1 image.
    System must reject early and provide actionable remediation.
    """
    query = "What changed between these two dates, and where did the change occur?"
    images = [
        {"image_id": "solo", "file_name": "single_optical.tif", "format": "GeoTIFF", "modality": "OPTICAL"}
    ]

    sqo, plan = engine.process_query(query, images)

    assert sqo.input_compatibility.is_compatible is False
    assert sqo.ambiguity_report.is_ambiguous is True
    assert plan.routing_status == RoutingStatus.AMBIGUOUS_REJECTED
    assert len(plan.rejection_errors) > 0
    assert "requires a bi-temporal image pair" in plan.rejection_errors[0]
