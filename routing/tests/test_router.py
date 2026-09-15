"""
Unit and Integration Tests for SatQuery AI Agent Router.
"""

import sys
from pathlib import Path
import pytest

# Ensure routing directory and query understanding package are on sys.path
_ROUTING_DIR = Path(__file__).resolve().parent.parent
_ROOT_DIR = _ROUTING_DIR.parent
_QU_DIR = _ROOT_DIR / "query understanding"

for p in [str(_ROUTING_DIR), str(_ROOT_DIR), str(_QU_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from database.db import init_db, SessionLocal
from services.registry_service import AgentRegistryService
from client.registry_client import AgentRegistryClient
from schemas.agent_schema import (
    AgentRegisterRequest,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)
from query_understanding import QueryUnderstandingPipeline
from satquery_routing import SatQueryRouter, RoutingStatus


@pytest.fixture(scope="module")
def db_session():
    init_db()
    db = SessionLocal()
    AgentRegistryService.seed_mock_agents(db)
    yield db
    db.close()


@pytest.fixture
def router(db_session):
    return SatQueryRouter(db_session=db_session, cache_ttl_seconds=10.0)


@pytest.fixture
def qu_pipeline():
    return QueryUnderstandingPipeline()


def test_single_image_captioning(router, qu_pipeline):
    query = "Describe the land-cover and major objects visible in this image."
    images = [{"image_id": "img1", "format": "GeoTIFF", "modality": "OPTICAL"}]
    sqo = qu_pipeline.process_query(query, images)
    
    plan = router.route(sqo)
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-caption-agent"
    assert len(plan.steps) == 3
    assert plan.steps[1].agent_id == "rs-caption-agent"
    assert plan.auditable_summary.selected_task == "scene_captioning"


def test_region_grounding(router, qu_pipeline):
    query = "Highlight the water body referred to in the query."
    images = [{"image_id": "img1", "format": "PNG", "modality": "OPTICAL"}]
    sqo = qu_pipeline.process_query(query, images)
    
    plan = router.route(sqo)
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-grounding-agent"
    assert plan.auditable_summary.selected_task == "region_grounding"


def test_bitemporal_change_detection(router, qu_pipeline):
    query = "What changed between these two dates, and where did the change occur?"
    images = [
        {"image_id": "t1", "format": "GeoTIFF", "modality": "OPTICAL"},
        {"image_id": "t2", "format": "GeoTIFF", "modality": "OPTICAL"}
    ]
    sqo = qu_pipeline.process_query(query, images)
    
    plan = router.route(sqo)
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-change-agent"
    assert plan.requirements.temporal_structure == "BITEMPORAL_PAIR"


def test_cross_modal_fusion(router, qu_pipeline):
    query = "Use the optical and SAR images together to identify built-up and water-covered regions."
    images = [
        {"image_id": "opt", "format": "GeoTIFF", "modality": "OPTICAL"},
        {"image_id": "sar", "format": "GeoTIFF", "modality": "SAR"}
    ]
    sqo = qu_pipeline.process_query(query, images)
    
    plan = router.route(sqo)
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-crossmodal-agent"
    assert plan.requirements.required_modality == "cross_modal_optical_sar"


def test_ambiguity_rejection(router, qu_pipeline):
    # Change query with only 1 image attached
    query = "What changed between these two dates, and where did the change occur?"
    images = [{"image_id": "solo", "format": "GeoTIFF", "modality": "OPTICAL"}]
    sqo = qu_pipeline.process_query(query, images)
    
    plan = router.route(sqo)
    assert plan.routing_status == RoutingStatus.AMBIGUOUS_REJECTED
    assert len(plan.rejection_errors) > 0


def test_fallback_assignment(router, db_session, qu_pipeline):
    # Register a secondary VQA agent so there are two candidates for VQA
    custom_vqa = AgentRegisterRequest(
        agent_id="rs-vqa-secondary",
        name="Secondary VQA Agent",
        version="1.0.0",
        description="Backup lightweight VQA agent for failover.",
        capabilities=[AgentCapability.SINGLE_IMAGE_VQA],
        input_modalities=[InputModality.OPTICAL],
        supported_formats=[InputFormat.GEOTIFF, InputFormat.PNG],
        endpoint_url="http://localhost:8010/v1/vqa_backup",
        metadata={"training_dataset": "RSVQA", "latency_ms": 150},
        status=AgentStatus.ACTIVE
    )
    try:
        AgentRegistryService.register_agent(db_session, custom_vqa)
    except Exception:
        pass

    router.cache.invalidate()

    query = "What is the primary land use in this scene?"
    images = [{"image_id": "img1", "format": "GeoTIFF", "modality": "OPTICAL"}]
    sqo = qu_pipeline.process_query(query, images)

    plan = router.route(sqo)
    assert plan.routing_status == RoutingStatus.SUCCESS
    assert plan.selected_agent_id == "rs-vqa-agent"
    assert len(plan.fallback_routes) > 0
    fallback_ids = [fb.fallback_agent_id for fb in plan.fallback_routes]
    assert any("rs-vqa-" in fid for fid in fallback_ids)
