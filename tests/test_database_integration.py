"""
Unit & Integration Tests for SatQuery AI PostgreSQL Database Layer.
"""

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if "agent registry" not in p and p != str(_ROOT_DIR / "routing")]
sys.path.insert(0, str(_ROOT_DIR))



import pytest
import os
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, check_db_health, init_db
from database.models import AgentModel, QuerySessionModel, TraceModel, TraceSpanModel, EvidenceArtifactModel, QueryResultModel
from database.repositories import AgentRepository, QueryRepository, TraceRepository



# Test database engine using SQLite in-memory for testing isolation
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh, clean in-memory database session for each test function."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_agent_repository_crud(db_session):
    """Verifies agent registration, retrieval, and heartbeat updates."""
    repo = AgentRepository(db_session)
    agent_id = f"test_agent_{uuid.uuid4().hex[:6]}"

    agent_data = {
        "agent_id": agent_id,
        "name": "Optical Change Detection Agent",
        "version": "1.2.0",
        "description": "Specialized model for detecting land cover changes.",
        "capabilities": ["change_detection"],
        "input_modalities": ["OPTICAL"],
        "supported_formats": ["GeoTIFF"],
        "endpoint_url": "http://localhost:9001/detect",
        "status": "active"
    }

    # 1. Register Agent
    created = repo.register_agent(agent_data)
    assert created.agent_id == agent_id
    assert created.name == "Optical Change Detection Agent"
    assert "change_detection" in created.capabilities

    # 2. Retrieve Agent
    fetched = repo.get_agent_by_id(agent_id)
    assert fetched is not None
    assert fetched.version == "1.2.0"

    # 3. List Agents
    all_agents = repo.get_all_agents(status="active")
    assert any(a.agent_id == agent_id for a in all_agents)

    # 4. Update Heartbeat
    success = repo.update_heartbeat(agent_id)
    assert success is True


def test_query_repository_pipeline_flow(db_session):
    """Verifies end-to-end query session, plan update, evidence artifact, and query result persistence."""
    repo = QueryRepository(db_session)
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # 1. Create Query Session
    session = repo.create_query_session(
        raw_query="Detect flooding in Assam using Sentinel-1 SAR",
        image_inputs=[{"file_name": "assam_sar.tif", "modality": "SAR"}],
        session_id=session_id
    )
    assert session.session_id == session_id
    assert session.status == "PENDING"

    # 2. Update Plan & SQO
    sqo_data = {"task": "FLOOD_MAPPING", "confidence": 0.95}
    plan_data = {"selected_agent": "sar_flood_agent", "status": "OPTIMAL"}
    updated = repo.update_query_plan(session_id, sqo=sqo_data, execution_plan=plan_data, status="PROCESSING")
    assert updated.status == "PROCESSING"
    assert updated.sqo["task"] == "FLOOD_MAPPING"

    # 3. Save Evidence Artifact
    artifact = repo.save_evidence_artifact(
        session_id=session_id,
        agent_id="sar_flood_agent",
        modality="SAR",
        evidence_type="HEATMAP",
        file_path="/evidence_artifacts/flood_map.png",
        confidence_score=0.92
    )
    assert artifact.session_id == session_id
    assert artifact.evidence_type == "HEATMAP"

    # 4. Save Query Result
    result = repo.save_query_result(
        session_id=session_id,
        consensus_score=0.94,
        synthesized_text="Severe flooding detected covering approximately 140 sq km in Brahmaputra basin.",
        evidence_summary=[{"file_path": artifact.file_path}]
    )
    assert result.session_id == session_id
    assert "flooding" in result.synthesized_text

    # Verify session marked COMPLETED
    completed_session = repo.get_query_session(session_id)
    assert completed_session.status == "COMPLETED"


def test_trace_repository(db_session):
    """Verifies execution trace and span persistence."""
    repo = TraceRepository(db_session)
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    trace_id = f"trc_{uuid.uuid4().hex[:8]}"

    # 1. Create Root Trace
    trace = repo.create_trace(session_id=session_id, request_id="req_123", query="Analyze deforestation", trace_id=trace_id)
    assert trace.trace_id == trace_id
    assert trace.status == "INITIALIZED"

    # 2. Add Trace Span
    span = repo.save_span(
        trace_id=trace_id,
        name="Query Understanding",
        component="query_understanding",
        start_time=100.0,
        end_time=105.0,
        duration_ms=5.0,
        status="SUCCESS",
        confidence=0.98
    )
    assert span.trace_id == trace_id
    assert span.component == "query_understanding"

    # 3. Finalize Trace Status
    finalized = repo.update_trace_status(trace_id=trace_id, status="SUCCESS", total_duration_ms=120.0, metrics={"total_spans": 1})
    assert finalized.status == "SUCCESS"
    assert finalized.total_duration_ms == 120.0


def test_db_health_check():
    """Verifies database health check helper output."""
    health = check_db_health()
    assert "status" in health
    assert "dialect" in health
