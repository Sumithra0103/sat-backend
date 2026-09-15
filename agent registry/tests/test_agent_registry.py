"""
Comprehensive Test Suite for SatQuery AI Agent Registry (Steps 1 through 15).
Tests Schema Validation, Database Persistence, REST APIs, Capability Search,
Health Telemetry, Security Middleware, Exception Handlers, and Router Client SDK.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database.db import Base, get_db
from schemas.agent_schema import (
    AgentRegisterRequest,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)
from client.registry_client import AgentRegistryClient

# Use StaticPool to ensure all connections in SQLite in-memory DB share the same instance
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

ADMIN_HEADERS = {"X-API-Key": "satquery-admin-key-2026"}
ROUTER_HEADERS = {"X-API-Key": "satquery-router-key-2026"}


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


# --- 1. Schema Validation Tests ---

def test_agent_schema_valid():
    req = AgentRegisterRequest(
        agent_id="test-rs-agent",
        name="Test RS Agent",
        version="1.0.0",
        description="A test remote sensing specialist agent.",
        capabilities=[AgentCapability.SINGLE_IMAGE_VQA],
        input_modalities=[InputModality.OPTICAL],
        supported_formats=[InputFormat.GEOTIFF],
        endpoint_url="http://localhost:9000/v1/test",
        parameters_schema={"type": "object"},
        metadata={"backbone": "resnet50"}
    )
    assert req.agent_id == "test-rs-agent"
    assert req.capabilities == [AgentCapability.SINGLE_IMAGE_VQA]


def test_agent_schema_invalid_url():
    with pytest.raises(ValueError):
        AgentRegisterRequest(
            agent_id="test-agent",
            name="Test Agent",
            version="1.0.0",
            description="Invalid URL test description",
            capabilities=[AgentCapability.SINGLE_IMAGE_VQA],
            input_modalities=[InputModality.OPTICAL],
            supported_formats=[InputFormat.GEOTIFF],
            endpoint_url="ftp://invalid-url.com"
        )


# --- 2. API Registration & Security Tests ---

def test_register_agent_unauthorized():
    payload = {
        "agent_id": "test-vqa-agent",
        "name": "RS VQA Agent",
        "version": "1.0.0",
        "description": "Domain-adapted visual question answering agent.",
        "capabilities": ["single_image_vqa"],
        "input_modalities": ["optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8001/v1/vqa"
    }
    # No header -> 401
    response = client.post("/agents/register", json=payload)
    assert response.status_code == 401

    # Invalid header -> 403
    response = client.post("/agents/register", json=payload, headers={"X-API-Key": "invalid-key"})
    assert response.status_code == 403


def test_register_agent_success():
    payload = {
        "agent_id": "rs-vqa-agent",
        "name": "SatQuery RS-VQA Specialist",
        "version": "1.2.0",
        "description": "Domain-adapted remote sensing visual question answering agent.",
        "capabilities": ["single_image_vqa", "scene_captioning"],
        "input_modalities": ["optical", "sar"],
        "supported_formats": ["geotiff", "png"],
        "endpoint_url": "http://localhost:8001/v1/vqa/execute",
        "parameters_schema": {"confidence_threshold": 0.7},
        "metadata": {"training_dataset": "BigEarthNet.txt"}
    }
    response = client.post("/agents/register", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 201
    data = response.json()
    assert data["agent_id"] == "rs-vqa-agent"
    assert data["status"] == "active"
    assert "single_image_vqa" in data["capabilities"]


def test_register_duplicate_agent_409():
    payload = {
        "agent_id": "rs-dup-agent",
        "name": "Duplicate Agent",
        "version": "1.0.0",
        "description": "Test duplicate registration constraint enforcement.",
        "capabilities": ["single_image_vqa"],
        "input_modalities": ["optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8001/v1/vqa"
    }
    res1 = client.post("/agents/register", json=payload, headers=ADMIN_HEADERS)
    assert res1.status_code == 201

    res2 = client.post("/agents/register", json=payload, headers=ADMIN_HEADERS)
    assert res2.status_code == 409
    assert res2.json()["error_code"] == "AGENT_ALREADY_EXISTS"


# --- 3. Discovery & Capability-Based Search Tests ---

def test_agent_discovery_and_search():
    # Register 2 distinct specialist agents
    agent1 = {
        "agent_id": "rs-grounding",
        "name": "RS Grounding Agent",
        "version": "1.0.0",
        "description": "Land cover region grounding agent for optical imagery.",
        "capabilities": ["region_grounding"],
        "input_modalities": ["optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8002/v1/grounding"
    }
    agent2 = {
        "agent_id": "rs-change",
        "name": "RS Change Detection Agent",
        "version": "1.0.0",
        "description": "Bi-temporal change detection and change VQA agent.",
        "capabilities": ["bitemporal_change_detection", "change_vqa"],
        "input_modalities": ["bitemporal_optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8003/v1/change"
    }
    client.post("/agents/register", json=agent1, headers=ADMIN_HEADERS)
    client.post("/agents/register", json=agent2, headers=ADMIN_HEADERS)

    # 1. Get all agents
    res = client.get("/agents", headers=ROUTER_HEADERS)
    assert res.status_code == 200
    assert len(res.json()) == 2

    # 2. Capability Search: bitemporal_change_detection
    res_cap = client.get("/agents?capability=bitemporal_change_detection", headers=ROUTER_HEADERS)
    assert res_cap.status_code == 200
    matched = res_cap.json()
    assert len(matched) == 1
    assert matched[0]["agent_id"] == "rs-change"

    # 3. Modality Search: optical
    res_mod = client.get("/agents?modality=optical", headers=ROUTER_HEADERS)
    assert res_mod.status_code == 200
    matched_mod = res_mod.json()
    assert len(matched_mod) == 1
    assert matched_mod[0]["agent_id"] == "rs-grounding"


# --- 4. Details, Health Check & Deregistration Tests ---

def test_agent_details_and_not_found():
    payload = {
        "agent_id": "rs-fusion-agent",
        "name": "Optical-SAR Fusion Agent",
        "version": "1.0.0",
        "description": "Co-registered optical and SAR cross-modal joint analysis agent.",
        "capabilities": ["cross_modal_fusion"],
        "input_modalities": ["cross_modal_optical_sar"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8004/v1/fusion"
    }
    client.post("/agents/register", json=payload, headers=ADMIN_HEADERS)

    # Get existing agent
    res = client.get("/agents/rs-fusion-agent", headers=ROUTER_HEADERS)
    assert res.status_code == 200
    assert res.json()["name"] == "Optical-SAR Fusion Agent"

    # Get non-existent agent -> 404
    res_404 = client.get("/agents/non-existent-id", headers=ROUTER_HEADERS)
    assert res_404.status_code == 404
    assert res_404.json()["error_code"] == "AGENT_NOT_FOUND"


def test_health_check():
    payload = {
        "agent_id": "rs-health-agent",
        "name": "Health Test Agent",
        "version": "1.0.0",
        "description": "Agent for testing health check telemetry pings.",
        "capabilities": ["single_image_vqa"],
        "input_modalities": ["optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8005/v1/health"
    }
    client.post("/agents/register", json=payload, headers=ADMIN_HEADERS)

    health_ping = {
        "status": "active",
        "metrics": {"gpu_utilization_pct": 34.5, "latency_ms": 120},
        "message": "All model weights loaded cleanly."
    }
    res = client.post("/agents/rs-health-agent/health", json=health_ping, headers=ROUTER_HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["agent_id"] == "rs-health-agent"
    assert data["status"] == "active"
    assert "last_heartbeat" in data


def test_update_and_deregister():
    payload = {
        "agent_id": "rs-update-agent",
        "name": "Original Name",
        "version": "1.0.0",
        "description": "Agent to be updated and deregistered.",
        "capabilities": ["single_image_vqa"],
        "input_modalities": ["optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8006/v1/update"
    }
    client.post("/agents/register", json=payload, headers=ADMIN_HEADERS)

    # PUT Update
    update_payload = {"name": "Updated Agent Name", "version": "1.1.0", "status": "degraded"}
    res_put = client.put("/agents/rs-update-agent", json=update_payload, headers=ADMIN_HEADERS)
    assert res_put.status_code == 200
    assert res_put.json()["name"] == "Updated Agent Name"
    assert res_put.json()["status"] == "degraded"

    # DELETE Deregister
    res_del = client.delete("/agents/rs-update-agent", headers=ADMIN_HEADERS)
    assert res_del.status_code == 200

    # Verify agent no longer exists
    res_get = client.get("/agents/rs-update-agent", headers=ROUTER_HEADERS)
    assert res_get.status_code == 404


# --- 5. Router Client SDK Integration Test ---

def test_router_client_sdk_integration():
    db = TestingSessionLocal()
    sdk_client = AgentRegistryClient(db_session=db)

    reg_req = AgentRegisterRequest(
        agent_id="sdk-change-agent",
        name="SDK Change Specialist",
        version="2.0.0",
        description="Change detection agent registered via Client SDK.",
        capabilities=[AgentCapability.BITEMPORAL_CHANGE_DETECTION],
        input_modalities=[InputModality.BITEMPORAL_OPTICAL],
        supported_formats=[InputFormat.GEOTIFF],
        endpoint_url="http://localhost:8010/v1/change"
    )
    res = sdk_client.register(reg_req)
    assert res.agent_id == "sdk-change-agent"

    # Discover via SDK
    discovered = sdk_client.discover_agents(
        capability=AgentCapability.BITEMPORAL_CHANGE_DETECTION,
        modality=InputModality.BITEMPORAL_OPTICAL
    )
    assert len(discovered) == 1
    assert discovered[0].agent_id == "sdk-change-agent"

    db.close()
