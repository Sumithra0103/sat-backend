"""
SatQuery AI - API Subsystem Automated Test Suite
=================================================
Tests all FastAPI endpoints in satquery_api.py:
1. GET / (Health check)
2. POST /api/query/process (Query Understanding & Routing)
3. POST /api/query/execute (Full End-to-End Pipeline)
4. GET /api/agents (Agent Discovery)
5. POST /api/agents/register (Custom Agent Registration)
6. GET /api/trace/s (Trace Listing)
7. GET /api/trace/{trace_id} (Trace Retrieval)
8. GET /api/trace/{trace_id}/viz (Visual Timeline Render)
"""

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
for _sub in [_ROOT_DIR / "agent registry", _ROOT_DIR / "routing", _ROOT_DIR / "query understanding", _ROOT_DIR / "trace"]:
    if str(_sub) not in sys.path:
        sys.path.append(str(_sub))
if str(_ROOT_DIR) in sys.path:
    sys.path.remove(str(_ROOT_DIR))
sys.path.insert(0, str(_ROOT_DIR))


from fastapi.testclient import TestClient
from satquery_api import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "registered_agents_count" in data


def test_process_query_endpoint():
    payload = {
        "raw_query": "Describe the land-cover and major objects visible in this scene.",
        "image_inputs": [
            {
                "file_name": "scene_01.tif",
                "format": "GeoTIFF",
                "modality": "OPTICAL"
            }
        ]
    }
    response = client.post("/api/query/process", json=payload)
    if response.status_code != 200:
        print("ERROR DETAIL:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert "sqo" in data
    assert "execution_plan" in data
    assert data["execution_plan"]["routing_status"] == "success"


def test_execute_query_endpoint():
    payload = {
        "raw_query": "Use optical and SAR images to identify water and built-up regions.",
        "image_inputs": [
            {"file_name": "opt.tif", "format": "GeoTIFF", "modality": "OPTICAL"},
            {"file_name": "sar.tif", "format": "GeoTIFF", "modality": "SAR"}
        ]
    }
    response = client.post("/api/query/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "trace_id" in data
    assert "final_answer" in data
    assert data["aggregated_confidence"] > 0.0


def test_list_agents_endpoint():
    response = client.get("/api/agents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5


def test_register_custom_agent_endpoint():
    import uuid
    unique_agent_id = f"rs-test-agent-{uuid.uuid4().hex[:6]}"
    new_agent = {
        "agent_id": unique_agent_id,
        "name": "Custom Test Agent",
        "version": "1.0.0",
        "description": "API registration test agent for SatQuery Engine.",
        "capabilities": ["single_image_vqa"],
        "input_modalities": ["optical"],
        "supported_formats": ["geotiff"],
        "endpoint_url": "http://localhost:8999/v1/test",
        "status": "active"
    }
    response = client.post("/api/agents/register", json=new_agent)
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["agent_id"] == unique_agent_id


def test_trace_api_endpoints():
    # 1. Execute query to generate trace
    exec_res = client.post("/api/query/execute", json={
        "raw_query": "Highlight water body.",
        "image_inputs": [{"file_name": "w.tif", "format": "GeoTIFF", "modality": "OPTICAL"}]
    }).json()
    trace_id = exec_res["trace_id"]

    # 2. Retrieve trace details
    trace_res = client.get(f"/api/trace/{trace_id}")
    assert trace_res.status_code == 200
    assert trace_res.json()["trace_id"] == trace_id

    # 3. Retrieve trace HTML visualization
    viz_res = client.get(f"/api/trace/{trace_id}/viz")
    assert viz_res.status_code == 200
    assert "html" in viz_res.headers["content-type"]
    assert "SatQuery AI" in viz_res.text


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
