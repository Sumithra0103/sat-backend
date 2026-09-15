"""Integration tests for FastAPI REST API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


@pytest.fixture
def dummy_image_bytes():
    """Generates dummy PNG image bytes for testing multipart uploads."""
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(70, 130, 180))
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_root_endpoint():
    """Verifies GET / returns 200 and project metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "SatQuery AI" in data["project"]
    assert "endpoints" in data


def test_health_endpoint():
    """Verifies GET /health returns 200 and active mock models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "MockVQA" in data["active_models"]["vqa"]


def test_vqa_endpoint_success(dummy_image_bytes):
    """Verifies POST /vqa returns standard 200 response."""
    files = {"image": ("test.png", dummy_image_bytes, "image/png")}
    data = {"question": "What is the primary land use?"}

    response = client.post("/vqa", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["task"] == "single_image_vqa"
    assert res["question"] == "What is the primary land use?"
    assert res["model"] == "MockVQA"
    assert res["confidence"] is None
    assert isinstance(res["evidence"], list)


def test_vqa_endpoint_empty_question(dummy_image_bytes):
    """Verifies POST /vqa with empty question returns 400 Bad Request."""
    files = {"image": ("test.png", dummy_image_bytes, "image/png")}
    data = {"question": "   "}

    response = client.post("/vqa", files=files, data=data)
    assert response.status_code == 400
    assert "Question field cannot be empty" in response.json()["detail"]


def test_vqa_endpoint_unsupported_file():
    """Verifies POST /vqa with non-image file returns 400 Bad Request."""
    files = {"image": ("test.pdf", b"%PDF-1.4 dummy", "application/pdf")}
    data = {"question": "What is here?"}

    response = client.post("/vqa", files=files, data=data)
    assert response.status_code == 400


def test_caption_endpoint_success(dummy_image_bytes):
    """Verifies POST /caption returns standard 200 response."""
    files = {"image": ("test.png", dummy_image_bytes, "image/png")}

    response = client.post("/caption", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["task"] == "single_image_captioning"
    assert len(res["caption"]) > 0
    assert res["model"] == "MockCaption"
    assert res["confidence"] is None


def test_grounding_endpoint_success(dummy_image_bytes):
    """Verifies POST /grounding returns standard 200 response."""
    files = {"image": ("test.png", dummy_image_bytes, "image/png")}
    data = {"query": "locate airplane"}

    response = client.post("/grounding", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["task"] == "visual_grounding"
    assert res["query"] == "locate airplane"
    assert res["boxes"] == []
    assert res["model"] == "MockGrounding"


def test_grounding_endpoint_empty_query(dummy_image_bytes):
    """Verifies POST /grounding with empty query returns 400 Bad Request."""
    files = {"image": ("test.png", dummy_image_bytes, "image/png")}
    data = {"query": " "}

    response = client.post("/grounding", files=files, data=data)
    assert response.status_code == 400


def test_analyze_endpoint(dummy_image_bytes):
    """Verifies POST /analyze handles automatic detection and evidence creation."""
    files = {"image": ("test.png", dummy_image_bytes, "image/png")}
    data = {"query": "describe this image"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["task"] == "single_image_captioning"
    assert len(res["evidence"]) > 0
