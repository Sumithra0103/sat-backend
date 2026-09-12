"""Unit tests for RS-LLaVA integration hook and configuration.

CRITICAL:
These tests run entirely WITHOUT downloading or loading 7B model weights.
"""

import os
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from app.config import get_settings, Settings
from app.models.rsllava_model import RSLLaVAModel
from app.models.vqa_model import RSLLaVAVQA
from app.models.caption_model import RSLLaVACaption
from app.models.mock_models import MockVQA, MockCaption
from app.services.vqa_service import VQAService
from app.services.caption_service import CaptionService


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (100, 100), color=(50, 100, 150))


def test_default_configuration():
    """Verifies that default configuration runs in safe mock mode without 4-bit."""
    cfg = get_settings()
    assert cfg.use_real_rsllava is False
    assert cfg.load_in_4bit is False
    assert "RS-llava" in cfg.model_name
    assert "neural-chat" in cfg.model_base


def test_rsllava_model_instantiation_is_lazy():
    """Verifies creating RSLLaVAModel does NOT load weights or processors."""
    adapter = RSLLaVAModel()
    assert adapter.is_loaded() is False
    assert adapter.model is None
    assert adapter.tokenizer is None
    assert adapter.image_processor is None


def test_rsllava_vqa_and_caption_wrappers_are_lazy():
    """Verifies RSLLaVAVQA and RSLLaVACaption wrappers do NOT load weights on init."""
    vqa_adapter = RSLLaVAVQA()
    assert vqa_adapter.is_loaded() is False

    cap_adapter = RSLLaVACaption()
    assert cap_adapter.is_loaded() is False


def test_vqa_service_resolves_mock_by_default():
    """Verifies VQAService uses MockVQA when SATQUERY_USE_REAL_RSLLAVA is false."""
    with patch.dict(os.environ, {"SATQUERY_USE_REAL_RSLLAVA": "false"}):
        service = VQAService()
        assert isinstance(service.model, MockVQA)
        assert service.model.model_name == "MockVQA"


def test_caption_service_resolves_mock_by_default():
    """Verifies CaptionService uses MockCaption when SATQUERY_USE_REAL_RSLLAVA is false."""
    with patch.dict(os.environ, {"SATQUERY_USE_REAL_RSLLAVA": "false"}):
        service = CaptionService()
        assert isinstance(service.model, MockCaption)
        assert service.model.model_name == "MockCaption"


def test_vqa_service_resolves_rsllava_when_enabled():
    """Verifies VQAService uses RSLLaVAVQA when SATQUERY_USE_REAL_RSLLAVA is true."""
    with patch.dict(os.environ, {"SATQUERY_USE_REAL_RSLLAVA": "true"}):
        service = VQAService()
        assert isinstance(service.model, RSLLaVAVQA)
        # Weights must still be lazy
        assert service.model.is_loaded() is False


def test_caption_service_resolves_rsllava_when_enabled():
    """Verifies CaptionService uses RSLLaVACaption when SATQUERY_USE_REAL_RSLLAVA is true."""
    with patch.dict(os.environ, {"SATQUERY_USE_REAL_RSLLAVA": "true"}):
        service = CaptionService()
        assert isinstance(service.model, RSLLaVACaption)
        # Weights must still be lazy
        assert service.model.is_loaded() is False


def test_rsllava_mocked_inference(dummy_image):
    """Verifies output format of RSLLaVAModel when inference is executed with a mock."""
    adapter = RSLLaVAModel()
    with patch.object(adapter, "_run_inference", return_value="Identified two runways and control tower."):
        # Test VQA output
        vqa_res = adapter.answer(dummy_image, "What structures are here?")
        assert vqa_res["answer"] == "Identified two runways and control tower."
        assert vqa_res["confidence"] is None  # Never invent confidence
        assert vqa_res["evidence"] == []

        # Test Caption output
        cap_res = adapter.generate(dummy_image)
        assert cap_res["caption"] == "Identified two runways and control tower."
        assert cap_res["confidence"] is None


def test_service_with_injected_mock():
    """Verifies service accepts injected model explicitly."""
    mock_model = MockVQA()
    service = VQAService(model=mock_model)
    assert service.model is mock_model


def test_ensure_loaded_raises_runtime_error_when_llava_missing():
    """Verifies that _ensure_loaded raises a clear RuntimeError when official 'llava' is not installed."""
    adapter = RSLLaVAModel()
    with pytest.raises(RuntimeError, match="Official RS-LLaVA package.*not installed"):
        adapter._ensure_loaded()
