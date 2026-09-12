"""Unit tests for services: VQA, Caption, Grounding, Evidence, and Pipeline."""

import pytest
from pathlib import Path
from PIL import Image

from app.services.vqa_service import VQAService
from app.services.caption_service import CaptionService
from app.services.grounding_service import GroundingService
from app.services.evidence_service import EvidenceService
from app.services.pipeline_service import SingleImagePipeline, detect_task_from_query


@pytest.fixture
def test_image():
    """Provides a simple 200x200 RGB test image."""
    return Image.new("RGB", (200, 200), color=(80, 140, 90))


def test_vqa_service_valid_question(test_image):
    """Verifies VQA service output format and strictly uninvented confidence."""
    service = VQAService()
    question = "How many runways are visible in the satellite scene?"
    result = service.process_question(test_image, question)

    assert result["task"] == "single_image_vqa"
    assert result["question"] == question
    assert len(result["answer"]) > 0
    assert result["confidence"] is None  # Never invent confidence
    assert result["model"] == "MockVQA"
    assert isinstance(result["evidence"], list)


def test_vqa_service_empty_question(test_image):
    """Verifies empty questions raise ValueError."""
    service = VQAService()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.process_question(test_image, "")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.process_question(test_image, "   ")


def test_caption_service(test_image):
    """Verifies Caption service output format."""
    service = CaptionService()
    result = service.process_caption(test_image)

    assert result["task"] == "single_image_captioning"
    assert len(result["caption"]) > 0
    assert result["confidence"] is None
    assert result["model"] == "MockCaption"


def test_grounding_service_valid_query(test_image):
    """Verifies Grounding service output format and ensures no fake coordinates."""
    service = GroundingService()
    query = "locate the airplane on the runway"
    result = service.process_grounding(test_image, query)

    assert result["task"] == "visual_grounding"
    assert result["query"] == query
    assert result["boxes"] == []  # Mock model must not fabricate fake boxes
    assert result["labels"] == []
    assert result["confidence"] is None
    assert result["model"] == "MockGrounding"


def test_grounding_service_empty_query(test_image):
    """Verifies empty grounding query raises ValueError."""
    service = GroundingService()
    with pytest.raises(ValueError, match="Grounding query cannot be empty"):
        service.process_grounding(test_image, "")


def test_evidence_service_without_boxes(tmp_path, test_image):
    """Verifies evidence creation with bottom Q&A panel when no boxes are available."""
    evidence_service = EvidenceService(output_dir=tmp_path)
    res = evidence_service.create_evidence(
        image=test_image,
        query="What is in the image?",
        answer="A lush green agricultural field.",
        boxes=[],
    )

    assert res["type"] == "qa_panel"
    assert res["boxes_count"] == 0
    saved_path = Path(res["evidence_path"])
    assert saved_path.exists()
    assert saved_path.suffix == ".png"


def test_evidence_service_with_boxes(tmp_path, test_image):
    """Verifies evidence creation with bounding boxes drawn directly on the image copy."""
    evidence_service = EvidenceService(output_dir=tmp_path)
    boxes = [[20, 20, 80, 80], [100, 100, 150, 160]]
    labels = ["Structure 1", "Structure 2"]

    res = evidence_service.create_evidence(
        image=test_image,
        query="locate structures",
        answer="Found 2 structures",
        boxes=boxes,
        labels=labels,
    )

    assert res["type"] == "bounding_boxes"
    assert res["boxes_count"] == 2
    saved_path = Path(res["evidence_path"])
    assert saved_path.exists()


def test_task_detection_heuristics():
    """Verifies query-based task detection heuristics."""
    assert detect_task_from_query("where is the airplane?") == "grounding"
    assert detect_task_from_query("locate all storage tanks") == "grounding"
    assert detect_task_from_query("find ships near harbor") == "grounding"

    assert detect_task_from_query("describe this image") == "caption"
    assert detect_task_from_query("generate a caption for this scene") == "caption"

    assert detect_task_from_query("what is the dominant land cover?") == "vqa"
    assert detect_task_from_query("how many bridges cross the river?") == "vqa"
    assert detect_task_from_query("is there an airport visible?") == "vqa"


def test_pipeline_execution(tmp_path, test_image):
    """Verifies end-to-end execution of SingleImagePipeline."""
    evidence_service = EvidenceService(output_dir=tmp_path)
    pipeline = SingleImagePipeline(evidence_service=evidence_service)

    # Test auto VQA
    res_vqa = pipeline.analyze(test_image, query="How many buildings?", task=None)
    assert res_vqa["task"] == "single_image_vqa"
    assert res_vqa["success"] is True

    # Test auto Captioning
    res_cap = pipeline.analyze(test_image, query="describe this image", task=None)
    assert res_cap["task"] == "single_image_captioning"
    assert res_cap["success"] is True

    # Test auto Grounding
    res_grd = pipeline.analyze(test_image, query="locate runway", task=None)
    assert res_grd["task"] == "visual_grounding"
    assert res_grd["success"] is True
