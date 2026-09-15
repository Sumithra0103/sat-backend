"""Mock implementations of vision-language models for remote sensing tasks.

These lightweight mock models allow end-to-end pipeline, API, and integration testing
without requiring GPU resources or downloading large model weights (e.g., 7B RS-LLaVA).
No fake coordinates or fake confidence scores are fabricated.
"""

from typing import Dict, Any, List, Optional
from PIL import Image

from app.models.base_model import VQAModel, CaptionModel, GroundingModel


class MockVQA(VQAModel):
    """Mock model for Single-Image Remote-Sensing Visual Question Answering."""

    @property
    def model_name(self) -> str:
        return "MockVQA"

    def answer(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Returns a clearly marked mock answer without fabricating confidence."""
        width, height = image.size
        # Generate a deterministic, clean mock response reflecting question and image size
        mock_answer = (
            f"[TEST MOCK ANSWER] Remote sensing scene analyzed ({width}x{height} px). "
            f"Query: '{question}'. Pipeline functional."
        )
        return {
            "answer": mock_answer,
            "confidence": None,  # Never invent confidence values
            "evidence": [],
        }


class MockCaption(CaptionModel):
    """Mock model for Single-Image Remote-Sensing Captioning."""

    @property
    def model_name(self) -> str:
        return "MockCaption"

    def generate(self, image: Image.Image) -> Dict[str, Any]:
        """Returns a clearly marked mock remote sensing caption."""
        width, height = image.size
        mock_caption = (
            f"[TEST MOCK CAPTION] Aerial/satellite remote sensing imagery with dimensions "
            f"{width}x{height} pixels displaying terrain, infrastructure, or vegetation features."
        )
        return {
            "caption": mock_caption,
            "confidence": None,  # Confidence remains null when not provided by model
        }


class MockGrounding(GroundingModel):
    """Mock model for Text-Guided Visual Grounding."""

    @property
    def model_name(self) -> str:
        return "MockGrounding"

    def ground(self, image: Image.Image, query: str) -> Dict[str, Any]:
        """Returns an honest structured result with empty boxes until real grounding model is connected."""
        return {
            "query": query,
            "boxes": [],  # Do NOT fabricate bounding boxes
            "labels": [],
            "confidence": None,
        }
