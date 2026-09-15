"""Abstract base classes defining interfaces for remote sensing vision-language models.

These interfaces ensure that real models (e.g. RS-LLaVA, GeoChat) and mock models
can be used interchangeably across services without altering downstream application logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image


class BaseModel(ABC):
    """Base class for all remote sensing vision-language models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the identifier or name of the model."""
        pass


class VQAModel(BaseModel):
    """Abstract interface for Single-Image Remote-Sensing Visual Question Answering (VQA)."""

    @abstractmethod
    def answer(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answers a text question about the given remote sensing image.

        Args:
            image: PIL Image in RGB format.
            question: Natural language question regarding the scene.

        Returns:
            Dictionary with structure:
            {
                "answer": str,
                "confidence": Optional[float] (null if unavailable),
                "evidence": list (optional visual evidence details)
            }
        """
        pass


class CaptionModel(BaseModel):
    """Abstract interface for Single-Image Remote-Sensing Captioning."""

    @abstractmethod
    def generate(self, image: Image.Image) -> Dict[str, Any]:
        """Generates a natural language descriptive caption for the remote sensing image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Dictionary with structure:
            {
                "caption": str,
                "confidence": Optional[float] (null if unavailable)
            }
        """
        pass


class GroundingModel(BaseModel):
    """Abstract interface for Text-Guided Visual Grounding in Remote-Sensing."""

    @abstractmethod
    def ground(self, image: Image.Image, query: str) -> Dict[str, Any]:
        """Grounds a text query to visual bounding boxes within the remote sensing image.

        Args:
            image: PIL Image in RGB format.
            query: Text query specifying targets (e.g. "airplanes", "storage tanks").

        Returns:
            Dictionary with structure:
            {
                "query": str,
                "boxes": list of bounding boxes [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2],
                "labels": list of string labels,
                "confidence": Optional[float] (null if unavailable)
            }
        """
        pass
