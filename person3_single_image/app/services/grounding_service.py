"""Text-Guided Visual Grounding Service for Remote Sensing Imagery."""

from typing import Dict, Any, Optional
from PIL import Image

from app.models.base_model import GroundingModel
from app.models.mock_models import MockGrounding


class GroundingService:
    """Orchestrates visual grounding on remote sensing images."""

    def __init__(self, model: Optional[GroundingModel] = None):
        """Initializes the service with a grounding model.

        Defaults to MockGrounding for lightweight execution.
        """
        self.model = model if model is not None else MockGrounding()

    def process_grounding(self, image: Image.Image, query: str) -> Dict[str, Any]:
        """Grounds a text query to visual bounding boxes.

        Args:
            image: PIL Image in RGB format.
            query: Target phrase or object description.

        Returns:
            Structured dictionary:
            {
                "task": "visual_grounding",
                "query": str,
                "boxes": list,
                "labels": list,
                "confidence": Optional[float] (null if unavailable),
                "model": str
            }

        Raises:
            ValueError: If query is empty or image is missing.
        """
        if image is None:
            raise ValueError("An image must be provided for visual grounding.")

        clean_query = query.strip() if query else ""
        if not clean_query:
            raise ValueError("Grounding query cannot be empty.")

        result = self.model.ground(image=image, query=clean_query)

        raw_confidence = result.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else None

        return {
            "task": "visual_grounding",
            "query": clean_query,
            "boxes": result.get("boxes", []),
            "labels": result.get("labels", []),
            "confidence": confidence,
            "model": self.model.model_name,
        }
