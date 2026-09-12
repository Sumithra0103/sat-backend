"""Single-Image Remote-Sensing Captioning Service."""

from typing import Dict, Any, Optional
from PIL import Image

from app.config import get_settings
from app.models.base_model import CaptionModel
from app.models.mock_models import MockCaption
from app.models.caption_model import RSLLaVACaption


class CaptionService:
    """Orchestrates caption generation for single remote sensing images."""

    def __init__(self, model: Optional[CaptionModel] = None):
        """Initializes the service with a captioning model.

        If model is not explicitly injected, resolves based on configuration:
        - SATQUERY_USE_REAL_RSLLAVA=true  -> RSLLaVACaption (lazy-loaded adapter)
        - SATQUERY_USE_REAL_RSLLAVA=false -> MockCaption (default lightweight mock)
        """
        if model is not None:
            self.model = model
        else:
            settings = get_settings()
            if settings.use_real_rsllava:
                self.model = RSLLaVACaption()
            else:
                self.model = MockCaption()

    def process_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generates a caption describing the remote sensing scene.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Structured dictionary:
            {
                "task": "single_image_captioning",
                "caption": str,
                "model": str,
                "confidence": Optional[float] (null if unavailable)
            }

        Raises:
            ValueError: If image is missing.
        """
        if image is None:
            raise ValueError("An image must be provided for captioning.")

        # Delegate generation to the model (MockCaption or RSLLaVACaption)
        result = self.model.generate(image=image)

        # Never invent confidence values
        raw_confidence = result.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else None

        return {
            "task": "single_image_captioning",
            "caption": result.get("caption", ""),
            "model": self.model.model_name,
            "confidence": confidence,
        }
