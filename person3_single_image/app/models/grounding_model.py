"""Grounding Model abstraction and GeoChat visual grounding placeholder.

Note: RS-LLaVA is primarily designed for VQA and captioning, NOT bounding-box grounding.
Visual grounding for remote sensing is typically handled by specialized models like GeoChat
or SAM-RS / Grounding-DINO. This placeholder provides the integration target.
"""

from typing import Dict, Any, Optional
from PIL import Image

from app.models.base_model import GroundingModel


class GeoChatGrounding(GroundingModel):
    """Placeholder integration for GeoChat or remote sensing grounding models.

    GeoChat supports region-level conversation and visual grounding in remote sensing.
    """

    def __init__(self, model_name_or_path: str = "MBZUAI/geochat-7b", device: str = "cuda"):
        self._model_name = "GeoChat-7b (Visual Grounding)"
        self.model_path = model_name_or_path
        self.device = device
        self.model = None

        # =========================================================================
        # TODO: LOAD GEOCHAT OR SPECIALIZED GROUNDING MODEL HERE
        # =========================================================================

    @property
    def model_name(self) -> str:
        return self._model_name

    def ground(self, image: Image.Image, query: str) -> Dict[str, Any]:
        """Runs visual grounding to detect objects/regions matching the query."""
        if self.model is None:
            raise NotImplementedError(
                f"{self.model_name} is not loaded. "
                "Use MockGrounding for local testing or populate GeoChatGrounding initialization."
            )

        # =========================================================================
        # TODO: REAL GROUNDING INFERENCE CODE
        # =========================================================================
        # Returns parsed bounding boxes [x1, y1, x2, y2], labels, confidence.
        # =========================================================================
