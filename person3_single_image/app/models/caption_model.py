"""Caption Model abstraction and RS-LLaVA captioning adapter.

Provides RSLLaVACaption implementing the CaptionModel interface, delegating to the
lazy-loaded RSLLaVAModel adapter.
"""

from typing import Dict, Any, Optional
from PIL import Image

from app.models.base_model import CaptionModel
from app.models.rsllava_model import RSLLaVAModel


class RSLLaVACaption(CaptionModel):
    """RS-LLaVA integration for Single-Image Remote-Sensing Captioning.

    Wraps RSLLaVAModel and implements the CaptionModel interface.
    Model weights are lazy-loaded only when generate() is called.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        adapter: Optional[RSLLaVAModel] = None,
    ):
        self.adapter = adapter or RSLLaVAModel(
            model_path=model_path,
            model_base=model_base,
            device=device,
            load_in_4bit=load_in_4bit,
        )

    @property
    def model_name(self) -> str:
        return f"{self.adapter.model_name} (Captioning)"

    def is_loaded(self) -> bool:
        return self.adapter.is_loaded()

    def generate(self, image: Image.Image, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generates a remote sensing caption using RS-LLaVA.

        Confidence is never fabricated and remains None if not calibrated.
        """
        return self.adapter.generate(image=image, prompt=prompt)
