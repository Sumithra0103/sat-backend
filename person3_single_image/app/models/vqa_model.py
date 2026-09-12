"""VQA Model abstraction and RS-LLaVA integration adapter.

Provides RSLLaVAVQA implementing the VQAModel interface, delegating to the
lazy-loaded RSLLaVAModel adapter.
"""

from typing import Dict, Any, Optional
from PIL import Image

from app.models.base_model import VQAModel
from app.models.rsllava_model import RSLLaVAModel


class RSLLaVAVQA(VQAModel):
    """RS-LLaVA integration for Single-Image Remote-Sensing VQA.

    Wraps RSLLaVAModel and implements the VQAModel interface.
    Model weights are lazy-loaded only when answer() is called.
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
        return self.adapter.model_name

    def is_loaded(self) -> bool:
        return self.adapter.is_loaded()

    def answer(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answers a remote-sensing question using RS-LLaVA.

        Confidence is never fabricated and remains None if not calibrated.
        """
        return self.adapter.answer(image=image, question=question)
