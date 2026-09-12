"""Single-Image Remote-Sensing Visual Question Answering (VQA) Service."""

from typing import Dict, Any, Optional
from PIL import Image

from app.config import get_settings
from app.models.base_model import VQAModel
from app.models.mock_models import MockVQA
from app.models.vqa_model import RSLLaVAVQA


class VQAService:
    """Orchestrates VQA queries on single remote sensing images."""

    def __init__(self, model: Optional[VQAModel] = None):
        """Initializes the service with a VQA model.

        If model is not explicitly injected, resolves based on configuration:
        - SATQUERY_USE_REAL_RSLLAVA=true  -> RSLLaVAVQA (lazy-loaded adapter)
        - SATQUERY_USE_REAL_RSLLAVA=false -> MockVQA (default lightweight mock)
        """
        if model is not None:
            self.model = model
        else:
            settings = get_settings()
            if settings.use_real_rsllava:
                self.model = RSLLaVAVQA()
            else:
                self.model = MockVQA()

    def process_question(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answers a question about the given remote sensing image.

        Args:
            image: PIL Image in RGB format.
            question: Natural language question.

        Returns:
            Structured dictionary:
            {
                "task": "single_image_vqa",
                "question": str,
                "answer": str,
                "model": str,
                "confidence": Optional[float] (null if unavailable),
                "evidence": list
            }

        Raises:
            ValueError: If question is empty or image is missing.
        """
        if image is None:
            raise ValueError("An image must be provided for VQA.")

        clean_question = question.strip() if question else ""
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # Delegate inference to the model (MockVQA or RSLLaVAVQA)
        result = self.model.answer(image=image, question=clean_question)

        # Never invent confidence values
        raw_confidence = result.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else None

        return {
            "task": "single_image_vqa",
            "question": clean_question,
            "answer": result.get("answer", ""),
            "model": self.model.model_name,
            "confidence": confidence,
            "evidence": result.get("evidence", []),
        }
