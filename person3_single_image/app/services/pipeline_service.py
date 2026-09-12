"""Unified single-image analysis pipeline.

Provides analyze_single_image(image_path, query, task=None) with simple
keyword-based task detection when the task is not explicitly specified.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image

from app.preprocessing.image_loader import load_image
from app.services.vqa_service import VQAService
from app.services.caption_service import CaptionService
from app.services.grounding_service import GroundingService
from app.services.evidence_service import EvidenceService


def detect_task_from_query(query: str) -> str:
    """Heuristic keyword-based detection for task type.

    Rules:
    - Grounding: queries containing 'locate', 'where is', 'show me where', 'find', 'detect'
    - Captioning: queries like 'describe this image', 'generate a caption', 'caption'
    - VQA: queries like 'what is', 'how many', 'is there', 'where', 'does this image contain', or general questions.

    Args:
        query: User input query text.

    Returns:
        One of 'vqa', 'caption', 'grounding'.
    """
    q_lower = query.strip().lower()

    # Grounding keywords take priority if requesting localization
    grounding_triggers = ["locate", "where is", "show me where", "find", "bounding box", "detect"]
    if any(trigger in q_lower for trigger in grounding_triggers):
        return "grounding"

    # Captioning keywords
    caption_triggers = ["describe this image", "generate a caption", "caption this", "describe scene", "image description"]
    if any(trigger in q_lower for trigger in caption_triggers):
        return "caption"

    # Default to VQA for question patterns or fallback
    return "vqa"


class SingleImagePipeline:
    """Unified pipeline coordinator for single-image remote sensing tasks."""

    def __init__(
        self,
        vqa_service: Optional[VQAService] = None,
        caption_service: Optional[CaptionService] = None,
        grounding_service: Optional[GroundingService] = None,
        evidence_service: Optional[EvidenceService] = None,
    ):
        self.vqa_service = vqa_service or VQAService()
        self.caption_service = caption_service or CaptionService()
        self.grounding_service = grounding_service or GroundingService()
        self.evidence_service = evidence_service or EvidenceService()

    def analyze(
        self,
        image_input: Union[str, Path, Image.Image],
        query: str = "",
        task: Optional[str] = None,
        generate_evidence: bool = True,
    ) -> Dict[str, Any]:
        """Analyzes a remote sensing image using the requested or detected task.

        Args:
            image_input: File path (str/Path) or pre-loaded PIL Image.
            query: Question or prompt text.
            task: Optional explicit task ('vqa', 'caption', 'grounding').
            generate_evidence: Whether to produce visual evidence under outputs/evidence/.

        Returns:
            Standardized response dictionary.
        """
        # 1. Load image safely
        if isinstance(image_input, (str, Path)):
            image = load_image(image_input)
            image_source = str(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
            image_source = "PIL.Image"
        else:
            raise TypeError(f"Expected str, Path, or PIL.Image, got {type(image_input)}")

        # 2. Determine task
        selected_task = task.lower().strip() if task else detect_task_from_query(query)

        # 3. Route to corresponding service
        if selected_task == "caption":
            caption_res = self.caption_service.process_caption(image)
            evidence_info = None
            if generate_evidence:
                evidence_info = self.evidence_service.create_evidence(
                    image=image,
                    query="Generate Caption",
                    answer=caption_res["caption"],
                    filename_prefix="caption_evidence",
                )
            return {
                "success": True,
                "task": "single_image_captioning",
                "caption": caption_res["caption"],
                "model": caption_res["model"],
                "confidence": caption_res["confidence"],
                "evidence": [evidence_info] if evidence_info else [],
                "image_source": image_source,
            }

        elif selected_task == "grounding":
            grounding_res = self.grounding_service.process_grounding(image, query=query)
            evidence_info = None
            if generate_evidence:
                evidence_info = self.evidence_service.create_evidence(
                    image=image,
                    query=query,
                    answer=f"Found {len(grounding_res['boxes'])} matching regions.",
                    boxes=grounding_res["boxes"],
                    labels=grounding_res["labels"],
                    filename_prefix="grounding_evidence",
                )
            return {
                "success": True,
                "task": "visual_grounding",
                "query": query,
                "boxes": grounding_res["boxes"],
                "labels": grounding_res["labels"],
                "confidence": grounding_res["confidence"],
                "model": grounding_res["model"],
                "evidence": [evidence_info] if evidence_info else [],
                "image_source": image_source,
            }

        elif selected_task == "vqa":
            vqa_res = self.vqa_service.process_question(image, question=query)
            evidence_info = None
            if generate_evidence:
                evidence_info = self.evidence_service.create_evidence(
                    image=image,
                    query=query,
                    answer=vqa_res["answer"],
                    filename_prefix="vqa_evidence",
                )
            return {
                "success": True,
                "task": "single_image_vqa",
                "question": query,
                "answer": vqa_res["answer"],
                "model": vqa_res["model"],
                "confidence": vqa_res["confidence"],
                "evidence": [evidence_info] if evidence_info else vqa_res.get("evidence", []),
                "image_source": image_source,
            }

        else:
            raise ValueError(f"Unknown task '{selected_task}'. Supported tasks: 'vqa', 'caption', 'grounding'")


# Convenience function for direct module usage
_default_pipeline = SingleImagePipeline()


def analyze_single_image(
    image_path: Union[str, Path, Image.Image],
    query: str = "",
    task: Optional[str] = None,
    generate_evidence: bool = True,
) -> Dict[str, Any]:
    """Top-level functional interface to analyze a single remote sensing image."""
    return _default_pipeline.analyze(
        image_input=image_path,
        query=query,
        task=task,
        generate_evidence=generate_evidence,
    )
