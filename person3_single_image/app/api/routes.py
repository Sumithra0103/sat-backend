"""FastAPI routes for SatQuery AI - Person 3 Single Image Analysis."""

from typing import Optional, List, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.preprocessing.image_loader import (
    load_image_from_bytes,
    ImagePreprocessingError,
    UnsupportedImageFormatError,
    ImageLoadError,
)
from app.services.vqa_service import VQAService
from app.services.caption_service import CaptionService
from app.services.grounding_service import GroundingService
from app.services.pipeline_service import SingleImagePipeline


router = APIRouter()

# Instantiate services (resolves to Mock or RS-LLaVA adapter based on SATQUERY_USE_REAL_RSLLAVA)
vqa_service = VQAService()
caption_service = CaptionService()
grounding_service = GroundingService()
pipeline = SingleImagePipeline(
    vqa_service=vqa_service,
    caption_service=caption_service,
    grounding_service=grounding_service,
)


# =============================================================================
# Pydantic Response Schemas
# =============================================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    module: str = "person3_single_image"
    real_rsllava_enabled: bool
    active_models: dict


class RootResponse(BaseModel):
    project: str = "SatQuery AI"
    module: str = "Person 3 — Single Image Remote-Sensing Module"
    endpoints: List[str] = [
        "GET /health",
        "POST /vqa",
        "POST /caption",
        "POST /grounding",
        "POST /analyze",
        "GET /docs (Swagger UI)",
    ]


class VQAResponse(BaseModel):
    success: bool = True
    task: str = "single_image_vqa"
    question: str
    answer: str
    model: str
    confidence: Optional[float] = None
    evidence: List[Any] = []


class CaptionResponse(BaseModel):
    success: bool = True
    task: str = "single_image_captioning"
    caption: str
    model: str
    confidence: Optional[float] = None


class GroundingResponse(BaseModel):
    success: bool = True
    task: str = "visual_grounding"
    query: str
    boxes: List[Any] = []
    labels: List[str] = []
    confidence: Optional[float] = None
    model: str


class AnalyzeResponse(BaseModel):
    success: bool = True
    task: str
    result: dict


# =============================================================================
# Routes
# =============================================================================

@router.get("/", response_model=RootResponse, tags=["General"])
async def root():
    """Returns overview of Person 3 module and available endpoints."""
    return RootResponse()


@router.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health status and current model backend configuration.

    NEVER loads or downloads 7B weights.
    """
    settings = get_settings()
    active_models = {
        "vqa": vqa_service.model.model_name,
        "caption": caption_service.model.model_name,
        "grounding": grounding_service.model.model_name,
    }
    return HealthResponse(
        status="ok",
        module="person3_single_image",
        real_rsllava_enabled=settings.use_real_rsllava,
        active_models=active_models,
    )


@router.post("/vqa", response_model=VQAResponse, tags=["Single Image Tasks"])
async def visual_question_answering(
    image: UploadFile = File(..., description="Remote sensing image file (JPG, PNG, TIFF)"),
    question: str = Form(..., description="Natural language question about the image"),
):
    """Answers questions regarding single remote sensing imagery."""
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty.",
        )

    try:
        content = await image.read()
        pil_image = load_image_from_bytes(content, filename=image.filename)
    except (UnsupportedImageFormatError, ImageLoadError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded image: {err}",
        )

    try:
        result = vqa_service.process_question(pil_image, question)
        return VQAResponse(
            success=True,
            task=result["task"],
            question=result["question"],
            answer=result["answer"],
            model=result["model"],
            confidence=result["confidence"],
            evidence=result["evidence"],
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VQA processing error: {err}",
        )


@router.post("/caption", response_model=CaptionResponse, tags=["Single Image Tasks"])
async def generate_caption(
    image: UploadFile = File(..., description="Remote sensing image file (JPG, PNG, TIFF)"),
):
    """Generates a descriptive caption for a single remote sensing image."""
    try:
        content = await image.read()
        pil_image = load_image_from_bytes(content, filename=image.filename)
    except (UnsupportedImageFormatError, ImageLoadError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded image: {err}",
        )

    try:
        result = caption_service.process_caption(pil_image)
        return CaptionResponse(
            success=True,
            task=result["task"],
            caption=result["caption"],
            model=result["model"],
            confidence=result["confidence"],
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Caption processing error: {err}",
        )


@router.post("/grounding", response_model=GroundingResponse, tags=["Single Image Tasks"])
async def visual_grounding(
    image: UploadFile = File(..., description="Remote sensing image file (JPG, PNG, TIFF)"),
    query: str = Form(..., description="Target object/region phrase to ground"),
):
    """Grounds a text query into visual bounding boxes on the image."""
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Grounding query cannot be empty.",
        )

    try:
        content = await image.read()
        pil_image = load_image_from_bytes(content, filename=image.filename)
    except (UnsupportedImageFormatError, ImageLoadError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded image: {err}",
        )

    try:
        result = grounding_service.process_grounding(pil_image, query)
        return GroundingResponse(
            success=True,
            task=result["task"],
            query=result["query"],
            boxes=result["boxes"],
            labels=result["labels"],
            confidence=result["confidence"],
            model=result["model"],
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grounding processing error: {err}",
        )


@router.post("/analyze", tags=["Unified Pipeline"])
async def analyze_image_endpoint(
    image: UploadFile = File(..., description="Remote sensing image file"),
    query: Optional[str] = Form("", description="Question, prompt, or target description"),
    task: Optional[str] = Form(None, description="Optional task override: 'vqa', 'caption', 'grounding'"),
):
    """Unified endpoint with automatic keyword-based task detection and evidence generation."""
    try:
        content = await image.read()
        pil_image = load_image_from_bytes(content, filename=image.filename)
    except (UnsupportedImageFormatError, ImageLoadError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded image: {err}",
        )

    try:
        result = pipeline.analyze(
            image_input=pil_image,
            query=query or "",
            task=task,
            generate_evidence=True,
        )
        return result
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {err}",
        )
