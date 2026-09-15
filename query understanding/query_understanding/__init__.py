"""
SatQuery AI - Query Understanding Package
=========================================
Agentic Remote Sensing Vision-Language Query Understanding Module.
Processes natural language queries, handles multi-turn conversation context,
validates satellite image metadata/formats (GeoTIFF, TIFF, PNG, JPEG),
classifies remote sensing tasks, and produces a Structured Query Object (SQO).
"""

from .models import (
    Modality,
    TaskType,
    ImageFormat,
    SensorType,
    ImageMetadata,
    ProjectContext,
    ExtractedEntities,
    InputCompatibility,
    AmbiguityReport,
    ExecutionTraceStep,
    StructuredQueryObject,
    ConversationTurn,
    ConversationSession,
)
from .pipeline import QueryUnderstandingPipeline

__all__ = [
    "Modality",
    "TaskType",
    "ImageFormat",
    "SensorType",
    "ImageMetadata",
    "ProjectContext",
    "ExtractedEntities",
    "InputCompatibility",
    "AmbiguityReport",
    "ExecutionTraceStep",
    "StructuredQueryObject",
    "ConversationTurn",
    "ConversationSession",
    "QueryUnderstandingPipeline",
]
