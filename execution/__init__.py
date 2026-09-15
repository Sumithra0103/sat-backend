"""
SatQuery AI Execution Subsystem.
Implements the 19-stage remote sensing specialist execution pipeline.
"""

from execution.schemas import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    EvidenceItem,
    EvidenceType,
    GeospatialMetadata,
    ResourceRequirements,
    ResourceMetrics,
    ExecutionTraceItem
)
from execution.engine import ExecutionEngine
from execution.validation import (
    RequestValidator,
    RequestValidationError,
    InputValidator,
    InputValidationError,
    GeospatialValidator,
    GeospatialValidationError
)
from execution.preprocessing import ImageLoader, ImagePreprocessor, LoadedRaster, PreprocessedBatch
from execution.adapters import AdapterFactory, BaseAgentAdapter
from execution.evidence import EvidenceCollector, EvidenceVisualizer
from execution.models import ModelManager

__all__ = [
    "ExecutionEngine",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "EvidenceItem",
    "EvidenceType",
    "GeospatialMetadata",
    "ResourceRequirements",
    "ResourceMetrics",
    "ExecutionTraceItem",
    "RequestValidator",
    "RequestValidationError",
    "InputValidator",
    "InputValidationError",
    "GeospatialValidator",
    "GeospatialValidationError",
    "ImageLoader",
    "ImagePreprocessor",
    "LoadedRaster",
    "PreprocessedBatch",
    "AdapterFactory",
    "BaseAgentAdapter",
    "EvidenceCollector",
    "EvidenceVisualizer",
    "ModelManager"
]
