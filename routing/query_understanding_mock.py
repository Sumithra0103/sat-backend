"""
Query Understanding Bridge Module for SatQuery AI.
Connects the routing component with the authoritative query_understanding package
from 'query understanding/query_understanding'.
"""

import sys
from pathlib import Path

_QU_DIR = Path(__file__).resolve().parent.parent / "query understanding"
if str(_QU_DIR) not in sys.path:
    sys.path.insert(0, str(_QU_DIR))

from query_understanding import (
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
    QueryUnderstandingPipeline,
)

# Backward-compatibility alias
PrimaryTask = TaskType

__all__ = [
    "Modality",
    "TaskType",
    "PrimaryTask",
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
