import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "query understanding"), str(_ROOT / "routing"), str(_ROOT / "agent registry")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aggregation.schemas import (

    AgentResult,
    NormalizedResult,
    SpatialEvidence,
    TemporalEvidence,
    ModalEvidence,
    UnifiedEvidenceBundle,
    AggregationResult,
    AggregationMetrics
)
from aggregation.validator import ResultValidator
from aggregation.normalizer import ResultNormalizer
from aggregation.grouping import EntityGrouper, DuplicateDetector
from aggregation.conflict_resolver import ConflictResolver
from aggregation.confidence_calculator import ConfidenceCalculator
from aggregation.evidence_fusion import SpatialEvidenceFusion, TemporalEvidenceFusion, OpticalSarEvidenceFusion
from aggregation.evidence_builder import UnifiedEvidenceBuilder
from aggregation.synthesizer import AnswerSynthesizer
from aggregation.visual_generator import VisualEvidenceGenerator
from aggregation.aggregator import SatQueryAggregator

__all__ = [
    "AgentResult",
    "NormalizedResult",
    "SpatialEvidence",
    "TemporalEvidence",
    "ModalEvidence",
    "UnifiedEvidenceBundle",
    "AggregationResult",
    "AggregationMetrics",
    "ResultValidator",
    "ResultNormalizer",
    "EntityGrouper",
    "DuplicateDetector",
    "ConflictResolver",
    "ConfidenceCalculator",
    "SpatialEvidenceFusion",
    "TemporalEvidenceFusion",
    "OpticalSarEvidenceFusion",
    "UnifiedEvidenceBuilder",
    "AnswerSynthesizer",
    "VisualEvidenceGenerator",
    "SatQueryAggregator"
]
