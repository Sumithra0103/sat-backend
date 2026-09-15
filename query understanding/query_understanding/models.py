"""
Data models and Enums for SatQuery AI Query Understanding System.
Provides type-safe definitions, validation structures, and JSON serialization methods.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


class Modality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    MULTISPECTRAL = "MULTISPECTRAL"
    THERMAL = "THERMAL"
    UNKNOWN = "UNKNOWN"


class ImageFormat(str, Enum):
    GEOTIFF = "GeoTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    UNKNOWN = "UNKNOWN"


class SensorType(str, Enum):
    CARTOSAT_2S = "Cartosat-2S"
    RISAT = "RISAT"
    SENTINEL_1 = "Sentinel-1"
    SENTINEL_2 = "Sentinel-2"
    PUBLIC_BENCHMARK = "Public_Benchmark"
    GENERIC_OPTICAL = "Generic_Optical"
    GENERIC_SAR = "Generic_SAR"
    UNKNOWN = "UNKNOWN"


class TaskType(str, Enum):
    SINGLE_IMAGE_CAPTIONING = "SINGLE_IMAGE_CAPTIONING"
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    TEXT_GUIDED_GROUNDING = "TEXT_GUIDED_GROUNDING"
    BI_TEMPORAL_CHANGE_UNDERSTANDING = "BI_TEMPORAL_CHANGE_UNDERSTANDING"
    BI_TEMPORAL_CHANGE_QUANTIFICATION = "BI_TEMPORAL_CHANGE_QUANTIFICATION"
    CROSS_MODAL_OPTICAL_SAR_FUSION = "CROSS_MODAL_OPTICAL_SAR_FUSION"
    UNKNOWN = "UNKNOWN"


class UserRole(str, Enum):
    VIEWER = "VIEWER"
    EDITOR = "EDITOR"
    ADMIN = "ADMIN"


@dataclass
class ImageMetadata:
    image_id: str
    file_name: str
    format: ImageFormat
    modality: Modality
    sensor_type: SensorType = SensorType.UNKNOWN
    timestamp: Optional[str] = None
    spatial_bounds: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    resolution_m: Optional[float] = None
    is_co_registered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['format'] = self.format.value
        d['modality'] = self.modality.value
        d['sensor_type'] = self.sensor_type.value
        return d


@dataclass
class ProjectContext:
    project_id: str
    user_id: str
    user_role: UserRole = UserRole.EDITOR
    can_execute_heavy_models: bool = True
    permissions: List[str] = field(default_factory=lambda: ["read", "write", "execute"])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['user_role'] = self.user_role.value
        return d


@dataclass
class ExtractedEntities:
    target_classes: List[str] = field(default_factory=list)
    spatial_actions: List[str] = field(default_factory=list)
    temporal_specs: List[str] = field(default_factory=list)
    is_coreference_resolved: bool = False
    inherited_from_turn: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InputCompatibility:
    is_compatible: bool
    status: str  # VALID, AMBIGUOUS, INCOMPATIBLE
    detected_modalities: List[str] = field(default_factory=list)
    image_count: int = 0
    formats: List[str] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    temporal_structure: str = "SINGLE_TEMPORAL"  # SINGLE_TEMPORAL, BI_TEMPORAL, MULTI_TEMPORAL, CROSS_MODAL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AmbiguityReport:
    is_ambiguous: bool
    issues: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    suggested_remediations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTraceStep:
    step: str
    status: str
    details: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskClassification:
    primary_task: TaskType
    confidence: float
    required_specialist_tools: List[str] = field(default_factory=list)
    benchmark_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['primary_task'] = self.primary_task.value
        return d


@dataclass
class RoutingMetadata:
    target_agent_family: str
    required_model_capabilities: List[str] = field(default_factory=list)
    permitted_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredQueryObject:
    query_id: str
    session_id: str
    original_query: str
    effective_resolved_query: str
    project_context: ProjectContext
    task_classification: TaskClassification
    extracted_entities: ExtractedEntities
    input_compatibility: InputCompatibility
    ambiguity_report: AmbiguityReport
    routing_metadata: RoutingMetadata
    image_inputs: List[Dict[str, Any]] = field(default_factory=list)
    execution_trace: List[ExecutionTraceStep] = field(default_factory=list)

    @property
    def raw_query(self) -> str:
        return self.original_query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "original_query": self.original_query,
            "raw_query": self.raw_query,
            "effective_resolved_query": self.effective_resolved_query,
            "project_context": self.project_context.to_dict(),
            "task_classification": self.task_classification.to_dict(),
            "extracted_entities": self.extracted_entities.to_dict(),
            "input_compatibility": self.input_compatibility.to_dict(),
            "ambiguity_report": self.ambiguity_report.to_dict(),
            "routing_metadata": self.routing_metadata.to_dict(),
            "image_inputs": self.image_inputs,
            "execution_trace": [step.to_dict() for step in self.execution_trace]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class ConversationTurn:
    turn_index: int
    raw_query: str
    sqo: StructuredQueryObject
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationSession:
    session_id: str
    project_context: ProjectContext
    active_images: List[ImageMetadata] = field(default_factory=list)
    turns: List[ConversationTurn] = field(default_factory=list)
