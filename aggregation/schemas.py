"""
SatQuery AI - Aggregation Subsystem Data Schemas
=================================================
Defines Pydantic data models for AgentResult, NormalizedResult, Spatial/Temporal/Modal Evidence,
Unified Evidence Bundles, and Aggregation Results.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
import uuid


class EvidenceModality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    CROSS_MODAL = "cross_modal"
    MULTITEMPORAL = "multitemporal"


class SpatialType(str, Enum):
    BOUNDING_BOX = "bounding_box"
    SEGMENTATION_MASK = "segmentation_mask"
    GEOJSON_POINT = "geojson_point"
    GEOJSON_POLYGON = "geojson_polygon"
    HEATMAP = "heatmap"


class AgentResult(BaseModel):
    """
    Phase 1: Raw output collected from an executed agent step.
    Conforms to ExecutionResult / OrchestrationResult structures.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str = Field(..., description="ID of executing agent")
    task_type: str = Field(..., description="Task classification (e.g. vqa, grounding, change_detection, optical_sar)")
    status: str = Field(default="success", description="Execution status")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent reported confidence score")
    execution_time: float = Field(default=0.0, description="Agent execution time in seconds")
    result_data: Dict[str, Any] = Field(default_factory=dict, description="Raw structured results from model execution")
    evidence_items: List[Dict[str, Any]] = Field(default_factory=list, description="Raw evidence dictionary list")
    domain_adaptation: Optional[str] = Field(default=None, description="Dataset adaptation tag e.g. BigEarthNet, CDVQA")
    modality: Optional[str] = Field(default=None, description="Modality tag e.g. optical, sar, bi-temporal")
    execution_id: Optional[str] = Field(default=None)


class SpatialEvidence(BaseModel):
    """Spatial evidence item (bounding boxes, segmentation masks, region features)."""
    evidence_id: str = Field(default_factory=lambda: f"sp_{uuid.uuid4().hex[:8]}")
    label: str
    spatial_type: SpatialType
    bounding_box: Optional[List[float]] = Field(default=None, description="Normalized [ymin, xmin, ymax, xmax]")
    pixel_box: Optional[List[int]] = Field(default=None, description="Pixel coords [ymin, xmin, ymax, xmax]")
    geo_bounds: Optional[List[float]] = Field(default=None, description="[minx, miny, maxx, maxy]")
    mask_path: Optional[str] = None
    geojson: Optional[Dict[str, Any]] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_agent: str


class TemporalEvidence(BaseModel):
    """Bi-temporal change detection evidence."""
    evidence_id: str = Field(default_factory=lambda: f"temp_{uuid.uuid4().hex[:8]}")
    change_type: str = Field(..., description="e.g., built_up_increase, deforestation, water_body_decrease")
    change_magnitude: float = Field(default=0.0, ge=0.0, le=1.0, description="Relative area/intensity change")
    changed_area_sq_km: Optional[float] = None
    change_map_path: Optional[str] = None
    pre_state: Optional[str] = None
    post_state: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_agent: str


class ModalEvidence(BaseModel):
    """Cross-modal optical-SAR evidence."""
    evidence_id: str = Field(default_factory=lambda: f"mod_{uuid.uuid4().hex[:8]}")
    feature_name: str = Field(..., description="e.g. water_reflectance_sar_smoothness, urban_sar_double_bounce")
    optical_finding: Optional[str] = None
    sar_finding: Optional[str] = None
    synergy_description: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_agent: str


class NormalizedResult(BaseModel):
    """
    Phase 2: Standardized representation of single agent result.
    """
    agent_id: str
    task_type: str
    normalized_text: str = ""
    vqa_answer: Optional[str] = None
    captions: List[str] = Field(default_factory=list)
    spatial_evidence: List[SpatialEvidence] = Field(default_factory=list)
    temporal_evidence: List[TemporalEvidence] = Field(default_factory=list)
    modal_evidence: List[ModalEvidence] = Field(default_factory=list)
    raw_confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    domain_adaptation_weight: float = Field(default=1.0)


class UnifiedEvidenceBundle(BaseModel):
    """Fused multi-modal and multi-temporal evidence bundle."""
    request_id: str
    spatial_evidence: List[SpatialEvidence] = Field(default_factory=list)
    temporal_evidence: List[TemporalEvidence] = Field(default_factory=list)
    modal_evidence: List[ModalEvidence] = Field(default_factory=list)
    resolved_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    fused_change_map_path: Optional[str] = None
    fused_annotated_image_paths: List[str] = Field(default_factory=list)
    land_cover_summary: Dict[str, Any] = Field(default_factory=dict)


class AggregationMetrics(BaseModel):
    total_agents_collected: int = 0
    valid_agents_count: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    spatial_features_fused: int = 0
    temporal_features_fused: int = 0
    cross_modal_features_fused: int = 0
    aggregation_time_seconds: float = 0.0


class AggregationResult(BaseModel):
    """
    Phase 15: Final output produced by SatQuery Aggregation Layer.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    status: str = "success"
    final_answer: str = Field(..., description="Evidence-grounded natural language answer")
    aggregated_confidence: float = Field(..., ge=0.0, le=1.0, description="Composite aggregated confidence score")
    evidence_bundle: UnifiedEvidenceBundle
    visual_evidence_urls: List[str] = Field(default_factory=list, description="Paths to generated visual artifacts")
    auditable_summary: Dict[str, Any] = Field(default_factory=dict, description="Audit trace summary")
    metrics: AggregationMetrics = Field(default_factory=AggregationMetrics)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
