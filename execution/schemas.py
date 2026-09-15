"""
Data models and schemas for SatQuery AI Execution Subsystem.
Defines contracts for execution requests, resource allocation, geospatial metadata,
evidence items, execution trace events, and final execution results.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    RETRY_FAILED = "retry_failed"


class EvidenceType(str, Enum):
    CHANGE_MAP = "change_map"
    BOUNDING_BOX = "bounding_box"
    SEGMENTATION_MASK = "segmentation_mask"
    GEOJSON = "geojson"
    CONFIDENCE_HEATMAP = "confidence_heatmap"
    TEXT_RATIONALE = "text_rationale"


class ResourceRequirements(BaseModel):
    gpu_required: bool = False
    min_cpu_cores: int = 1
    min_ram_mb: int = 512
    max_disk_mb: int = 2048
    timeout_seconds: float = 60.0


class ResourceMetrics(BaseModel):
    allocated_gpu: Optional[str] = None
    cpu_cores_used: int = 1
    ram_mb_peak: float = 0.0
    disk_mb_used: float = 0.0
    allocated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    released_at: Optional[str] = None
    execution_duration_sec: float = 0.0


class GeospatialMetadata(BaseModel):
    file_path: str
    format: str
    modality: str
    crs: str = "EPSG:4326"
    bounds: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0], description="[minx, miny, maxx, maxy]")
    resolution_m: float = 10.0
    width: int = 512
    height: int = 512
    bands: List[str] = Field(default_factory=lambda: ["B02", "B03", "B04"])
    nodata_value: Optional[float] = None
    is_registered: bool = True
    sensor_type: Optional[str] = None


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: EvidenceType
    label: str
    description: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    file_path: Optional[str] = None
    geojson: Optional[Dict[str, Any]] = None
    bounding_box: Optional[List[float]] = Field(default=None, description="[ymin, xmin, ymax, xmax] or [x, y, w, h]")
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)
    source_reference: Optional[str] = None


class ExecutionTraceItem(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str
    action: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ExecutionRequest(BaseModel):
    """
    Standard request passed to the Execution subsystem.
    Constructed directly or translated from Router ExecutionStep / ExecutionPlan.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    execution_id: str
    agent_id: str
    agent_version: Optional[str] = "1.0.0"
    task: str
    inputs: List[Dict[str, Any]] = Field(..., description="List of image metadata or raster paths")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task specific parameters")
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    timeout_seconds: float = Field(default=60.0, description="Max execution timeout in seconds")
    fallback_agent_id: Optional[str] = None
    session_id: Optional[str] = None
    resolved_query: Optional[str] = None


class ExecutionResult(BaseModel):
    """
    Standard output format conforming to the requested schema:
    {
      "agent": "...",
      "status": "success",
      "result": {...},
      "confidence": 0.93,
      "execution_time": 4.8
    }
    Extended with auditable trace, evidence list, and resource metrics.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: str = Field(..., description="Executing agent ID")
    status: ExecutionStatus
    result: Dict[str, Any] = Field(default_factory=dict, description="Structured task results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    execution_time: float = Field(..., description="Wall clock execution time in seconds")
    
    # Extended auditable fields
    execution_id: str
    agent_version: Optional[str] = None
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Grounding & change evidence items")
    resource_metrics: Optional[ResourceMetrics] = None
    execution_trace: List[ExecutionTraceItem] = Field(default_factory=list)
    error_details: Optional[Dict[str, Any]] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns the concise JSON format matching the primary spec."""
        return {
            "agent": self.agent,
            "status": self.status.value,
            "result": self.result,
            "confidence": round(self.confidence, 4),
            "execution_time": round(self.execution_time, 3)
        }
