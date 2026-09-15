"""
Pydantic schemas for SatQuery AI Agent Registry.
Defines metadata, capability enums, modality enums, registration requests,
update requests, health check schemas, and standardized error models.
"""

import sys
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class AgentCapability(str, Enum):
    SINGLE_IMAGE_VQA = "single_image_vqa"
    REGION_GROUNDING = "region_grounding"
    SCENE_CAPTIONING = "scene_captioning"
    BITEMPORAL_CHANGE_DETECTION = "bitemporal_change_detection"
    CHANGE_VQA = "change_vqa"
    CROSS_MODAL_FUSION = "cross_modal_fusion"


class InputModality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    BITEMPORAL_OPTICAL = "bitemporal_optical"
    BITEMPORAL_SAR = "bitemporal_sar"
    CROSS_MODAL_OPTICAL_SAR = "cross_modal_optical_sar"


class InputFormat(str, Enum):
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


class AgentRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_id: str = Field(
        ...,
        description="Unique string identifier for the specialist agent",
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]+$"
    )
    name: str = Field(..., min_length=3, max_length=100, description="Display name of the agent")
    version: str = Field(..., min_length=1, max_length=32, description="Semantic version string")
    description: str = Field(..., min_length=10, max_length=1000, description="Detailed operational description")
    capabilities: List[AgentCapability] = Field(..., min_length=1, description="List of supported functional capabilities")
    input_modalities: List[InputModality] = Field(..., min_length=1, description="Supported remote sensing image modalities")
    supported_formats: List[InputFormat] = Field(..., min_length=1, description="Supported file formats")
    endpoint_url: str = Field(..., description="REST endpoint URL where agent service is hosted")
    parameters_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema dictionary of allowed execution parameters"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Remote sensing domain metadata (e.g. training set, backbone architecture, resolution)"
    )
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Initial operational status")

    @field_validator("endpoint_url")
    def validate_endpoint_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("endpoint_url must start with http:// or https://")
        return v


class AgentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = Field(None, min_length=3, max_length=100)
    version: Optional[str] = Field(None, min_length=1, max_length=32)
    description: Optional[str] = Field(None, min_length=10, max_length=1000)
    capabilities: Optional[List[AgentCapability]] = None
    input_modalities: Optional[List[InputModality]] = None
    supported_formats: Optional[List[InputFormat]] = None
    endpoint_url: Optional[str] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None

    @field_validator("endpoint_url")
    def validate_endpoint_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("endpoint_url must start with http:// or https://")
        return v


class HealthCheckRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: AgentStatus = Field(..., description="Current self-reported status of the agent")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Telemetry metrics e.g. latency_ms, gpu_memory_pct")
    message: Optional[str] = Field(None, description="Optional status message or diagnostic details")


class HealthCheckResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_id: str
    status: AgentStatus
    last_heartbeat: datetime
    message: str


class AgentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_id: str
    name: str
    version: str
    description: str
    capabilities: List[AgentCapability]
    input_modalities: List[InputModality]
    supported_formats: List[InputFormat]
    endpoint_url: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus
    created_at: datetime
    updated_at: datetime
    last_heartbeat: datetime


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human readable error description")
    error_code: str = Field(..., description="Machine readable error code")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
