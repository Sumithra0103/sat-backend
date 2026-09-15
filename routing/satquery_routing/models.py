"""
Data models for the SatQuery AI Router.
Defines schemas for routing requirements, candidate evaluations, execution steps,
fallback policies, and the complete Execution Plan consumed by the Orchestrator.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class RoutingStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    AMBIGUOUS_REJECTED = "ambiguous_rejected"


class ExecutionStepType(str, Enum):
    PREPROCESSING = "preprocessing"
    SPECIALIST_INFERENCE = "specialist_inference"
    POSTPROCESSING = "postprocessing"
    EVIDENCE_GROUNDING = "evidence_grounding"


class ExecutionStep(BaseModel):
    step_id: str = Field(..., description="Unique step identifier")
    step_type: ExecutionStepType
    agent_id: Optional[str] = Field(None, description="Assigned agent ID if specialist inference")
    agent_name: Optional[str] = Field(None, description="Display name of the executing agent")
    endpoint_url: Optional[str] = Field(None, description="REST or local endpoint")
    inputs: List[Dict[str, Any]] = Field(default_factory=list, description="Input images or prior step outputs")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Configured task parameters")
    timeout_seconds: float = Field(default=30.0, description="Step timeout")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs that must complete first")


class FallbackRoute(BaseModel):
    primary_agent_id: str
    fallback_agent_id: str
    trigger_condition: str = Field(
        default="on_failure_or_timeout",
        description="Condition triggering fallback (e.g. timeout, 5xx, degraded_health)"
    )
    fallback_endpoint_url: str
    fallback_parameters: Dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    agent_id: str
    name: str
    capability_score: float = Field(..., ge=0.0, le=1.0)
    modality_score: float = Field(..., ge=0.0, le=1.0)
    adaptation_score: float = Field(..., ge=0.0, le=1.0, description="Remote sensing domain adaptation weight")
    latency_score: float = Field(..., ge=0.0, le=1.0)
    health_score: float = Field(..., ge=0.0, le=1.0)
    composite_score: float = Field(..., ge=0.0, le=1.0)
    rejection_reason: Optional[str] = None


class RoutingRequirements(BaseModel):
    query_id: str
    resolved_query: str
    primary_capability: str
    secondary_capabilities: List[str] = Field(default_factory=list)
    required_modality: str
    supported_formats: List[str] = Field(default_factory=list)
    benchmark_reference: Optional[str] = None
    target_classes: List[str] = Field(default_factory=list)
    image_count: int
    temporal_structure: str
    max_acceptable_latency_ms: int = Field(default=2000)
    requires_cloud_resilience: bool = False
    requires_gpu: bool = False


class RouterExecutionTraceItem(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str
    action: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditableExecutionSummary(BaseModel):
    query_id: str
    selected_task: str
    primary_agent_id: str
    primary_agent_name: str
    models_and_tools: List[str]
    applied_parameters: Dict[str, Any]
    benchmark_dataset: Optional[str] = None
    input_configuration: Dict[str, Any]
    routing_confidence: float
    fallback_assigned: Optional[str] = None


class ExecutionPlan(BaseModel):
    """
    Final output of the Router consumed by the SatQuery AI Orchestrator.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_id: str
    query_id: str
    routing_status: RoutingStatus
    requirements: RoutingRequirements
    candidate_evaluations: List[CandidateScore] = Field(default_factory=list)
    selected_agent_id: Optional[str] = None
    steps: List[ExecutionStep] = Field(default_factory=list)
    fallback_routes: List[FallbackRoute] = Field(default_factory=list)
    auditable_summary: Optional[AuditableExecutionSummary] = None
    router_trace: List[RouterExecutionTraceItem] = Field(default_factory=list)
    rejection_errors: List[str] = Field(default_factory=list)
