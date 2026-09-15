"""
SatQuery AI - Trace Data Models
===============================
Pydantic data models for TraceData, TraceSpan, TraceEvent, and supporting schemas
satisfying Steps 1-3 of the Trace Specification.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
from datetime import datetime, timezone


class TraceStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class SpanStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRYING = "RETRYING"
    FALLBACK = "FALLBACK"
    SKIPPED = "SKIPPED"


class TraceEvent(BaseModel):
    """Point-in-time trace event or log milestone."""
    event_id: str
    timestamp: float = Field(default_factory=time.time)
    iso_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    component: str
    status: str = "success"
    message: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    """Single execution span representing a component or agent invocation."""
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    component: str  # e.g., "query_understanding", "router", "agent", "orchestrator"
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_version: Optional[str] = "1.0.0"
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.RUNNING
    confidence: Optional[float] = None
    input_references: Dict[str, Any] = Field(default_factory=dict)
    output_references: Dict[str, Any] = Field(default_factory=dict)
    evidence_references: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    retries: int = 0
    retry_history: List[Dict[str, Any]] = Field(default_factory=list)
    fallbacks: int = 0
    fallback_history: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None


class TraceData(BaseModel):
    """Complete root trace record for a SatQuery request."""
    trace_id: str
    request_id: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    iso_created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    image_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    query_understanding: Optional[Dict[str, Any]] = None
    router_decision: Optional[Dict[str, Any]] = None
    selected_agents: List[Dict[str, Any]] = Field(default_factory=list)
    spans: List[TraceSpan] = Field(default_factory=list)
    events: List[TraceEvent] = Field(default_factory=list)
    total_duration_ms: Optional[float] = None
    status: TraceStatus = TraceStatus.INITIALIZED
    metrics: Dict[str, Any] = Field(default_factory=dict)
