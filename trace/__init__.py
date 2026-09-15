"""
SatQuery AI - Trace Subsystem Module
====================================
Exposes trace data models, context manager, service API, interactive visualizer,
and traced engine integration.
"""

from trace.models import (
    TraceData,
    TraceSpan,
    TraceEvent,
    TraceStatus,
    SpanStatus
)
from trace.context import (
    TraceContext,
    generate_trace_id,
    generate_span_id
)
from trace.service import TraceService
from trace.api import router as trace_router, get_trace_service
from trace.visualization import render_trace_timeline_html
from trace.integration import TracedSatQueryEngine

__all__ = [
    "TraceData",
    "TraceSpan",
    "TraceEvent",
    "TraceStatus",
    "SpanStatus",
    "TraceContext",
    "generate_trace_id",
    "generate_span_id",
    "TraceService",
    "trace_router",
    "get_trace_service",
    "render_trace_timeline_html",
    "TracedSatQueryEngine",
]
