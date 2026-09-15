"""
SatQuery AI - Trace Context Manager
===================================
Manages trace ID generation and active trace context across thread/async contexts.
Satisfies Steps 4 and 5 of the Trace Specification.
"""

import uuid
import contextvars
from typing import Optional, List
from trace.models import TraceData, TraceSpan


# ContextVar for active trace context
_ACTIVE_TRACE: contextvars.ContextVar[Optional[TraceData]] = contextvars.ContextVar("_ACTIVE_TRACE", default=None)
_SPAN_STACK: contextvars.ContextVar[List[str]] = contextvars.ContextVar("_SPAN_STACK", default=[])


def generate_trace_id() -> str:
    """Step 4: Generates a unique trace ID (trc_<uuid>)."""
    return f"trc_{uuid.uuid4().hex[:16]}"


def generate_span_id() -> str:
    """Generates a unique span ID (spn_<uuid>)."""
    return f"spn_{uuid.uuid4().hex[:12]}"


class TraceContext:
    """Step 5: Manages active TraceContext and active parent spans."""

    @staticmethod
    def get_current_trace() -> Optional[TraceData]:
        return _ACTIVE_TRACE.get()

    @staticmethod
    def set_current_trace(trace: Optional[TraceData]) -> None:
        _ACTIVE_TRACE.set(trace)
        _SPAN_STACK.set([])

    @staticmethod
    def push_span_id(span_id: str) -> None:
        stack = list(_SPAN_STACK.get())
        stack.append(span_id)
        _SPAN_STACK.set(stack)

    @staticmethod
    def pop_span_id() -> Optional[str]:
        stack = list(_SPAN_STACK.get())
        if stack:
            popped = stack.pop()
            _SPAN_STACK.set(stack)
            return popped
        return None

    @staticmethod
    def current_parent_span_id() -> Optional[str]:
        stack = _SPAN_STACK.get()
        return stack[-1] if stack else None
