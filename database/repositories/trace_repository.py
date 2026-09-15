"""
Repository for Telemetry Traces and Trace Spans in PostgreSQL.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import Session
from database.models.trace_model import TraceModel, TraceSpanModel


class TraceRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_trace(
        self,
        session_id: str,
        request_id: str,
        query: str = "",
        trace_id: Optional[str] = None
    ) -> TraceModel:
        """Creates a root trace record for a request."""
        tid = trace_id or str(uuid.uuid4())
        trace = TraceModel(
            trace_id=tid,
            session_id=session_id,
            request_id=request_id,
            query=query,
            status="INITIALIZED",
            metrics={}
        )
        self.db.add(trace)
        self.db.commit()
        self.db.refresh(trace)
        return trace

    def save_span(
        self,
        trace_id: str,
        name: str,
        component: str,
        start_time: float,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_version: str = "1.0.0",
        end_time: Optional[float] = None,
        duration_ms: Optional[float] = None,
        status: str = "RUNNING",
        confidence: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> TraceSpanModel:
        """Saves or updates a trace span."""
        sid = span_id or str(uuid.uuid4())
        span = self.db.query(TraceSpanModel).filter(TraceSpanModel.span_id == sid).first()

        if span:
            span.end_time = end_time if end_time is not None else span.end_time
            span.duration_ms = duration_ms if duration_ms is not None else span.duration_ms
            span.status = status
            span.confidence = confidence if confidence is not None else span.confidence
            if payload:
                span.payload = payload
        else:
            span = TraceSpanModel(
                span_id=sid,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=name,
                component=component,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_version=agent_version,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                status=status,
                confidence=confidence,
                payload=payload or {}
            )
            self.db.add(span)

        self.db.commit()
        self.db.refresh(span)
        return span

    def update_trace_status(
        self,
        trace_id: str,
        status: str,
        total_duration_ms: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[TraceModel]:
        """Finalizes trace status, metrics, and duration."""
        trace = self.db.query(TraceModel).filter(TraceModel.trace_id == trace_id).first()
        if trace:
            trace.status = status
            if total_duration_ms is not None:
                trace.total_duration_ms = total_duration_ms
            if metrics:
                trace.metrics = metrics
            trace.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(trace)
        return trace

    def get_trace_by_id(self, trace_id: str) -> Optional[TraceModel]:
        """Fetches root trace by trace_id with preloaded spans."""
        return self.db.query(TraceModel).filter(TraceModel.trace_id == trace_id).first()

    def get_traces_by_session(self, session_id: str) -> List[TraceModel]:
        """Fetches all trace runs for a given query session."""
        return self.db.query(TraceModel).filter(TraceModel.session_id == session_id).all()
