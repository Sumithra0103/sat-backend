"""
SQLAlchemy ORM Models for Telemetry Traces and Trace Spans in PostgreSQL.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Text, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import relationship
from database.base import Base

JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")


class TraceModel(Base):
    __tablename__ = "traces"

    trace_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    session_id = Column(String(64), ForeignKey("query_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    request_id = Column(String(64), nullable=False, index=True)
    query = Column(Text, nullable=False, default="")
    
    status = Column(String(32), nullable=False, default="INITIALIZED", index=True)
    total_duration_ms = Column(Float, nullable=True)
    metrics = Column(JSON_TYPE, nullable=False, default=dict)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    spans = relationship("TraceSpanModel", back_populates="trace", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "query": self.query,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "metrics": self.metrics or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "spans": [span.to_dict() for span in self.spans] if self.spans else []
        }


class TraceSpanModel(Base):
    __tablename__ = "trace_spans"

    span_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    trace_id = Column(String(64), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False, index=True)
    parent_span_id = Column(String(64), nullable=True)
    
    name = Column(String(100), nullable=False, index=True)
    component = Column(String(64), nullable=False, index=True)
    agent_id = Column(String(64), nullable=True)
    agent_name = Column(String(100), nullable=True)
    agent_version = Column(String(32), nullable=True, default="1.0.0")

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=True)
    duration_ms = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="RUNNING")
    confidence = Column(Float, nullable=True)

    payload = Column(JSON_TYPE, nullable=False, default=dict)

    trace = relationship("TraceModel", back_populates="spans")

    def to_dict(self):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "component": self.component,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "confidence": self.confidence,
            "payload": self.payload or {}
        }
