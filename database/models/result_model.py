"""
SQLAlchemy ORM Model for Aggregated Synthesized Answers in PostgreSQL.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Text, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from database.base import Base

JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")


class QueryResultModel(Base):
    __tablename__ = "query_results"

    result_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    session_id = Column(String(64), ForeignKey("query_sessions.session_id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    
    consensus_score = Column(Float, nullable=False, default=1.0)
    synthesized_text = Column(Text, nullable=False)
    
    evidence_summary = Column(JSON_TYPE, nullable=False, default=list)
    conflict_resolution = Column(JSON_TYPE, nullable=True, default=dict)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "result_id": self.result_id,
            "session_id": self.session_id,
            "consensus_score": self.consensus_score,
            "synthesized_text": self.synthesized_text,
            "evidence_summary": self.evidence_summary or [],
            "conflict_resolution": self.conflict_resolution or {},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
