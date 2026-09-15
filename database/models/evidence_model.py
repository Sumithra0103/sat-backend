"""
SQLAlchemy ORM Model for Specialist Visual & Spatial Evidence Artifacts in PostgreSQL.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Text, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from database.base import Base

JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")


class EvidenceArtifactModel(Base):
    __tablename__ = "evidence_artifacts"

    artifact_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    session_id = Column(String(64), ForeignKey("query_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(String(64), nullable=False, index=True)
    
    modality = Column(String(32), nullable=False, default="OPTICAL")
    evidence_type = Column(String(64), nullable=False, default="HEATMAP")
    file_path = Column(String(512), nullable=True)
    
    bbox_coords = Column(JSON_TYPE, nullable=True, default=list)
    confidence_score = Column(Float, nullable=False, default=1.0)
    metadata_json = Column(JSON_TYPE, nullable=False, default=dict)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "modality": self.modality,
            "evidence_type": self.evidence_type,
            "file_path": self.file_path,
            "bbox_coords": self.bbox_coords or [],
            "confidence_score": self.confidence_score,
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
