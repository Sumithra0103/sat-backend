"""
SQLAlchemy ORM Model for User Query Sessions & SQO in PostgreSQL.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from database.base import Base

JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")


class QuerySessionModel(Base):
    __tablename__ = "query_sessions"

    session_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(64), nullable=True, index=True)
    raw_query = Column(Text, nullable=False)
    
    image_inputs = Column(JSON_TYPE, nullable=False, default=list)
    project_context = Column(JSON_TYPE, nullable=True, default=dict)
    sqo = Column(JSON_TYPE, nullable=True, default=dict)  # Structured Query Object
    execution_plan = Column(JSON_TYPE, nullable=True, default=dict)  # Router plan
    
    status = Column(String(32), nullable=False, default="PENDING", index=True)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "raw_query": self.raw_query,
            "image_inputs": self.image_inputs or [],
            "project_context": self.project_context or {},
            "sqo": self.sqo or {},
            "execution_plan": self.execution_plan or {},
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
