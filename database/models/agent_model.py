"""
SQLAlchemy ORM Model for Specialist Agent Registry in PostgreSQL.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from database.base import Base

# JSONB for PostgreSQL with fallback to generic JSON for SQLite
JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")


class AgentModel(Base):
    __tablename__ = "agents"

    agent_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    version = Column(String(32), nullable=False, default="1.0.0")
    description = Column(Text, nullable=False, default="")

    capabilities = Column(JSON_TYPE, nullable=False, default=list)
    input_modalities = Column(JSON_TYPE, nullable=False, default=list)
    supported_formats = Column(JSON_TYPE, nullable=False, default=list)
    parameters_schema = Column(JSON_TYPE, nullable=False, default=dict)
    agent_metadata = Column(JSON_TYPE, nullable=False, default=dict)

    endpoint_url = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, default="active", index=True)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    last_heartbeat = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities or [],
            "input_modalities": self.input_modalities or [],
            "supported_formats": self.supported_formats or [],
            "endpoint_url": self.endpoint_url,
            "parameters_schema": self.parameters_schema or {},
            "metadata": self.agent_metadata or {},
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }
