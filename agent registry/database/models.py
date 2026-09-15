"""
SQLAlchemy ORM models for Agent Registry storing remote sensing specialist agent metadata.
"""

import json
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime
from .db import Base


class AgentModel(Base):
    __tablename__ = "agents"

    agent_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    version = Column(String(32), nullable=False)
    description = Column(Text, nullable=False)
    
    # Serialized JSON string fields for arrays and dicts
    _capabilities = Column("capabilities", Text, nullable=False)
    _input_modalities = Column("input_modalities", Text, nullable=False)
    _supported_formats = Column("supported_formats", Text, nullable=False)
    _parameters_schema = Column("parameters_schema", Text, nullable=False)
    _agent_metadata = Column("agent_metadata", Text, nullable=False)
    
    endpoint_url = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, default="active")
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_heartbeat = Column(DateTime, default=datetime.utcnow, nullable=False)

    @property
    def capabilities(self):
        return json.loads(self._capabilities) if self._capabilities else []

    @capabilities.setter
    def capabilities(self, value):
        self._capabilities = json.dumps(value)

    @property
    def input_modalities(self):
        return json.loads(self._input_modalities) if self._input_modalities else []

    @input_modalities.setter
    def input_modalities(self, value):
        self._input_modalities = json.dumps(value)

    @property
    def supported_formats(self):
        return json.loads(self._supported_formats) if self._supported_formats else []

    @supported_formats.setter
    def supported_formats(self, value):
        self._supported_formats = json.dumps(value)

    @property
    def parameters_schema(self):
        return json.loads(self._parameters_schema) if self._parameters_schema else {}

    @parameters_schema.setter
    def parameters_schema(self, value):
        self._parameters_schema = json.dumps(value)

    @property
    def agent_metadata(self):
        return json.loads(self._agent_metadata) if self._agent_metadata else {}

    @agent_metadata.setter
    def agent_metadata(self, value):
        self._agent_metadata = json.dumps(value)

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities,
            "input_modalities": self.input_modalities,
            "supported_formats": self.supported_formats,
            "endpoint_url": self.endpoint_url,
            "parameters_schema": self.parameters_schema,
            "metadata": self.agent_metadata,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_heartbeat": self.last_heartbeat
        }
