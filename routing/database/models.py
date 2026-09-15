
"""
SQLAlchemy models for SatQuery AI Agent Registry.
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .db import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AgentModel(Base):
    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", index=True)
    
    # Store JSON attributes
    _capabilities: Mapped[str] = mapped_column("capabilities", Text, default="[]")
    _input_modalities: Mapped[str] = mapped_column("input_modalities", Text, default="[]")
    _supported_formats: Mapped[str] = mapped_column("supported_formats", Text, default="[]")
    _parameters_schema: Mapped[str] = mapped_column("parameters_schema", Text, default="{}")
    _agent_metadata: Mapped[str] = mapped_column("agent_metadata", Text, default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    @property
    def capabilities(self) -> List[str]:
        try:
            return json.loads(self._capabilities or "[]")
        except Exception:
            return []

    @capabilities.setter
    def capabilities(self, value: Any) -> None:
        self._capabilities = json.dumps(value if value is not None else [])

    @property
    def input_modalities(self) -> List[str]:
        try:
            return json.loads(self._input_modalities or "[]")
        except Exception:
            return []

    @input_modalities.setter
    def input_modalities(self, value: Any) -> None:
        self._input_modalities = json.dumps(value if value is not None else [])

    @property
    def supported_formats(self) -> List[str]:
        try:
            return json.loads(self._supported_formats or "[]")
        except Exception:
            return []

    @supported_formats.setter
    def supported_formats(self, value: Any) -> None:
        self._supported_formats = json.dumps(value if value is not None else [])

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        try:
            return json.loads(self._parameters_schema or "{}")
        except Exception:
            return {}

    @parameters_schema.setter
    def parameters_schema(self, value: Any) -> None:
        self._parameters_schema = json.dumps(value if value is not None else {})

    @property
    def agent_metadata(self) -> Dict[str, Any]:
        try:
            return json.loads(self._agent_metadata or "{}")
        except Exception:
            return {}

    @agent_metadata.setter
    def agent_metadata(self, value: Any) -> None:
        self._agent_metadata = json.dumps(value if value is not None else {})

    def to_dict(self) -> Dict[str, Any]:
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
            "created_at": self.created_at or _utc_now(),
            "updated_at": self.updated_at or _utc_now(),
            "last_heartbeat": self.last_heartbeat or _utc_now(),
        }
