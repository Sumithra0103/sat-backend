"""
Agent Registry Client SDK for SatQuery AI Router Integration.
Step 15: Connect with Router.
Provides clean Python interface for the SatQuery Router to discover specialist
agents based on capability, input image modality, file format, and operational status.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from database.db import SessionLocal
from schemas.agent_schema import (
    AgentResponse,
    AgentRegisterRequest,
    AgentUpdateRequest,
    HealthCheckRequest,
    HealthCheckResponse,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)
from services.registry_service import AgentRegistryService


class AgentRegistryClient:
    """
    Client library used by SatQuery Router and external controllers to communicate
    with the Agent Registry. Works seamlessly via direct service calls or HTTP API calls.
    Automatically initializes a database session if none is provided.
    """

    def __init__(
        self,
        db_session: Optional[Session] = None,
        api_key: str = "satquery-router-key-2026"
    ):
        self._owns_session = False
        if db_session is None:
            self.db: Session = SessionLocal()
            self._owns_session = True
        else:
            self.db = db_session
        self.api_key = api_key

    def close(self) -> None:
        """Closes the database session if owned by this client instance."""
        if self._owns_session and self.db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def register(self, request: AgentRegisterRequest) -> AgentResponse:
        """Registers a new specialist agent."""
        model = AgentRegistryService.register_agent(self.db, request)
        return AgentResponse.model_validate(model.to_dict())

    def discover_agents(
        self,
        capability: Optional[AgentCapability] = None,
        modality: Optional[InputModality] = None,
        format_type: Optional[InputFormat] = None,
        status: Optional[AgentStatus] = AgentStatus.ACTIVE
    ) -> List[AgentResponse]:
        """
        Step 8: Discover active specialist agents matching capability and modality requirements.
        Example: Find agent for capability='bitemporal_change_detection' and modality='bitemporal_optical'.
        """
        cap_val = capability.value if capability else None
        mod_val = modality.value if modality else None
        fmt_val = format_type.value if format_type else None
        stat_val = status.value if status else None

        models = AgentRegistryService.list_agents(
            self.db,
            capability=cap_val,
            modality=mod_val,
            format_type=fmt_val,
            status=stat_val
        )
        return [AgentResponse.model_validate(m.to_dict()) for m in models]

    def get_agent(self, agent_id: str) -> AgentResponse:
        """Step 9: Retrieve detailed agent metadata."""
        model = AgentRegistryService.get_agent_by_id(self.db, agent_id)
        return AgentResponse.model_validate(model.to_dict())

    def send_health_ping(
        self,
        agent_id: str,
        status_val: AgentStatus,
        metrics: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResponse:
        """Step 10: Report health ping."""
        req = HealthCheckRequest(status=status_val, metrics=metrics or {})
        return AgentRegistryService.update_health(self.db, agent_id, req)

    def deregister(self, agent_id: str) -> bool:
        """Step 11: Deregister an agent."""
        return AgentRegistryService.deregister_agent(self.db, agent_id)
