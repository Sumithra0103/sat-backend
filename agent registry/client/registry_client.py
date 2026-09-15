"""
Agent Registry Client SDK for SatQuery AI Router Integration.
Step 15: Connect with Router.
Provides clean Python interface for the SatQuery Router to discover specialist
agents based on capability, input image modality, file format, and operational status.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

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
    """

    def __init__(self, db_session: Optional[Session] = None, api_key: str = "satquery-router-key-2026"):
        self.db = db_session
        self.api_key = api_key

    def register(self, request: AgentRegisterRequest) -> AgentResponse:
        """Registers a new specialist agent."""
        if not self.db:
            raise RuntimeError("Database session required for direct client execution.")
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
        if not self.db:
            raise RuntimeError("Database session required for direct client execution.")
        
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
        if not self.db:
            raise RuntimeError("Database session required for direct client execution.")
        model = AgentRegistryService.get_agent_by_id(self.db, agent_id)
        return AgentResponse.model_validate(model.to_dict())

    def send_health_ping(self, agent_id: str, status_val: AgentStatus, metrics: Optional[Dict[str, Any]] = None) -> HealthCheckResponse:
        """Step 10: Report health ping."""
        if not self.db:
            raise RuntimeError("Database session required for direct client execution.")
        req = HealthCheckRequest(status=status_val, metrics=metrics or {})
        return AgentRegistryService.update_health(self.db, agent_id, req)

    def deregister(self, agent_id: str) -> bool:
        """Step 11: Deregister an agent."""
        if not self.db:
            raise RuntimeError("Database session required for direct client execution.")
        return AgentRegistryService.deregister_agent(self.db, agent_id)
