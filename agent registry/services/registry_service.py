"""
Core Registry Business Logic Service for SatQuery AI Agent Registry.
Provides registration, metadata validation, discovery, capability search,
health ping monitoring, updates, and deregistration.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from database.models import AgentModel
from schemas.agent_schema import (
    AgentRegisterRequest,
    AgentUpdateRequest,
    AgentResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)
from agents.mock_agents import get_mock_agent_requests


class AgentAlreadyExistsError(Exception):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent with ID '{agent_id}' is already registered.")


class AgentNotFoundError(Exception):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent with ID '{agent_id}' was not found in registry.")


class InvalidMetadataError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AgentRegistryService:

    @staticmethod
    def register_agent(db: Session, request: AgentRegisterRequest) -> AgentModel:
        """
        Step 4, 5, 6: Validate Metadata, Check Uniqueness, and Store Agent in DB.
        """
        # Step 5: Validate Unique Agent ID
        existing = db.query(AgentModel).filter(AgentModel.agent_id == request.agent_id).first()
        if existing:
            raise AgentAlreadyExistsError(request.agent_id)

        # Validate capabilities and modalities non-empty
        if not request.capabilities:
            raise InvalidMetadataError("Agent must specify at least one capability.")
        if not request.input_modalities:
            raise InvalidMetadataError("Agent must specify at least one input modality.")
        if not request.supported_formats:
            raise InvalidMetadataError("Agent must specify at least one supported format.")

        now = _utc_now()
        agent_model = AgentModel(
            agent_id=request.agent_id,
            name=request.name,
            version=request.version,
            description=request.description,
            endpoint_url=request.endpoint_url,
            status=request.status.value,
            created_at=now,
            updated_at=now,
            last_heartbeat=now
        )
        # Use setters for JSON fields
        agent_model.capabilities = [c.value if isinstance(c, AgentCapability) else c for c in request.capabilities]
        agent_model.input_modalities = [m.value if isinstance(m, InputModality) else m for m in request.input_modalities]
        agent_model.supported_formats = [f.value if isinstance(f, InputFormat) else f for f in request.supported_formats]
        agent_model.parameters_schema = request.parameters_schema
        agent_model.agent_metadata = request.metadata

        db.add(agent_model)
        db.commit()
        db.refresh(agent_model)
        return agent_model

    @staticmethod
    def list_agents(
        db: Session,
        capability: Optional[str] = None,
        modality: Optional[str] = None,
        format_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[AgentModel]:
        """
        Step 7 & 8: Agent Discovery API & Capability-Based Search.
        Supports filtering by capability, input modality, supported format, and operational status.
        """
        query = db.query(AgentModel)
        
        if status:
            query = query.filter(AgentModel.status == status)
        
        agents = query.all()
        
        filtered = []
        for agent in agents:
            match = True
            if capability and capability not in agent.capabilities:
                match = False
            if modality and modality not in agent.input_modalities:
                match = False
            if format_type and format_type not in agent.supported_formats:
                match = False
            if match:
                filtered.append(agent)
                
        return filtered

    @staticmethod
    def get_agent_by_id(db: Session, agent_id: str) -> AgentModel:
        """
        Step 9: Agent Details API.
        """
        agent = db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
        if not agent:
            raise AgentNotFoundError(agent_id)
        return agent

    @staticmethod
    def update_health(db: Session, agent_id: str, health_req: HealthCheckRequest) -> HealthCheckResponse:
        """
        Step 10: Health & Status Check telemetry ping.
        """
        agent = db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
        if not agent:
            raise AgentNotFoundError(agent_id)

        now = _utc_now()
        agent.status = health_req.status.value
        agent.last_heartbeat = now
        agent.updated_at = now
        
        db.commit()
        db.refresh(agent)

        msg = health_req.message or f"Agent '{agent_id}' health updated to status {health_req.status.value}"
        return HealthCheckResponse(
            agent_id=agent.agent_id,
            status=AgentStatus(agent.status),
            last_heartbeat=now,
            message=msg
        )

    @staticmethod
    def update_agent(db: Session, agent_id: str, update_req: AgentUpdateRequest) -> AgentModel:
        """
        Step 11: Update Agent metadata and configuration.
        """
        agent = db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
        if not agent:
            raise AgentNotFoundError(agent_id)

        now = _utc_now()
        
        if update_req.name is not None:
            agent.name = update_req.name
        if update_req.version is not None:
            agent.version = update_req.version
        if update_req.description is not None:
            agent.description = update_req.description
        if update_req.endpoint_url is not None:
            agent.endpoint_url = update_req.endpoint_url
        if update_req.status is not None:
            agent.status = update_req.status.value
        if update_req.capabilities is not None:
            agent.capabilities = [c.value if isinstance(c, AgentCapability) else c for c in update_req.capabilities]
        if update_req.input_modalities is not None:
            agent.input_modalities = [m.value if isinstance(m, InputModality) else m for m in update_req.input_modalities]
        if update_req.supported_formats is not None:
            agent.supported_formats = [f.value if isinstance(f, InputFormat) else f for f in update_req.supported_formats]
        if update_req.parameters_schema is not None:
            agent.parameters_schema = update_req.parameters_schema
        if update_req.metadata is not None:
            agent.agent_metadata = update_req.metadata

        agent.updated_at = now
        db.commit()
        db.refresh(agent)
        return agent

    @staticmethod
    def deregister_agent(db: Session, agent_id: str) -> bool:
        """
        Step 11: Deregister agent by removing it from database.
        """
        agent = db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
        if not agent:
            raise AgentNotFoundError(agent_id)

        db.delete(agent)
        db.commit()
        return True

    @classmethod
    def seed_mock_agents(cls, db: Session) -> List[AgentModel]:
        """
        Step 2: Seed database with standard mock specialist agents if missing.
        """
        seeded = []
        for req in get_mock_agent_requests():
            try:
                agent = cls.register_agent(db, req)
                seeded.append(agent)
            except AgentAlreadyExistsError:
                agent = cls.get_agent_by_id(db, req.agent_id)
                seeded.append(agent)
        return seeded
