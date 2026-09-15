"""
Repository for Specialist Agent Registry database CRUD operations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from database.models.agent_model import AgentModel


class AgentRepository:
    def __init__(self, db: Session):
        self.db = db

    def register_agent(self, agent_data: Dict[str, Any]) -> AgentModel:
        """Registers a new agent or updates an existing agent record."""
        agent_id = agent_data["agent_id"]
        agent = self.db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()

        now = datetime.now(timezone.utc)
        if agent:
            agent.name = agent_data.get("name", agent.name)
            agent.version = agent_data.get("version", agent.version)
            agent.description = agent_data.get("description", agent.description)
            agent.capabilities = agent_data.get("capabilities", agent.capabilities)
            agent.input_modalities = agent_data.get("input_modalities", agent.input_modalities)
            agent.supported_formats = agent_data.get("supported_formats", agent.supported_formats)
            agent.endpoint_url = agent_data.get("endpoint_url", agent.endpoint_url)
            agent.parameters_schema = agent_data.get("parameters_schema", agent.parameters_schema)
            agent.agent_metadata = agent_data.get("metadata", agent.agent_metadata)
            agent.status = agent_data.get("status", agent.status)
            agent.updated_at = now
            agent.last_heartbeat = now
        else:
            agent = AgentModel(
                agent_id=agent_id,
                name=agent_data.get("name", ""),
                version=agent_data.get("version", "1.0.0"),
                description=agent_data.get("description", ""),
                capabilities=agent_data.get("capabilities", []),
                input_modalities=agent_data.get("input_modalities", []),
                supported_formats=agent_data.get("supported_formats", []),
                endpoint_url=agent_data.get("endpoint_url", ""),
                parameters_schema=agent_data.get("parameters_schema", {}),
                agent_metadata=agent_data.get("metadata", {}),
                status=agent_data.get("status", "active"),
                created_at=now,
                updated_at=now,
                last_heartbeat=now
            )
            self.db.add(agent)

        self.db.commit()
        self.db.refresh(agent)
        return agent

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentModel]:
        """Retrieves agent by primary key agent_id."""
        return self.db.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()

    def get_all_agents(self, status: Optional[str] = None) -> List[AgentModel]:
        """Retrieves list of registered agents, optionally filtered by status."""
        query = self.db.query(AgentModel)
        if status:
            query = query.filter(AgentModel.status == status)
        return query.all()

    def update_heartbeat(self, agent_id: str) -> bool:
        """Updates last_heartbeat timestamp for active agent."""
        agent = self.get_agent_by_id(agent_id)
        if agent:
            agent.last_heartbeat = datetime.now(timezone.utc)
            self.db.commit()
            return True
        return False
