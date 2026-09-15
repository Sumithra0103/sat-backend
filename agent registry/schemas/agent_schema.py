"""
Pydantic schemas for SatQuery AI Agent Registry.
Re-exports canonical schemas to ensure unified nominal types across modules.
"""

from routing.schemas.agent_schema import (
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus,
    AgentRegisterRequest,
    AgentUpdateRequest,
    HealthCheckRequest,
    HealthCheckResponse,
    AgentResponse,
    ErrorResponse,
)

__all__ = [
    "AgentCapability",
    "InputModality",
    "InputFormat",
    "AgentStatus",
    "AgentRegisterRequest",
    "AgentUpdateRequest",
    "HealthCheckRequest",
    "HealthCheckResponse",
    "AgentResponse",
    "ErrorResponse",
]
