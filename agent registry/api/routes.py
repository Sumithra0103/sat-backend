"""
FastAPI REST API Routes for SatQuery AI Agent Registry.
Implements Registration, Discovery, Capability Search, Agent Details,
Health Monitoring, Updates, Deregistration, and Security.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Path, status
from sqlalchemy.orm import Session

from database.db import get_db
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
from services.registry_service import AgentRegistryService
from security.auth import verify_admin_key, verify_read_key

router = APIRouter(prefix="/agents", tags=["Agent Registry"])


@router.post(
    "/register",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_admin_key)],
    summary="Step 4: Register New Specialist Agent"
)
def register_agent(
    request: AgentRegisterRequest,
    db: Session = Depends(get_db)
):
    """
    Registers a new remote sensing specialist agent into the SatQuery AI Registry.
    Validates metadata, enforces unique agent_id, and stores agent specifications.
    """
    agent = AgentRegistryService.register_agent(db, request)
    return AgentResponse.model_validate(agent.to_dict())


@router.get(
    "",
    response_model=List[AgentResponse],
    dependencies=[Depends(verify_read_key)],
    summary="Step 7 & 8: Discover Agents & Capability-Based Search"
)
def list_agents(
    capability: Optional[AgentCapability] = Query(
        None,
        description="Filter agents offering specific functional capability (e.g. single_image_vqa, change_vqa, cross_modal_fusion)"
    ),
    modality: Optional[InputModality] = Query(
        None,
        description="Filter agents supporting specific remote sensing image modality (e.g. optical, sar, bitemporal_optical)"
    ),
    format_type: Optional[InputFormat] = Query(
        None,
        alias="format",
        description="Filter agents supporting specific image format (e.g. geotiff, tiff, png, jpeg)"
    ),
    status: Optional[AgentStatus] = Query(
        None,
        description="Filter agents by operational status (active, degraded, inactive, maintenance)"
    ),
    db: Session = Depends(get_db)
):
    """
    Retrieves all registered agents matching optional search filters.
    Used by the SatQuery Router for capability-based agent discovery.
    """
    cap_val = capability.value if capability else None
    mod_val = modality.value if modality else None
    fmt_val = format_type.value if format_type else None
    stat_val = status.value if status else None

    agents = AgentRegistryService.list_agents(
        db,
        capability=cap_val,
        modality=mod_val,
        format_type=fmt_val,
        status=stat_val
    )
    return [AgentResponse.model_validate(a.to_dict()) for a in agents]


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    dependencies=[Depends(verify_read_key)],
    summary="Step 9: Retrieve Agent Details"
)
def get_agent_details(
    agent_id: str = Path(..., description="Unique string identifier of the target agent"),
    db: Session = Depends(get_db)
):
    """
    Returns full metadata, parameter schemas, and telemetry status for a specific agent.
    """
    agent = AgentRegistryService.get_agent_by_id(db, agent_id)
    return AgentResponse.model_validate(agent.to_dict())


@router.post(
    "/{agent_id}/health",
    response_model=HealthCheckResponse,
    dependencies=[Depends(verify_read_key)],
    summary="Step 10: Update Agent Health & Heartbeat Status"
)
def update_agent_health(
    request: HealthCheckRequest,
    agent_id: str = Path(..., description="Unique agent identifier"),
    db: Session = Depends(get_db)
):
    """
    Heartbeat and telemetry ping endpoint for agents to report operational status.
    """
    return AgentRegistryService.update_health(db, agent_id, request)


@router.put(
    "/{agent_id}",
    response_model=AgentResponse,
    dependencies=[Depends(verify_admin_key)],
    summary="Step 11: Update Registered Agent Metadata"
)
def update_agent(
    request: AgentUpdateRequest,
    agent_id: str = Path(..., description="Unique agent identifier"),
    db: Session = Depends(get_db)
):
    """
    Updates existing agent specifications, capabilities, endpoint URL, or operational status.
    """
    agent = AgentRegistryService.update_agent(db, agent_id, request)
    return AgentResponse.model_validate(agent.to_dict())


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_admin_key)],
    summary="Step 11: Deregister Agent"
)
def deregister_agent(
    agent_id: str = Path(..., description="Unique agent identifier"),
    db: Session = Depends(get_db)
):
    """
    Removes an agent permanently from the registry database.
    """
    AgentRegistryService.deregister_agent(db, agent_id)
    return {"message": f"Agent '{agent_id}' has been successfully deregistered."}
