"""
Demonstration & Verification Script for SatQuery AI Agent Registry.
Runs all 15 steps of the Agent Registry workflow end-to-end and outputs structured logs.
"""

import sys
import json
from datetime import datetime

from database.db import init_db, SessionLocal
from schemas.agent_schema import (
    AgentRegisterRequest,
    AgentUpdateRequest,
    HealthCheckRequest,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)
from services.registry_service import (
    AgentRegistryService,
    AgentAlreadyExistsError,
    AgentNotFoundError,
    InvalidMetadataError
)
from client.registry_client import AgentRegistryClient


def run_agent_registry_demo():
    print("=" * 80)
    print(" SatQuery AI - Agent Registry End-to-End Workflow Demonstration ")
    print("=" * 80)

    # 1. Set Up Database (Step 3)
    print("\n[Step 3] Initializing SQLite Database (agent_registry.db)...")
    init_db()
    db = SessionLocal()
    print("  --> Database initialized successfully.")

    # 2. Seed Mock Specialist Agents (Step 2)
    print("\n[Step 2] Seeding Mock Remote Sensing Specialist Agents...")
    seeded_agents = AgentRegistryService.seed_mock_agents(db)
    print(f"  --> Seeded {len(seeded_agents)} specialist agents into registry.")

    # 3. Agent Discovery API (Step 7)
    print("\n[Step 7] Agent Discovery - Listing all registered agents:")
    all_agents = AgentRegistryService.list_agents(db)
    for agent in all_agents:
        print(f"  - [{agent.agent_id}] {agent.name} (v{agent.version}) | Status: {agent.status}")
        print(f"    Capabilities: {agent.capabilities}")
        print(f"    Modalities:   {agent.input_modalities}")

    # 4. Capability-Based Search (Step 8)
    print("\n[Step 8] Capability-Based Search - Querying agents for capability='bitemporal_change_detection':")
    change_agents = AgentRegistryService.list_agents(db, capability="bitemporal_change_detection")
    for agent in change_agents:
        print(f"  --> Found Specialist Match: {agent.name} ({agent.agent_id})")
        print(f"      Endpoint URL: {agent.endpoint_url}")
        print(f"      Supported Bands/Datasets: {agent.agent_metadata.get('training_dataset')}")

    print("\n[Step 8] Capability-Based Search - Querying agents for modality='cross_modal_optical_sar':")
    fusion_agents = AgentRegistryService.list_agents(db, modality="cross_modal_optical_sar")
    for agent in fusion_agents:
        print(f"  --> Found Cross-Modal Specialist: {agent.name} ({agent.agent_id})")
        print(f"      Cloud Resilience: {agent.agent_metadata.get('cloud_resilience')}")

    # 5. Agent Details API (Step 9)
    print("\n[Step 9] Agent Details API - Retrieving details for 'rs-vqa-agent':")
    vqa_agent = AgentRegistryService.get_agent_by_id(db, "rs-vqa-agent")
    print(f"  ID:          {vqa_agent.agent_id}")
    print(f"  Name:        {vqa_agent.name}")
    print(f"  Description: {vqa_agent.description}")
    print(f"  Parameters:  {json.dumps(vqa_agent.parameters_schema, indent=2)}")

    # 6. Metadata Validation & Duplicate Enforcement (Step 5)
    print("\n[Step 5] Metadata Validation - Testing duplicate registration rejection:")
    try:
        duplicate_req = AgentRegisterRequest(
            agent_id="rs-vqa-agent",
            name="Duplicate VQA Agent",
            version="1.0.0",
            description="Testing duplicate validation rejection.",
            capabilities=[AgentCapability.SINGLE_IMAGE_VQA],
            input_modalities=[InputModality.OPTICAL],
            supported_formats=[InputFormat.GEOTIFF],
            endpoint_url="http://localhost:8001/v1/vqa"
        )
        AgentRegistryService.register_agent(db, duplicate_req)
    except AgentAlreadyExistsError as e:
        print(f"  --> [PASSED] Duplicate ID correctly rejected: {e}")

    # 7. Health & Status Check Telemetry (Step 10)
    print("\n[Step 10] Health & Status Ping - Reporting status for 'rs-grounding-agent':")
    health_req = HealthCheckRequest(
        status=AgentStatus.ACTIVE,
        metrics={"latency_ms": 85, "gpu_mem_used_mb": 4120},
        message="Grounding DINO backbone active and ready."
    )
    health_resp = AgentRegistryService.update_health(db, "rs-grounding-agent", health_req)
    print(f"  --> Telemetry Recorded: Agent={health_resp.agent_id}, Status={health_resp.status}, Last Heartbeat={health_resp.last_heartbeat}")

    # 8. Update Agent Metadata (Step 11)
    print("\n[Step 11] Update Agent Metadata - Updating 'rs-change-agent' version and status:")
    update_req = AgentUpdateRequest(
        version="1.6.0",
        description="Upgraded bi-temporal change detection agent with ISRO/SAC benchmark support.",
        status=AgentStatus.ACTIVE
    )
    updated_agent = AgentRegistryService.update_agent(db, "rs-change-agent", update_req)
    print(f"  --> Updated Agent: {updated_agent.agent_id} -> Version={updated_agent.version}")

    # 9. Register Custom Agent (Step 4 & 6)
    print("\n[Step 4 & 6] Registering Custom Agent 'rs-isro-sac-specialist'...")
    custom_req = AgentRegisterRequest(
        agent_id="rs-isro-sac-specialist",
        name="ISRO/SAC Cartosat-RISAT Specialist Agent",
        version="1.0.0",
        description="Specialist agent fine-tuned on Cartosat-2S optical and RISAT SAR co-registered image pairs.",
        capabilities=[AgentCapability.CROSS_MODAL_FUSION, AgentCapability.CHANGE_VQA],
        input_modalities=[InputModality.CROSS_MODAL_OPTICAL_SAR],
        supported_formats=[InputFormat.GEOTIFF, InputFormat.TIFF],
        endpoint_url="http://localhost:8009/v1/isro_sac/execute",
        parameters_schema={"resolution_m": 1.0, "sar_band": "C-band"},
        metadata={"institution": "ISRO/SAC", "sensors": ["Cartosat-2S", "RISAT-1"]}
    )
    new_agent = AgentRegistryService.register_agent(db, custom_req)
    print(f"  --> Successfully registered: {new_agent.name} ({new_agent.agent_id})")

    # 10. Router Client SDK Integration (Step 15)
    print("\n[Step 15] Router Client SDK - Simulating router querying the registry:")
    router_client = AgentRegistryClient(db_session=db)
    
    # Query for Single-Image VQA
    vqa_matches = router_client.discover_agents(
        capability=AgentCapability.SINGLE_IMAGE_VQA,
        modality=InputModality.OPTICAL
    )
    print(f"  --> Router Query 1 [Single-Image Optical VQA] Matches: {[a.agent_id for a in vqa_matches]}")

    # Query for Bi-Temporal Change Detection
    change_matches = router_client.discover_agents(
        capability=AgentCapability.BITEMPORAL_CHANGE_DETECTION,
        modality=InputModality.BITEMPORAL_OPTICAL
    )
    print(f"  --> Router Query 2 [Bi-Temporal Optical Change] Matches: {[a.agent_id for a in change_matches]}")

    # 11. Deregister Agent (Step 11)
    print("\n[Step 11] Deregistering custom agent 'rs-isro-sac-specialist'...")
    AgentRegistryService.deregister_agent(db, "rs-isro-sac-specialist")
    print("  --> Agent successfully deregistered from database.")

    db.close()
    print("\n" + "=" * 80)
    print(" SatQuery AI Agent Registry Workflow Completed Successfully! ")
    print("=" * 80)


if __name__ == "__main__":
    run_agent_registry_demo()
