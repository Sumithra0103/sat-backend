"""
Comprehensive Demonstration & Verification Script for SatQuery AI Agentic Router.
Connects Query Understanding -> Router -> Agent Registry -> Orchestrator Execution Plan.
Executes all representative remote sensing scenarios from the problem statement:
- Single-image scene captioning & description
- Text-guided spatial region grounding
- Bi-temporal change understanding & CDVQA
- Cross-modal (Optical + SAR) joint reasoning
- Multi-turn conversation with coreference resolution
- Ambiguity & missing input rejection
- Health-aware fallback routing
"""

import sys
import json
from pathlib import Path

# Ensure stdout handles UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_ROUTING_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _ROUTING_DIR.parent
_QU_DIR = _ROOT_DIR / "query understanding"
_REG_DIR = _ROOT_DIR / "agent registry"

for p in [str(_ROUTING_DIR), str(_ROOT_DIR), str(_QU_DIR), str(_REG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from database.db import init_db, SessionLocal
from services.registry_service import AgentRegistryService
from client.registry_client import AgentRegistryClient
from query_understanding import QueryUnderstandingPipeline
from satquery_routing import SatQueryRouter, RoutingStatus


def print_separator(title: str):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def display_execution_plan(plan):
    print(f"Status           : {plan.routing_status.value.upper()}")
    print(f"Plan ID          : {plan.plan_id}")
    print(f"Query ID         : {plan.query_id}")

    if plan.routing_status == RoutingStatus.AMBIGUOUS_REJECTED:
        print("\n  [Router Rejection & Remediations]")
        for err in plan.rejection_errors:
            print(f"    - REJECTION: {err}")
        return

    summary = plan.auditable_summary
    if summary:
        print(f"Selected Task    : {summary.selected_task}")
        print(f"Primary Agent    : {summary.primary_agent_name} ({summary.primary_agent_id})")
        print(f"Routing Score    : {summary.routing_confidence:.4f}")
        print(f"Benchmark Match  : {summary.benchmark_dataset}")
        print(f"Fallback Route   : {summary.fallback_assigned or 'None'}")
        print(f"Parameters       : {json.dumps(summary.applied_parameters, indent=2)}")

    print("\n  [Execution Steps for Orchestrator]")
    for step in plan.steps:
        print(f"    - Step: {step.step_id} | Type: {step.step_type.value.upper()} | Agent: {step.agent_name}")
        print(f"      Endpoint: {step.endpoint_url} | Timeout: {step.timeout_seconds}s | Depends on: {step.depends_on}")

    if plan.fallback_routes:
        print("\n  [Fallback & Fault Tolerance]")
        for fb in plan.fallback_routes:
            print(f"    - Trigger: {fb.trigger_condition} -> Fallback Agent: {fb.fallback_agent_id} ({fb.fallback_endpoint_url})")

    print("\n  [Candidate Evaluations & Scores]")
    for cand in plan.candidate_evaluations:
        if cand.rejection_reason:
            print(f"    - {cand.agent_id:25} | DISQUALIFIED: {cand.rejection_reason}")
        else:
            print(f"    - {cand.agent_id:25} | Composite: {cand.composite_score:.4f} (Cap: {cand.capability_score}, Adapt: {cand.adaptation_score}, Mod: {cand.modality_score}, Health: {cand.health_score})")

    print("\n  [Auditable Router Trace]")
    for tr in plan.router_trace:
        print(f"    |- [{tr.stage}] {tr.action}: {tr.status}")


def run_router_demo():
    print("=" * 90)
    print("      SATQUERY AI - AGENTIC ROUTER END-TO-END DEMONSTRATION        ")
    print("=" * 90)

    # 1. Initialize Database & Seed Specialists in Registry
    print("\n[Init] Initializing SQLite Agent Registry Database...")
    init_db()
    db = SessionLocal()
    AgentRegistryService.seed_mock_agents(db)
    print("  --> Registry seeded with standard remote-sensing specialist agents.")

    # 2. Instantiate Query Understanding Pipeline & Router
    qu_pipeline = QueryUnderstandingPipeline()
    router = SatQueryRouter(db_session=db, cache_ttl_seconds=60.0)

    project_context = {
        "project_id": "proj_disaster_monitor_2026",
        "user_id": "analyst_sumit"
    }

    # -------------------------------------------------------------
    # Scenario 1: Single-Image Land-Cover Captioning
    # -------------------------------------------------------------
    print_separator("SCENARIO 1: Single-Image Land-Cover Captioning (BigEarthNet Adapted)")
    query_1 = "Describe the land-cover and major objects visible in this image."
    images_1 = [{
        "image_id": "img_opt_01",
        "file_name": "scene_cartosat2s.tif",
        "format": "GeoTIFF",
        "modality": "OPTICAL",
        "sensor_type": "Cartosat-2S"
    }]
    sqo_1 = qu_pipeline.process_query(query_1, images_1, project_context, session_id="sess_1")
    plan_1 = router.route(sqo_1)
    display_execution_plan(plan_1)

    # -------------------------------------------------------------
    # Scenario 2: Text-Guided Region Grounding
    # -------------------------------------------------------------
    print_separator("SCENARIO 2: Text-Guided Region Grounding (VRSBench Grounding)")
    query_2 = "Highlight the water body referred to in the query."
    images_2 = [{
        "image_id": "img_patch_02",
        "file_name": "reservoir_patch.png",
        "format": "PNG",
        "modality": "OPTICAL",
        "sensor_type": "Public_Benchmark"
    }]
    sqo_2 = qu_pipeline.process_query(query_2, images_2, project_context, session_id="sess_2")
    plan_2 = router.route(sqo_2)
    display_execution_plan(plan_2)

    # -------------------------------------------------------------
    # Scenario 3: Bi-Temporal Change Understanding & CDVQA
    # -------------------------------------------------------------
    print_separator("SCENARIO 3: Bi-Temporal Change Detection & CDVQA")
    query_3 = "What changed between these two dates, and where did the change occur?"
    images_3 = [
        {
            "image_id": "img_t1",
            "file_name": "flood_t1_2024.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "timestamp": "2024-06-01"
        },
        {
            "image_id": "img_t2",
            "file_name": "flood_t2_2024.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "timestamp": "2024-09-15"
        }
    ]
    sqo_3 = qu_pipeline.process_query(query_3, images_3, project_context, session_id="sess_3")
    plan_3 = router.route(sqo_3)
    display_execution_plan(plan_3)

    # -------------------------------------------------------------
    # Scenario 4: Cross-Modal Optical + SAR Joint Analysis
    # -------------------------------------------------------------
    print_separator("SCENARIO 4: Cross-Modal Optical + SAR Joint Reasoning (Cartosat + RISAT)")
    query_4 = "Use the optical and SAR images together to identify built-up and water-covered regions."
    images_4 = [
        {
            "image_id": "img_opt_cartosat",
            "file_name": "urban_cartosat.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "is_co_registered": True
        },
        {
            "image_id": "img_sar_risat",
            "file_name": "urban_risat.tif",
            "format": "GeoTIFF",
            "modality": "SAR",
            "is_co_registered": True
        }
    ]
    sqo_4 = qu_pipeline.process_query(query_4, images_4, project_context, session_id="sess_4")
    plan_4 = router.route(sqo_4)
    display_execution_plan(plan_4)

    # -------------------------------------------------------------
    # Scenario 5: Multi-Turn Conversation with Context Inheritance
    # -------------------------------------------------------------
    print_separator("SCENARIO 5: Multi-Turn Conversation (Turn 2 - Inheriting Bi-Temporal Context)")
    # Turn 1
    qu_pipeline.process_query(
        "Highlight the built-up area in these two images.",
        [
            {"image_id": "city_2020", "format": "GeoTIFF", "modality": "OPTICAL"},
            {"image_id": "city_2026", "format": "GeoTIFF", "modality": "OPTICAL"}
        ],
        session_id="sess_conversation"
    )
    # Turn 2 (No images attached! Coreference 'it' resolves to 'built-up')
    sqo_5 = qu_pipeline.process_query(
        "Has it increased, decreased, or remained unchanged?",
        [],
        session_id="sess_conversation"
    )
    plan_5 = router.route(sqo_5)
    display_execution_plan(plan_5)

    # -------------------------------------------------------------
    # Scenario 6: Ambiguity & Incompatibility Rejection
    # -------------------------------------------------------------
    print_separator("SCENARIO 6: Ambiguity Detection (Change query with only 1 image)")
    query_6 = "What changed between these two dates, and where did the change occur?"
    images_6 = [{"image_id": "single_img", "format": "GeoTIFF", "modality": "OPTICAL"}]
    sqo_6 = qu_pipeline.process_query(query_6, images_6, project_context, session_id="sess_ambiguity")
    plan_6 = router.route(sqo_6)
    display_execution_plan(plan_6)

    db.close()
    print("\n" + "=" * 90)
    print("        ALL SATQUERY ROUTER SCENARIOS EXECUTED SUCCESSFULLY!        ")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_router_demo()
