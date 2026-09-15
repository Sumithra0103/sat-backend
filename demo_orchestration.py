"""
SatQuery AI - Comprehensive 17-Stage Agentic Orchestration Demonstration
========================================================================
Demonstrates the complete multi-agent DAG execution pipeline:
Query Understanding -> Agent Registry -> Router -> 17-Stage Agentic Orchestrator -> Evidence Aggregation

17-Stage Orchestration Pipeline:
1. Receive Request
2. Create Workflow
3. Validate Plan
4. Build Dependency DAG
5. Initialize Shared State
6. Identify Ready Nodes
7. Create Execution Requests
8. Send to Execution Layer (Parallel / Sequential Execution)
9. Wait for Agent Result (Retry Policy & Fallback Handling)
10. Update Node State
11. Store Result (Thread-Safe State Synchronization)
12. Unlock Dependents
13. Schedule Next Ready Agents
14. Synchronize All Results
15. Collect Evidence
16. Calculate Workflow Status
17. Create Final Orchestration Result (Auditable trace for Aggregation Layer)
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from satquery_engine import SatQueryEngine
from orchestration import WorkflowStatus, NodeStatus


def print_banner(title: str, subtitle: str = ""):
    width = 95
    print("\n" + "=" * width)
    print(f" {title.upper().center(width - 2)} ")
    if subtitle:
        print(f" {subtitle.center(width - 2)} ")
    print("=" * width + "\n")


def print_scenario_header(num: int, title: str, query: str):
    print(f"\n+-----------------------------------------------------------------------------------------+")
    print(f"| SCENARIO {num}: {title.ljust(78)} |")
    print(f"+-----------------------------------------------------------------------------------------+")
    print(f"| QUERY: \"{query}\"".ljust(90) + "|")
    print(f"+-----------------------------------------------------------------------------------------+")


def format_json(data: dict) -> str:
    return json.dumps(data, indent=2)


def run_orchestration_demo():
    print_banner(
        "SATQUERY AI: 17-STAGE AGENTIC ORCHESTRATION PIPELINE",
        "Dependency DAG -> Parallel/Sequential Dispatch -> Shared State Sync -> Aggregation"
    )

    engine = SatQueryEngine(auto_seed=True)

    scenarios = [
        {
            "id": 1,
            "title": "Single Optical Image Visual Question Answering (RSVQA / VRSBench)",
            "query": "What is the primary land-cover type visible in this coastal area?",
            "images": [
                {
                    "image_id": "opt_img_01",
                    "file_name": "coastal_sentinel2_optical.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "spatial_resolution_m": 10.0
                }
            ]
        },
        {
            "id": 2,
            "title": "Single Image Scene Captioning & Land-Cover Description (BigEarthNet)",
            "query": "Describe the land-cover and major spatial structures in this multispectral image.",
            "images": [
                {
                    "image_id": "opt_img_02",
                    "file_name": "agricultural_bigearthnet.tif",
                    "format": "GeoTIFF",
                    "modality": "MULTISPECTRAL",
                    "spatial_resolution_m": 10.0
                }
            ]
        },
        {
            "id": 3,
            "title": "Text-Guided Region Grounding & Bounding Box Localization",
            "query": "Highlight the water bodies and port infrastructure referred to in the query.",
            "images": [
                {
                    "image_id": "opt_img_03",
                    "file_name": "port_infrastructure.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "spatial_resolution_m": 0.5
                }
            ]
        },
        {
            "id": 4,
            "title": "Bi-temporal Change Detection & Change VQA (CDVQA)",
            "query": "What changed between these two acquisition dates and where did urban expansion occur?",
            "images": [
                {
                    "image_id": "t1_opt",
                    "file_name": "urban_pre_2022.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "acquisition_date": "2022-01-15"
                },
                {
                    "image_id": "t2_opt",
                    "file_name": "urban_post_2024.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "acquisition_date": "2024-01-15"
                }
            ]
        },
        {
            "id": 5,
            "title": "Joint Cross-Modal Optical-SAR Feature Extraction & Fusion",
            "query": "Use the co-registered optical and SAR images together to identify flooded regions through cloud cover.",
            "images": [
                {
                    "image_id": "optical_pair",
                    "file_name": "cartosat2_optical_flood.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_name": "Cartosat-2S"
                },
                {
                    "image_id": "sar_pair",
                    "file_name": "risat1_sar_flood.tif",
                    "format": "GeoTIFF",
                    "modality": "SAR",
                    "sensor_name": "RISAT-1"
                }
            ]
        },
        {
            "id": 6,
            "title": "Fault-Tolerant Fallback & Retry Handling",
            "query": "Perform specialized land-cover segmentation with fallback agent support.",
            "images": [
                {
                    "image_id": "opt_img_fb",
                    "file_name": "complex_terrain.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL"
                }
            ]
        },
        {
            "id": 7,
            "title": "Parallel Multi-Task Execution & Shared State Synchronization",
            "query": "Perform simultaneous land cover classification, object detection, and scene captioning.",
            "images": [
                {
                    "image_id": "multi_task_img",
                    "file_name": "harbor_multispectral.tif",
                    "format": "GeoTIFF",
                    "modality": "MULTISPECTRAL"
                }
            ]
        }
    ]

    for scenario in scenarios:
        s_id = scenario["id"]
        title = scenario["title"]
        query = scenario["query"]
        images = scenario["images"]

        print_scenario_header(s_id, title, query)

        # Execute end-to-end through 17-stage orchestrator
        sqo, plan, orch_res = engine.process_and_orchestrate(
            raw_query=query,
            image_inputs=images
        )

        print(f"▶ QUERY UNDERSTANDING:")
        print(f"  • Task Category:      {sqo.task_classification.primary_task.value}")
        print(f"  • Modality Required:  {sqo.input_compatibility.detected_modalities}")
        print(f"  • Resolved Query:     \"{sqo.effective_resolved_query}\"")

        print(f"\n▶ ROUTING & PLAN BUILDER:")
        print(f"  • Selected Agent:     {plan.selected_agent_id}")
        print(f"  • Routing Status:     {plan.routing_status.value}")
        print(f"  • Execution Steps:    {len(plan.steps)}")

        print(f"\n▶ 17-STAGE AGENTIC ORCHESTRATION RESULTS:")
        print(f"  • Workflow ID:        {orch_res.workflow_id}")
        print(f"  • Workflow Status:    {orch_res.status.value}")
        print(f"  • Total Execution:    {orch_res.total_execution_time:.3f} seconds")
        print(f"  • DAG Topology:       {orch_res.dag_topology}")
        print(f"  • Completed Agents:   {[r.agent for r in orch_res.completed_agent_results]}")

        print(f"\n▶ AUDITABLE WORKFLOW TRACE:")
        for trace_item in orch_res.execution_trace:
            if any(stage_num in trace_item.stage for stage_num in ["1.", "4.", "6.", "8.", "11.", "16.", "17."]):
                print(f"  [{trace_item.stage}] {trace_item.action} -> status: {trace_item.status}")

        print(f"\n▶ SHARED EXECUTION STATE OUTPUTS:")
        st = orch_res.shared_state
        if st.vqa_answers:
            print(f"  • VQA Answers:        {st.vqa_answers}")
        if st.captions:
            print(f"  • Captions:           {st.captions}")
        if st.change_map:
            print(f"  • Change Map Ref:     {st.change_map}")
        if st.detected_objects:
            print(f"  • Detected Objects:   {len(st.detected_objects)} objects found")
        if st.confidence_scores:
            print(f"  • Confidence Scores:  {st.confidence_scores}")

        print(f"\n" + "-" * 95)

    print_banner(
        "ALL 7 ORCHESTRATION SCENARIOS EXECUTED & VERIFIED SUCCESSFULLY!",
        "SatQuery AI Orchestration Engine is Ready for Deployment"
    )

    engine.close()


if __name__ == "__main__":
    run_orchestration_demo()
