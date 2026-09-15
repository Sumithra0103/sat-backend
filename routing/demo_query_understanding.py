"""
Demonstration script for SatQuery AI Query Understanding System.
Executes representative remote sensing queries across single-image, bi-temporal,
cross-modal (Optical + SAR), multi-turn conversation, and ambiguity edge cases.
Uses standard Python library (no third-party dependencies required).
"""

import io
import sys
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Any

# Ensure stdout handles UTF-8 on Windows console
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

_ROUTING_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _ROUTING_DIR.parent
_QU_DIR = _ROOT_DIR / "query understanding"

for p in [str(_ROUTING_DIR), str(_ROOT_DIR), str(_QU_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# pyrefly: ignore [missing-import]
from query_understanding import QueryUnderstandingPipeline, Modality, ImageFormat, SensorType


class Scenario(TypedDict):
    title: str
    query: str
    images: List[Dict[str, Any]]
    session_id: str


def run_demo():
    print("=" * 80)
    print("      SATQUERY AI - AGENTIC QUERY UNDERSTANDING SYSTEM DEMO        ")
    print("=" * 80 + "\n")

    pipeline = QueryUnderstandingPipeline()

    project_context = {
        "project_id": "proj_disaster_monitor_2026",
        "user_id": "analyst_sumit",
        "user_role": "EDITOR",
        "can_execute_heavy_models": True
    }

    scenarios: List[Scenario] = [
        {
            "title": "Scenario 1: Single-Image Land-Cover Captioning",
            "query": "Describe the land-cover and major objects visible in this image.",
            "images": [
                {
                    "image_id": "img_cartosat_01",
                    "file_name": "scene_cartosat2s_optical.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_type": "Cartosat-2S",
                    "timestamp": "2026-01-10T10:30:00Z"
                }
            ],
            "session_id": "sess_demo_1"
        },
        {
            "title": "Scenario 2: Text-Guided Region Grounding",
            "query": "Highlight the water body referred to in the query.",
            "images": [
                {
                    "image_id": "img_vrsbench_02",
                    "file_name": "reservoir_patch.png",
                    "format": "PNG",
                    "modality": "OPTICAL",
                    "sensor_type": "Public_Benchmark",
                    "timestamp": "2025-11-05T14:15:00Z"
                }
            ],
            "session_id": "sess_demo_2"
        },
        {
            "title": "Scenario 3: Bi-Temporal Change Understanding",
            "query": "What changed between these two dates, and where did the change occur?",
            "images": [
                {
                    "image_id": "img_t1",
                    "file_name": "flood_t1_2024.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_type": "Sentinel-2",
                    "timestamp": "2024-06-01T10:00:00Z"
                },
                {
                    "image_id": "img_t2",
                    "file_name": "flood_t2_2024.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_type": "Sentinel-2",
                    "timestamp": "2024-09-15T10:00:00Z"
                }
            ],
            "session_id": "sess_demo_3"
        },
        {
            "title": "Scenario 4: Cross-Modal Optical + SAR Joint Analysis",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "images": [
                {
                    "image_id": "img_opt",
                    "file_name": "urban_cartosat2s.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_type": "Cartosat-2S",
                    "timestamp": "2026-03-01T09:00:00Z",
                    "is_co_registered": True
                },
                {
                    "image_id": "img_sar",
                    "file_name": "urban_risat1_sar.tif",
                    "format": "GeoTIFF",
                    "modality": "SAR",
                    "sensor_type": "RISAT",
                    "timestamp": "2026-03-01T21:00:00Z",
                    "is_co_registered": True
                }
            ],
            "session_id": "sess_demo_4"
        },
        {
            "title": "Scenario 5: Multi-Turn Conversation (Turn 1)",
            "query": "Highlight the built-up area in these two images.",
            "images": [
                {
                    "image_id": "img_urban_2020",
                    "file_name": "city_2020.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_type": "Cartosat-2S",
                    "timestamp": "2020-01-01"
                },
                {
                    "image_id": "img_urban_2026",
                    "file_name": "city_2026.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL",
                    "sensor_type": "Cartosat-2S",
                    "timestamp": "2026-01-01"
                }
            ],
            "session_id": "sess_multi_turn_demo"
        },
        {
            "title": "Scenario 5: Multi-Turn Conversation (Turn 2 - Inheriting Context & Anaphora)",
            "query": "Has it increased, decreased, or remained unchanged?",
            "images": [],  # No images attached in turn 2! Inherits from turn 1
            "session_id": "sess_multi_turn_demo"
        },
        {
            "title": "Scenario 6: Ambiguity / Missing Input Edge Case",
            "query": "What changed between these two dates, and where did the change occur?",
            "images": [
                {
                    "image_id": "single_img",
                    "file_name": "solo_image.tif",
                    "format": "GeoTIFF",
                    "modality": "OPTICAL"
                }
            ],  # Only 1 image provided for a bi-temporal change query!
            "session_id": "sess_demo_ambiguity"
        }
    ]

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(f"  {scenario['title']}")
        print("=" * 80)
        print(f"Raw User Query      : '{scenario['query']}'")
        print(f"Input Images Count  : {len(scenario['images'])}")

        sqo = pipeline.process_query(
            raw_query=scenario['query'],
            image_inputs=scenario['images'],
            project_context_dict=project_context,
            session_id=scenario['session_id']
        )

        print("\n[Structured Query Object Summary]")
        print(f"  - Query ID                 : {sqo.query_id}")
        print(f"  - Effective Resolved Query : {sqo.effective_resolved_query}")
        print(f"  - Primary Task             : {sqo.task_classification.primary_task.value} (Confidence: {sqo.task_classification.confidence:.2f})")
        print(f"  - Target Agent Family      : {sqo.routing_metadata.target_agent_family}")
        print(f"  - Specialist Tools         : {', '.join(sqo.task_classification.required_specialist_tools)}")
        print(f"  - Benchmark Reference      : {sqo.task_classification.benchmark_reference}")
        print(f"  - Extracted Target Classes : {', '.join(sqo.extracted_entities.target_classes) or 'None'}")
        print(f"  - Coreference Resolved?    : {sqo.extracted_entities.is_coreference_resolved} (Inherited turn: {sqo.extracted_entities.inherited_from_turn})")
        print(f"  - Compatibility Status     : {sqo.input_compatibility.status}")
        print(f"  - Temporal Structure       : {sqo.input_compatibility.temporal_structure}")
        print(f"  - Is Ambiguous / Issues?   : {sqo.ambiguity_report.is_ambiguous} ({len(sqo.ambiguity_report.issues)} issue(s))")

        if sqo.ambiguity_report.issues:
            print("\n  [Ambiguity & Incompatibility Issues]")
            for issue in sqo.ambiguity_report.issues:
                print(f"    - ISSUE: {issue}")
            for rem in sqo.ambiguity_report.suggested_remediations:
                print(f"    - REMEDIATION: {rem}")

        print("\n  [Auditable Execution Trace Steps]")
        for step in sqo.execution_trace:
            print(f"    |- [{step.step}] ({step.status}): {step.details}")

    print("\n" + "=" * 80)
    print("                  DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()
