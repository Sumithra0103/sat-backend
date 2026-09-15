"""
SatQuery AI - End-to-End Aggregation Demonstration Script
=========================================================
Demonstrates full pipeline:
Query Understanding -> Agent Registry -> SatQuery Router -> Orchestration -> Execution -> AGGREGATION Layer.
"""

import os
import sys

from satquery_engine import SatQueryEngine


def main():
    print("=" * 80)
    print(" SatQuery AI - End-to-End Aggregation Subsystem Demonstration")
    print("=" * 80)

    # Instantiate unified engine
    engine = SatQueryEngine(auto_seed=True)

    query = (
        "Use the co-registered optical and SAR bi-temporal image pair to identify what changed "
        "between 2022 and 2024, highlight the water and built-up regions, and state if built-up area increased."
    )

    image_inputs = [
        {
            "file_name": "sentinel2_optical_t1.tif",
            "format": "GeoTIFF",
            "modality": "optical",
            "sensor": "Sentinel-2",
            "acquisition_date": "2022-03-15"
        },
        {
            "file_name": "sentinel1_sar_t1.tif",
            "format": "GeoTIFF",
            "modality": "sar",
            "sensor": "Sentinel-1",
            "acquisition_date": "2022-03-15"
        },
        {
            "file_name": "sentinel2_optical_t2.tif",
            "format": "GeoTIFF",
            "modality": "optical",
            "sensor": "Sentinel-2",
            "acquisition_date": "2024-03-15"
        },
        {
            "file_name": "sentinel1_sar_t2.tif",
            "format": "GeoTIFF",
            "modality": "sar",
            "sensor": "Sentinel-1",
            "acquisition_date": "2024-03-15"
        }
    ]

    print(f"\n[1] User Query: '{query}'")
    print(f"[2] Image Inputs: {len(image_inputs)} images (Bi-temporal Optical + SAR pairs)")

    # Execute full process and aggregate
    sqo, plan, orch_res, agg_res = engine.process_and_aggregate(
        raw_query=query,
        image_inputs=image_inputs,
        session_id="demo_session_agg_01"
    )

    print("\n" + "-" * 80)
    print(" STAGE 1: QUERY UNDERSTANDING")
    print("-" * 80)
    print(f"Primary Task Intent: {sqo.task_classification.primary_task.value}")
    print(f"Temporal Structure: {sqo.input_compatibility.temporal_structure}")
    print(f"Compatible Inputs: {sqo.input_compatibility.is_compatible}")


    print("\n" + "-" * 80)
    print(" STAGE 2: ROUTING & ORCHESTRATION")
    print("-" * 80)
    print(f"Routing Status: {plan.routing_status.value}")
    print(f"Orchestration Status: {orch_res.status.value}")
    print(f"Executed Agents: {[res.agent for res in orch_res.completed_agent_results]}")

    print("\n" + "-" * 80)
    print(" STAGE 3: AGGREGATION LAYER OUTPUT")
    print("-" * 80)
    print(f"Request ID: {agg_res.request_id}")
    print(f"Aggregated Confidence: {int(agg_res.aggregated_confidence * 100)}%")
    print(f"Total Agents Collected: {agg_res.metrics.total_agents_collected}")
    print(f"Valid Agents Count: {agg_res.metrics.valid_agents_count}")
    print(f"Conflicts Resolved: {agg_res.metrics.conflicts_resolved}")
    print(f"Visual Evidence Artifacts Generated: {len(agg_res.visual_evidence_urls)}")
    for url in agg_res.visual_evidence_urls:
        print(f"  -> Artifact: {url}")

    print("\n" + "=" * 80)
    print(" FINAL SYNTHESIZED NATURAL LANGUAGE RESPONSE")
    print("=" * 80)
    print(agg_res.final_answer)
    print("=" * 80)

    engine.close()


if __name__ == "__main__":
    main()
