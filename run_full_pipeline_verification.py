"""
SatQuery AI - Full Pipeline 7-Stage Execution & Verification Runner
====================================================================
Executes and displays outputs for all 7 architectural steps:
1. Query Understanding
2. Router
3. Agent Registry
4. Execution
5. Orchestration (multi-agent)
6. Trace
7. Aggregation
(Excludes Step 8: API per user instruction)
"""

import sys
import json
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
for _sub in [_ROOT_DIR, _ROOT_DIR / "agent registry", _ROOT_DIR / "routing", _ROOT_DIR / "query understanding", _ROOT_DIR / "trace"]:
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from satquery_engine import SatQueryEngine
from trace.integration import TracedSatQueryEngine


def main():
    print("=" * 95)
    print("   SATQUERY AI ARCHITECTURAL PIPELINE EXECUTION (STEPS 1 TO 7)")
    print("=" * 95)

    engine = SatQueryEngine(auto_seed=True)
    traced_engine = TracedSatQueryEngine(engine=engine)

    query = (
        "Use the co-registered optical and SAR bi-temporal image pair to identify what changed "
        "between 2022 and 2024, highlight the water and built-up regions, and state if built-up area increased."
    )

    image_inputs = [
        {
            "image_id": "sentinel2_opt_2022",
            "file_name": "sentinel2_optical_t1.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "sensor_type": "Sentinel-2",
            "timestamp": "2022-03-15T10:00:00Z",
            "bounds": [72.8, 18.9, 72.9, 19.0]
        },
        {
            "image_id": "sentinel1_sar_2022",
            "file_name": "sentinel1_sar_t1.tif",
            "format": "GeoTIFF",
            "modality": "SAR",
            "sensor_type": "Sentinel-1",
            "timestamp": "2022-03-15T10:00:00Z",
            "bounds": [72.8, 18.9, 72.9, 19.0]
        },
        {
            "image_id": "sentinel2_opt_2024",
            "file_name": "sentinel2_optical_t2.tif",
            "format": "GeoTIFF",
            "modality": "OPTICAL",
            "sensor_type": "Sentinel-2",
            "timestamp": "2024-03-15T10:00:00Z",
            "bounds": [72.8, 18.9, 72.9, 19.0]
        },
        {
            "image_id": "sentinel1_sar_2024",
            "file_name": "sentinel1_sar_t2.tif",
            "format": "GeoTIFF",
            "modality": "SAR",
            "sensor_type": "Sentinel-1",
            "timestamp": "2024-03-15T10:00:00Z",
            "bounds": [72.8, 18.9, 72.9, 19.0]
        }
    ]

    project_context = {
        "project_id": "proj_isro_satquery_pipeline",
        "user_id": "remote_sensing_lead",
        "user_role": "ANALYST"
    }

    print(f"\nUser Query     : '{query}'")
    print(f"Attached Inputs: {len(image_inputs)} images (Optical + SAR Bi-Temporal)")

    # Execute full workflow via Traced Engine wrapper
    sqo, plan, results, final_trace = traced_engine.process_and_execute(
        raw_query=query,
        image_inputs=image_inputs,
        project_context=project_context,
        session_id="sess_full_pipeline_001",
        request_id="req_full_pipeline_verification_2026"
    )

    # -------------------------------------------------------------------------
    # STEP 1: QUERY UNDERSTANDING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 1: QUERY UNDERSTANDING")
    print("-" * 95)
    print(f"• Query ID             : {sqo.query_id}")
    print(f"• Resolved Query       : '{sqo.effective_resolved_query}'")
    print(f"• Primary Task Intent  : {sqo.task_classification.primary_task.value} (Confidence: {sqo.task_classification.confidence:.2f})")
    print(f"• Benchmark Reference  : {sqo.task_classification.benchmark_reference}")
    print(f"• Target Entities      : {sqo.extracted_entities.target_classes}")
    print(f"• Spatial Actions      : {sqo.extracted_entities.spatial_actions}")
    print(f"• Coreference Resolved : {sqo.extracted_entities.is_coreference_resolved}")
    print(f"• Temporal Structure   : {sqo.input_compatibility.temporal_structure}")
    print(f"• Input Compatibility  : {sqo.input_compatibility.status} (Compatible: {sqo.input_compatibility.is_compatible})")

    # -------------------------------------------------------------------------
    # STEP 2: ROUTER
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 2: ROUTER")
    print("-" * 95)
    print(f"• Routing Status       : {plan.routing_status.value.upper()}")
    print(f"• Execution Plan ID    : {plan.plan_id}")
    if plan.auditable_summary:
        print(f"• Primary Selected Agent: {plan.auditable_summary.primary_agent_name} ({plan.auditable_summary.primary_agent_id})")
        print(f"• Routing Confidence   : {plan.auditable_summary.routing_confidence:.4f}")
        print(f"• Benchmark Match      : {plan.auditable_summary.benchmark_dataset}")
    print("• Generated Execution Steps:")
    for st in plan.steps:
        print(f"    - Step ID: {st.step_id} | Type: {st.step_type.value:20} | Agent: {st.agent_name} | Target: {st.endpoint_url}")
    if plan.fallback_routes:
        print("• Fallback Routes Configured:")
        for fb in plan.fallback_routes:
            print(f"    - Trigger: {fb.trigger_condition} -> Fallback Agent: {fb.fallback_agent_id}")

    # -------------------------------------------------------------------------
    # STEP 3: AGENT REGISTRY
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 3: AGENT REGISTRY")
    print("-" * 95)
    agents = engine.get_registered_agents()
    print(f"• Total Registered Agents in Database Catalog: {len(agents)}")
    for ag in agents:
        caps = [c.value for c in ag.capabilities]
        print(f"    - Agent ID: {ag.agent_id:25} | Name: {ag.name:45} | Status: {ag.status.value}")
        print(f"      Capabilities: {caps} | Adapters: {ag.metadata.get('training_dataset', 'Standard')}")

    # -------------------------------------------------------------------------
    # STEP 4: EXECUTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 4: EXECUTION")
    print("-" * 95)
    for idx, res in enumerate(results):
        print(f"• Execution Result #{idx+1}:")
        print(f"    - Agent ID         : {res.agent}")
        print(f"    - Status           : {res.status.value.upper()}")
        print(f"    - Confidence Score : {res.confidence:.4f}")
        print(f"    - Execution Time   : {res.execution_time:.4f}s")
        print(f"    - JSON Payload Result: {json.dumps(res.result)}")
        if res.evidence:
            print(f"    - Grounded Evidence Items ({len(res.evidence)}):")
            for ev in res.evidence:
                print(f"        * [{ev.evidence_type.value}] {ev.label} -> {ev.file_path or ev.description}")

    # -------------------------------------------------------------------------
    # STEP 5: ORCHESTRATION (MULTI-AGENT)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 5: ORCHESTRATION (MULTI-AGENT)")
    print("-" * 95)
    orch_res = engine.orchestrate_plan(
        plan=plan,
        sqo=sqo,
        image_inputs=image_inputs,
        resolved_query=sqo.effective_resolved_query
    )

    print(f"• Request / Workflow ID: {orch_res.request_id}")
    print(f"• Workflow Status     : {orch_res.status.value.upper()}")
    print(f"• Total Workflow Time : {orch_res.total_execution_time:.4f}s")
    print(f"• DAG Topology Graph  : {orch_res.dag_topology}")
    print(f"• Completed Agents    : {[res.agent for res in orch_res.completed_agent_results]}")
    print(f"• Shared Execution State Keys: {list(orch_res.shared_state.model_dump().keys())}")

    # -------------------------------------------------------------------------
    # STEP 6: TRACE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 6: TRACE")
    print("-" * 95)
    print(f"• Trace ID            : {final_trace.trace_id}")
    print(f"• Request ID          : {final_trace.request_id}")
    print(f"• Trace Status        : {final_trace.status.value}")
    print(f"• Total Duration MS   : {final_trace.total_duration_ms:.2f}ms" if final_trace.total_duration_ms else "• Total Duration MS   : N/A")
    print(f"• Recorded Spans Count: {len(final_trace.spans)}")
    for sp in final_trace.spans:
        print(f"    - Span ID: {sp.span_id} | Component: {sp.component:10} | Name: {sp.name:40} | Status: {sp.status.value}")

    # -------------------------------------------------------------------------
    # STEP 7: AGGREGATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" STEP 7: AGGREGATION")
    print("-" * 95)
    agg_res = engine.aggregator.aggregate(
        raw_query=query,
        input_data=orch_res,
        request_id=orch_res.request_id,
        image_inputs=image_inputs
    )

    print(f"• Request ID                     : {agg_res.request_id}")
    print(f"• Aggregated Composite Confidence: {int(agg_res.aggregated_confidence * 100)}%")
    print(f"• Total Agents Collected         : {agg_res.metrics.total_agents_collected}")
    print(f"• Valid Agents Count             : {agg_res.metrics.valid_agents_count}")
    print(f"• Conflicts Resolved             : {agg_res.metrics.conflicts_resolved}")
    print(f"• Generated Evidence Artifacts   : {len(agg_res.visual_evidence_urls)}")
    for url in agg_res.visual_evidence_urls:
        print(f"    - Artifact Path: {url}")
    print("\n" + "=" * 95)
    print(" FINAL SYNTHESIZED NATURAL LANGUAGE RESPONSE")
    print("=" * 95)
    print(agg_res.final_answer)
    print("=" * 95)

    engine.close()


if __name__ == "__main__":
    main()
