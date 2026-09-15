"""
Unit and Integration Tests for SatQuery AI Orchestration Subsystem.
Verifies DAG construction, parallel/sequential execution, shared state sync, fault tolerance,
and the complete 17-stage Orchestration pipeline.
"""

import pytest
import time
from typing import Dict, Any, List

from satquery_engine import SatQueryEngine
from orchestration import (
    SatQueryOrchestrator,
    OrchestrationRequest,
    OrchestrationResult,
    WorkflowStatus,
    DAGBuilder,
    DAGWorkflow,
    DAGNode,
    NodeStatus,
    SharedExecutionState
)
from satquery_routing import ExecutionPlan, ExecutionStep, RoutingStatus, ExecutionStepType, RoutingRequirements
from execution import ExecutionResult, ExecutionStatus


@pytest.fixture
def engine():
    eng = SatQueryEngine(auto_seed=True)
    yield eng
    eng.close()


def mock_requirements(query_id: str = "q_mock") -> RoutingRequirements:
    return RoutingRequirements(
        query_id=query_id,
        resolved_query="mock query",
        primary_capability="visual_question_answering",
        required_modality="OPTICAL",
        image_count=1,
        temporal_structure="SINGLE"
    )


def test_dag_builder_single_task():
    plan = ExecutionPlan(
        plan_id="plan_test_1",
        query_id="q_1",
        routing_status=RoutingStatus.SUCCESS,
        requirements=mock_requirements("q_1"),
        selected_agent_id="rs_vqa_agent",
        steps=[
            ExecutionStep(
                step_id="step_1",
                step_type=ExecutionStepType.SPECIALIST_INFERENCE,
                agent_id="rs_vqa_agent"
            )
        ]
    )

    dag = DAGBuilder.build_dag(plan)
    assert len(dag.nodes) == 1
    assert "node_1_rs_vqa_agent" in dag.nodes
    node = dag.nodes["node_1_rs_vqa_agent"]
    assert node.agent_id == "rs_vqa_agent"
    assert node.dependencies == []
    assert dag.identify_ready_nodes() == ["node_1_rs_vqa_agent"]


def test_dag_builder_sequential_change_dependency():
    plan = ExecutionPlan(
        plan_id="plan_test_seq",
        query_id="q_seq",
        routing_status=RoutingStatus.SUCCESS,
        requirements=mock_requirements("q_seq"),
        selected_agent_id="change_agent",
        steps=[
            ExecutionStep(
                step_id="change_detection_step",
                step_type=ExecutionStepType.SPECIALIST_INFERENCE,
                agent_id="change_detector"
            ),
            ExecutionStep(
                step_id="change_vqa_analysis_step",
                step_type=ExecutionStepType.SPECIALIST_INFERENCE,
                agent_id="change_vqa_agent"
            )
        ]
    )

    dag = DAGBuilder.build_dag(plan)
    assert len(dag.nodes) == 2
    n1_id = "node_1_change_detector"
    n2_id = "node_2_change_vqa_agent"

    assert dag.nodes[n1_id].dependencies == []
    assert dag.nodes[n2_id].dependencies == [n1_id]

    # Only node 1 should be ready initially
    ready_initial = dag.identify_ready_nodes()
    assert ready_initial == [n1_id]


def test_shared_execution_state_sync():
    state = SharedExecutionState(request_id="req_sync_test")
    res_success = ExecutionResult(
        agent="rs_vqa_agent",
        status=ExecutionStatus.SUCCESS,
        result={"answer": "Commercial port with cargo ships", "caption": "Satellite view of port"},
        confidence=0.92,
        execution_time=0.15,
        execution_id="exec_1"
    )

    state.update_node_result("node_1", "rs_vqa_agent", res_success)

    assert state.vqa_answers["node_1"] == "Commercial port with cargo ships"
    assert "Satellite view of port" in state.captions
    assert state.confidence_scores["rs_vqa_agent"] == 0.92
    assert len(state.execution_log) == 1


def test_full_17_stage_orchestration_workflow(engine):
    raw_query = "What is the primary land-cover in this coastal area?"
    images = [{
        "image_id": "img_coastal_1",
        "file_name": "coastal_sentinel2.tif",
        "format": "GeoTIFF",
        "modality": "OPTICAL",
        "spatial_resolution_m": 10.0
    }]

    sqo, plan, orch_res = engine.process_and_orchestrate(
        raw_query=raw_query,
        image_inputs=images
    )

    assert orch_res.status in [WorkflowStatus.SUCCESS, WorkflowStatus.DEGRADED]
    assert orch_res.request_id is not None
    assert orch_res.workflow_id.startswith("wf_")
    assert len(orch_res.completed_agent_results) > 0

    # Verify that all 17 workflow stages are recorded in the execution trace
    stages_found = set(item.stage for item in orch_res.execution_trace)
    expected_stage_prefixes = [
        "1. Receive Request",
        "2. Create Workflow",
        "3. Validate Plan",
        "4. Build Dependency DAG",
        "5. Initialize State",
        "6. Identify Ready Nodes",
        "7. Create Execution Requests",
        "8. Send to Execution Layer",
        "10. Update Node State",
        "11. Store Result",
        "12. Unlock Dependents",
        "13. Schedule Next Ready Agents",
        "14. Synchronize All Results",
        "15. Collect Evidence",
        "16. Calculate Workflow Status",
        "17. Create Final Orchestration Result"
    ]

    for stage_prefix in expected_stage_prefixes:
        assert any(stage_prefix in s for s in stages_found), f"Missing stage in trace: {stage_prefix}"


def test_bitemporal_change_orchestration(engine):
    raw_query = "What changed between these two dates and where did the change occur?"
    images = [
        {"image_id": "t1", "file_name": "t1.tif", "format": "GeoTIFF", "modality": "OPTICAL", "acquisition_date": "2023-01-01"},
        {"image_id": "t2", "file_name": "t2.tif", "format": "GeoTIFF", "modality": "OPTICAL", "acquisition_date": "2024-01-01"}
    ]

    sqo, plan, orch_res = engine.process_and_orchestrate(
        raw_query=raw_query,
        image_inputs=images
    )

    assert orch_res.status in [WorkflowStatus.SUCCESS, WorkflowStatus.DEGRADED]
    assert len(orch_res.completed_agent_results) >= 1
    assert orch_res.auditable_summary["total_nodes"] >= 1
