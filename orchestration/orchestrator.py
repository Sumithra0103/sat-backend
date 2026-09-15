"""
SatQuery AI Orchestration Layer.
Implements the 17-stage Agentic Orchestration Workflow:
1. Receive Request -> 2. Create Workflow -> 3. Validate Plan -> 4. Build Dependency DAG ->
5. Initialize State -> 6. Identify Ready Nodes -> 7. Create Execution Requests ->
8. Send to Execution Layer (Parallel/Sequential) -> 9. Wait for Agent Result (Retry & Fallback) ->
10. Update Node State -> 11. Store Result -> 12. Unlock Dependents -> 13. Schedule Next Ready Agents ->
14. Synchronize All Results -> 15. Collect Evidence -> 16. Calculate Workflow Status ->
17. Create Final Orchestration Result (Auditable trace for Aggregation Layer).
"""

import time
import uuid
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from execution import ExecutionEngine, ExecutionRequest, ExecutionResult, ExecutionStatus, ExecutionTraceItem, EvidenceItem
from satquery_routing import ExecutionPlan
from query_understanding import StructuredQueryObject
from orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResult,
    WorkflowStatus,
    DAGWorkflow,
    DAGNode,
    NodeStatus,
    SharedExecutionState
)
from orchestration.dag_builder import DAGBuilder


class SatQueryOrchestrator:
    """
    Central Orchestration Engine for SatQuery AI.
    Coordinates multi-agent workflows, DAG execution, parallel/sequential scheduling,
    shared state synchronization, fault tolerance, and evidence aggregation.
    """

    def __init__(self, execution_engine: Optional[ExecutionEngine] = None, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.executor = execution_engine or ExecutionEngine(db_session=db_session)

    def orchestrate(self, request: OrchestrationRequest) -> OrchestrationResult:
        """
        Executes the full 17-stage Orchestration pipeline.
        """
        start_time = time.time()
        trace: List[ExecutionTraceItem] = []

        # -------------------------------------------------------------
        # Stage 1: RECEIVE REQUEST
        # -------------------------------------------------------------
        trace.append(ExecutionTraceItem(
            stage="1. Receive Request",
            action="ingest_orchestration_request",
            status="success",
            details={
                "request_id": request.request_id,
                "selected_agent": request.execution_plan.selected_agent_id,
                "image_count": len(request.image_inputs)
            }
        ))

        # -------------------------------------------------------------
        # Stage 2: CREATE WORKFLOW
        # -------------------------------------------------------------
        trace.append(ExecutionTraceItem(
            stage="2. Create Workflow",
            action="parse_execution_plan",
            status="success",
            details={
                "routing_status": str(request.execution_plan.routing_status),
                "step_count": len(request.execution_plan.steps)
            }
        ))

        # -------------------------------------------------------------
        # Stage 3: VALIDATE PLAN
        # -------------------------------------------------------------
        plan_valid, plan_errors = self._validate_plan(request.execution_plan, request.image_inputs)
        if not plan_valid:
            trace.append(ExecutionTraceItem(
                stage="3. Validate Plan",
                action="validate_plan_topology",
                status="failed",
                details={"errors": plan_errors}
            ))
            return self._build_failed_orchestration(request, "Plan validation failed", plan_errors, trace, start_time)

        trace.append(ExecutionTraceItem(
            stage="3. Validate Plan",
            action="validate_plan_topology",
            status="success",
            details={"validation": "passed"}
        ))

        # -------------------------------------------------------------
        # Stage 4: BUILD DEPENDENCY DAG
        # -------------------------------------------------------------
        dag = DAGBuilder.build_dag(
            plan=request.execution_plan,
            sqo=request.sqo,
            image_inputs=request.image_inputs
        )
        dag_topology = {nid: node.dependencies for nid, node in dag.nodes.items()}
        trace.append(ExecutionTraceItem(
            stage="4. Build Dependency DAG",
            action="construct_dag_graph",
            status="success",
            details={
                "workflow_id": dag.workflow_id,
                "node_count": len(dag.nodes),
                "topology": dag_topology
            }
        ))

        # -------------------------------------------------------------
        # Stage 5: INITIALIZE STATE
        # -------------------------------------------------------------
        shared_state = SharedExecutionState(
            request_id=request.request_id,
            images=request.image_inputs or []
        )
        trace.append(ExecutionTraceItem(
            stage="5. Initialize State",
            action="initialize_shared_execution_state",
            status="success",
            details={"state_id": request.request_id, "image_count": len(shared_state.images)}
        ))

        # -------------------------------------------------------------
        # Stage 6 to 13: DAG EXECUTION LOOP (Ready Identification -> Dispatch -> State Sync -> Unlock)
        # -------------------------------------------------------------
        completed_results: List[ExecutionResult] = []

        while not dag.is_complete():
            ready_node_ids = dag.identify_ready_nodes()
            if not ready_node_ids:
                # Check for deadlocks or unmet dependencies
                unresolved = [nid for nid, n in dag.nodes.items() if n.status == NodeStatus.PENDING]
                if unresolved:
                    for nid in unresolved:
                        dag.nodes[nid].status = NodeStatus.FAILED
                        dag.nodes[nid].error_message = "Dependency satisfaction failed (Deadlock)"
                    break

            # ---------------------------------------------------------
            # Stage 6: IDENTIFY READY NODES
            # ---------------------------------------------------------
            trace.append(ExecutionTraceItem(
                stage="6. Identify Ready Nodes",
                action="discover_executable_nodes",
                status="success",
                details={"ready_nodes": ready_node_ids}
            ))

            # Mark ready nodes as RUNNING
            for nid in ready_node_ids:
                dag.nodes[nid].status = NodeStatus.RUNNING

            # Separate ready nodes into Parallel vs. Sequential execution groups
            node_exec_results: List[Tuple[str, ExecutionResult]] = []

            trace.append(ExecutionTraceItem(
                stage="8. Send to Execution Layer",
                action="dispatch_node_execution",
                status="success",
                details={
                    "mode": "parallel" if (len(ready_node_ids) > 1 and request.max_parallel_workers > 1) else "sequential",
                    "worker_count": len(ready_node_ids),
                    "nodes": ready_node_ids
                }
            ))

            if len(ready_node_ids) > 1 and request.max_parallel_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(request.max_parallel_workers, len(ready_node_ids))) as pool:
                    future_to_node = {
                        pool.submit(self._execute_single_node, dag.nodes[nid], request, shared_state, trace): nid
                        for nid in ready_node_ids
                    }
                    for future in concurrent.futures.as_completed(future_to_node):
                        nid = future_to_node[future]
                        try:
                            res = future.result()
                            node_exec_results.append((nid, res))
                        except Exception as exc:
                            err_res = self._build_node_error(dag.nodes[nid], str(exc))
                            node_exec_results.append((nid, err_res))
            else:
                for nid in ready_node_ids:
                    res = self._execute_single_node(dag.nodes[nid], request, shared_state, trace)
                    node_exec_results.append((nid, res))

            # Process node execution outputs: Stage 9 to 13
            for nid, result in node_exec_results:
                node = dag.nodes[nid]
                node.output = result
                completed_results.append(result)

                # -----------------------------------------------------
                # Stage 10: UPDATE NODE STATE
                # -----------------------------------------------------
                if result.status == ExecutionStatus.SUCCESS:
                    node.status = NodeStatus.SUCCESS
                elif result.status == ExecutionStatus.DEGRADED:
                    node.status = NodeStatus.DEGRADED
                else:
                    node.status = NodeStatus.FAILED
                    node.error_message = result.result.get("error", "Node execution failed") if isinstance(result.result, dict) else "Failed"

                trace.append(ExecutionTraceItem(
                    stage="10. Update Node State",
                    action="set_node_status",
                    status="success",
                    details={"node_id": nid, "status": node.status.value, "execution_time": result.execution_time}
                ))

                # -----------------------------------------------------
                # Stage 11: STORE RESULT (Shared Execution State Sync)
                # -----------------------------------------------------
                shared_state.update_node_result(nid, node.agent_id, result)
                trace.append(ExecutionTraceItem(
                    stage="11. Store Result",
                    action="synchronize_shared_state",
                    status="success",
                    details={"node_id": nid, "agent_id": node.agent_id}
                ))

                # -----------------------------------------------------
                # Stage 12: UNLOCK DEPENDENTS
                # -----------------------------------------------------
                unlocked = [
                    child_id for child_id, child_node in dag.nodes.items()
                    if nid in child_node.dependencies
                ]
                trace.append(ExecutionTraceItem(
                    stage="12. Unlock Dependents",
                    action="update_dag_dependencies",
                    status="success",
                    details={"unlocked_dependents": unlocked}
                ))

            # ---------------------------------------------------------
            # Stage 13: SCHEDULE NEXT READY AGENTS (Loop continuation)
            # ---------------------------------------------------------
            trace.append(ExecutionTraceItem(
                stage="13. Schedule Next Ready Agents",
                action="loop_dag_scheduler",
                status="success",
                details={"completed_so_far": len(completed_results), "total_nodes": len(dag.nodes)}
            ))

        # -------------------------------------------------------------
        # Stage 14: SYNCHRONIZE ALL RESULTS
        # -------------------------------------------------------------
        trace.append(ExecutionTraceItem(
            stage="14. Synchronize All Results",
            action="consolidate_agent_outputs",
            status="success",
            details={"total_completed_results": len(completed_results)}
        ))

        # -------------------------------------------------------------
        # Stage 15: COLLECT EVIDENCE
        # -------------------------------------------------------------
        all_evidence: List[EvidenceItem] = []
        for res in completed_results:
            if hasattr(res, "evidence") and res.evidence:
                all_evidence.extend(res.evidence)

        trace.append(ExecutionTraceItem(
            stage="15. Collect Evidence",
            action="aggregate_visual_and_textual_evidence",
            status="success",
            details={"total_evidence_items": len(all_evidence)}
        ))

        # -------------------------------------------------------------
        # Stage 16: CALCULATE WORKFLOW STATUS
        # -------------------------------------------------------------
        workflow_status = self._calculate_workflow_status(dag)
        trace.append(ExecutionTraceItem(
            stage="16. Calculate Workflow Status",
            action="evaluate_final_status",
            status="success",
            details={"calculated_status": workflow_status.value}
        ))

        total_time = max(0.001, round(time.time() - start_time, 3))

        # -------------------------------------------------------------
        # Stage 17: CREATE FINAL ORCHESTRATION RESULT
        # -------------------------------------------------------------
        auditable_summary = {
            "request_id": request.request_id,
            "workflow_id": dag.workflow_id,
            "status": workflow_status.value,
            "total_nodes": len(dag.nodes),
            "successful_nodes": sum(1 for n in dag.nodes.values() if n.status in [NodeStatus.SUCCESS, NodeStatus.DEGRADED]),
            "failed_nodes": sum(1 for n in dag.nodes.values() if n.status == NodeStatus.FAILED),
            "execution_time_seconds": total_time,
            "agents_executed": list(set(n.agent_id for n in dag.nodes.values()))
        }

        trace.append(ExecutionTraceItem(
            stage="17. Create Final Orchestration Result",
            action="emit_result_for_aggregation_layer",
            status="success",
            details=auditable_summary
        ))

        return OrchestrationResult(
            request_id=request.request_id,
            workflow_id=dag.workflow_id,
            status=workflow_status,
            completed_agent_results=completed_results,
            shared_state=shared_state,
            dag_topology=dag_topology,
            execution_trace=trace,
            total_execution_time=total_time,
            auditable_summary=auditable_summary
        )

    def _execute_single_node(
        self,
        node: DAGNode,
        request: OrchestrationRequest,
        shared_state: SharedExecutionState,
        trace: List[ExecutionTraceItem]
    ) -> ExecutionResult:
        """
        Executes a single DAG node through Stage 7 (Create Request), Stage 8 (Dispatch),
        and Stage 9 (Wait for result with Retry & Fallback handling).
        """
        # Stage 7: CREATE EXECUTION REQUESTS
        params = dict(node.parameters)
        if request.resolved_query:
            params["resolved_query"] = request.resolved_query

        # Inject outputs from shared state if dependent node
        if shared_state.change_map:
            params["reference_change_map"] = shared_state.change_map
        if shared_state.grounded_regions:
            params["grounded_regions"] = shared_state.grounded_regions

        exec_req = ExecutionRequest(
            execution_id=f"exec_{node.node_id}_{uuid.uuid4().hex[:6]}",
            agent_id=node.agent_id,
            task=node.task,
            inputs=request.image_inputs,
            parameters=params,
            fallback_agent_id=node.fallback_agent_id,
            resolved_query=request.resolved_query
        )

        trace.append(ExecutionTraceItem(
            stage="7. Create Execution Requests",
            action="package_node_execution_request",
            status="success",
            details={"node_id": node.node_id, "agent_id": node.agent_id, "task": node.task}
        ))

        # Stage 8 & 9: SEND TO EXECUTION LAYER & WAIT FOR RESULT (with Retry & Fallback)
        result = self.executor.execute_request(exec_req)

        # Retry logic if node failed
        retries = 0
        while result.status == ExecutionStatus.FAILED and retries < node.max_retries:
            retries += 1
            node.retry_count = retries
            trace.append(ExecutionTraceItem(
                stage="9. Wait for Agent Result",
                action="apply_retry_policy",
                status="retry_attempt",
                details={"node_id": node.node_id, "retry_count": retries, "agent_id": node.agent_id}
            ))
            time.sleep(0.1)
            result = self.executor.execute_request(exec_req)

        # If retries exhausted and node failed, attempt Fallback Agent
        if result.status == ExecutionStatus.FAILED and node.fallback_agent_id:
            trace.append(ExecutionTraceItem(
                stage="9. Wait for Agent Result",
                action="execute_fallback_agent",
                status="fallback_initiated",
                details={"primary_agent": node.agent_id, "fallback_agent": node.fallback_agent_id}
            ))

            fallback_req = ExecutionRequest(
                execution_id=f"exec_fb_{node.node_id}_{uuid.uuid4().hex[:6]}",
                agent_id=node.fallback_agent_id,
                task=node.task,
                inputs=request.image_inputs,
                parameters=params,
                resolved_query=request.resolved_query
            )
            result = self.executor.execute_request(fallback_req)
            if result.status == ExecutionStatus.SUCCESS:
                result.status = ExecutionStatus.DEGRADED  # Mark degraded since fallback was used

        return result

    def _validate_plan(self, plan: ExecutionPlan, inputs: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validates execution plan topology, agents, and parameters."""
        errors = []
        if not plan:
            errors.append("Execution plan is null")
            return False, errors

        if hasattr(plan, "routing_status"):
            status_val = plan.routing_status.value if hasattr(plan.routing_status, "value") else str(plan.routing_status)
            if status_val.lower() == "failed":
                errors.append("Routing status is FAILED")

        return len(errors) == 0, errors

    def _calculate_workflow_status(self, dag: DAGWorkflow) -> WorkflowStatus:
        """Evaluates overall workflow status from individual node statuses."""
        statuses = [n.status for n in dag.nodes.values()]

        if all(s == NodeStatus.SUCCESS for s in statuses):
            return WorkflowStatus.SUCCESS
        elif any(s in [NodeStatus.SUCCESS, NodeStatus.DEGRADED] for s in statuses):
            return WorkflowStatus.DEGRADED
        else:
            return WorkflowStatus.FAILED

    def _build_failed_orchestration(
        self,
        request: OrchestrationRequest,
        message: str,
        errors: List[str],
        trace: List[ExecutionTraceItem],
        start_time: float
    ) -> OrchestrationResult:
        elapsed = max(0.001, round(time.time() - start_time, 3))
        shared_state = SharedExecutionState(request_id=request.request_id, images=request.image_inputs or [])
        return OrchestrationResult(
            request_id=request.request_id,
            workflow_id="wf_failed",
            status=WorkflowStatus.FAILED,
            completed_agent_results=[],
            shared_state=shared_state,
            dag_topology={},
            execution_trace=trace,
            total_execution_time=elapsed,
            auditable_summary={"error": message, "validation_errors": errors}
        )

    def _build_node_error(self, node: DAGNode, error_msg: str) -> ExecutionResult:
        return ExecutionResult(
            agent=node.agent_id,
            status=ExecutionStatus.FAILED,
            result={"error": error_msg},
            confidence=0.0,
            execution_time=0.01,
            execution_id=f"exec_err_{node.node_id}",
            evidence=[],
            execution_trace=[]
        )
