"""
DAG Builder for SatQuery AI Orchestration.
Parses ExecutionPlan and constructs a dependency DAG (DAGWorkflow) for parallel and sequential agent scheduling.
"""

from typing import Dict, Any, List, Optional
from satquery_routing import ExecutionPlan, ExecutionStep
from query_understanding import StructuredQueryObject, TaskType
from orchestration.schemas import DAGNode, DAGWorkflow, NodeStatus


class DAGBuilder:
    """
    Translates an ExecutionPlan into a DAGWorkflow.
    Determines sequential vs. parallel execution based on task semantics and data dependencies.
    """

    @staticmethod
    def build_dag(
        plan: ExecutionPlan,
        sqo: Optional[StructuredQueryObject] = None,
        image_inputs: Optional[List[Dict[str, Any]]] = None
    ) -> DAGWorkflow:
        """
        Builds a DAGWorkflow from an ExecutionPlan.
        """
        workflow = DAGWorkflow()
        fallback_agent_id = None
        if hasattr(plan, "fallback_routes") and plan.fallback_routes:
            fallback_agent_id = plan.fallback_routes[0].fallback_agent_id

        # Determine task nodes from plan steps
        steps = [
            step for step in plan.steps
            if "specialist" in (step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)).lower()
            or "inference" in (step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)).lower()
        ]

        # If steps list is empty, create at least one node based on selected_agent_id
        if not steps:
            task_name = (
                plan.auditable_summary.selected_task
                if (hasattr(plan, "auditable_summary") and plan.auditable_summary)
                else "remote_sensing_analysis"
            )
            node_id = f"node_1_{plan.selected_agent_id}"
            node = DAGNode(
                node_id=node_id,
                agent_id=plan.selected_agent_id,
                task=task_name,
                dependencies=[],
                parameters=plan.parameters or {},
                fallback_agent_id=fallback_agent_id,
                inputs=image_inputs or []
            )
            workflow.nodes[node_id] = node
            workflow.topological_order.append(node_id)
            return workflow

        # Track created nodes by task category for sequential dependency wiring
        change_detection_nodes: List[str] = []
        fusion_nodes: List[str] = []
        previous_node_id: Optional[str] = None

        for idx, step in enumerate(steps):
            step_agent_id = step.agent_id or plan.selected_agent_id
            node_id = f"node_{idx+1}_{step_agent_id}"

            task_name = (
                getattr(step, "description", None)
                or getattr(step, "step_id", None)
                or (plan.auditable_summary.selected_task if (hasattr(plan, "auditable_summary") and plan.auditable_summary) else "remote_sensing_analysis")
            )
            task_lower = str(task_name).lower()

            dependencies: List[str] = []

            # 1. Dependency Rule: Change downstream tasks depend on Change Detection
            if any(k in task_lower for k in ["change_vqa", "change_description", "change_caption", "object_detection_on_changed"]):
                dependencies.extend(change_detection_nodes)
            
            # 2. Dependency Rule: Joint analysis tasks depend on Optical-SAR Fusion
            elif any(k in task_lower for k in ["joint_vqa", "fused_grounding", "joint_analysis"]):
                dependencies.extend(fusion_nodes)

            # 3. If explicit step dependencies exist in plan step
            elif hasattr(step, "depends_on") and step.depends_on:
                dependencies.extend(step.depends_on)

            # Register node
            node = DAGNode(
                node_id=node_id,
                agent_id=step_agent_id,
                task=task_name,
                dependencies=list(set(dependencies)),
                parameters=step.parameters or {},
                fallback_agent_id=fallback_agent_id,
                inputs=image_inputs or [],
                timeout_seconds=step.timeout_seconds if hasattr(step, "timeout_seconds") else 60.0
            )

            if "change_detection" in task_lower or "bitemporal" in task_lower:
                change_detection_nodes.append(node_id)
            if "fusion" in task_lower or "cross_modal" in task_lower:
                fusion_nodes.append(node_id)

            workflow.nodes[node_id] = node
            workflow.topological_order.append(node_id)
            previous_node_id = node_id

        return workflow
