"""
Orchestration Data Schemas for SatQuery AI.
Defines DAG Nodes, Node Statuses, Shared Execution State, Orchestration Requests, and Results.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
import time
import uuid

from execution.schemas import ExecutionResult, ExecutionRequest, ExecutionStatus, EvidenceItem, ExecutionTraceItem
from satquery_routing import ExecutionPlan, ExecutionStep
from query_understanding import StructuredQueryObject


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DAGNode(BaseModel):
    """Represents a single task node in the workflow dependency DAG."""
    node_id: str
    agent_id: str
    task: str
    dependencies: List[str] = Field(default_factory=list, description="List of parent node IDs")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    fallback_agent_id: Optional[str] = None
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    output: Optional[ExecutionResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


class DAGWorkflow(BaseModel):
    """Represents the complete execution DAG of agent tasks."""
    workflow_id: str = Field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    nodes: Dict[str, DAGNode] = Field(default_factory=dict)
    topological_order: List[str] = Field(default_factory=list)

    def identify_ready_nodes(self) -> List[str]:
        """Finds all nodes whose dependencies are satisfied and status is PENDING."""
        ready = []
        for node_id, node in self.nodes.items():
            if node.status == NodeStatus.PENDING:
                deps_satisfied = all(
                    self.nodes[parent_id].status in [NodeStatus.SUCCESS, NodeStatus.DEGRADED]
                    for parent_id in node.dependencies
                    if parent_id in self.nodes
                )
                if deps_satisfied:
                    ready.append(node_id)
        return ready

    def is_complete(self) -> bool:
        """Returns True if all nodes have reached a terminal state."""
        terminal_statuses = {NodeStatus.SUCCESS, NodeStatus.DEGRADED, NodeStatus.FAILED, NodeStatus.SKIPPED}
        return all(node.status in terminal_statuses for node in self.nodes.values())


class SharedExecutionState(BaseModel):
    """Thread-safe shared execution state storing intermediate and final outputs across agent steps."""
    request_id: str
    images: List[Dict[str, Any]] = Field(default_factory=list)
    change_map: Optional[str] = None
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list)
    land_cover_map: Optional[Dict[str, Any]] = None
    vqa_answers: Dict[str, Any] = Field(default_factory=dict)
    captions: List[str] = Field(default_factory=list)
    grounded_regions: List[Dict[str, Any]] = Field(default_factory=list)
    optical_sar_features: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)

    def update_node_result(self, node_id: str, agent_id: str, result: ExecutionResult):
        """Updates shared state with outputs from executed agent node."""
        self.agent_outputs[node_id] = result.result
        self.confidence_scores[agent_id] = result.confidence

        res_data = result.result or {}
        # Parse task-specific output into structured shared state fields
        if "change_map" in res_data:
            self.change_map = str(res_data.get("change_map"))
        if "detected_objects" in res_data:
            self.detected_objects.extend(res_data.get("detected_objects", []))
        if "bounding_boxes" in res_data:
            self.grounded_regions.extend(res_data.get("bounding_boxes", []))
        if "caption" in res_data:
            self.captions.append(str(res_data.get("caption")))
        if "answer" in res_data:
            self.vqa_answers[node_id] = res_data.get("answer")
        if "land_cover_summary" in res_data or "class_distribution" in res_data:
            self.land_cover_map = res_data

        if result.status == ExecutionStatus.FAILED:
            self.errors.append({
                "node_id": node_id,
                "agent_id": agent_id,
                "error": res_data.get("error", "Unknown error"),
                "details": result.error_details
            })

        self.execution_log.append({
            "timestamp": time.time(),
            "node_id": node_id,
            "agent_id": agent_id,
            "status": result.status.value,
            "execution_time": result.execution_time
        })


class OrchestrationRequest(BaseModel):
    """Input request sent from Router to Orchestrator."""
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    execution_plan: ExecutionPlan
    sqo: Optional[StructuredQueryObject] = None
    image_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    resolved_query: Optional[str] = None
    max_parallel_workers: int = 4


class OrchestrationResult(BaseModel):
    """Final output produced by Orchestrator for the Aggregation Layer."""
    request_id: str
    workflow_id: str
    status: WorkflowStatus
    completed_agent_results: List[ExecutionResult] = Field(default_factory=list)
    shared_state: SharedExecutionState
    dag_topology: Dict[str, List[str]] = Field(default_factory=dict, description="Adjacency list of DAG")
    execution_trace: List[ExecutionTraceItem] = Field(default_factory=list)
    total_execution_time: float = 0.0
    auditable_summary: Dict[str, Any] = Field(default_factory=dict)
