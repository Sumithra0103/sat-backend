"""
SatQuery AI - Unified Agentic Engine
====================================
Seamlessly connects:
1. Query Understanding (Natural Language parsing, Coreference resolution,
   Geospatial format validation, Ambiguity & Incompatibility checking, SQO generation)
2. Agent Registry (Specialist remote sensing model catalog, SQLite database,
   BigEarthNet / RSVQA / VRSBench / CDVQA / ISRO-SAC domain adaptations, health telemetry)
3. SatQuery Router (Multi-criteria scoring, Candidate filtering, Execution plan builder,
   Fault-tolerant fallback routes, Auditable execution trace)
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure paths to ensure all modules are directly importable
_ROOT_DIR = Path(__file__).resolve().parent
_QU_DIR = _ROOT_DIR / "query understanding"
_REG_DIR = _ROOT_DIR / "agent registry"
_ROUTER_DIR = _ROOT_DIR / "routing"

for _p in [str(_ROOT_DIR), str(_QU_DIR), str(_ROUTER_DIR), str(_REG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import Execution Subsystem
from execution import ExecutionEngine, ExecutionResult, ExecutionRequest

# Import Orchestration Subsystem
from orchestration import SatQueryOrchestrator, OrchestrationRequest, OrchestrationResult

# Import Aggregation Subsystem
from aggregation import SatQueryAggregator, AggregationResult


# Import Query Understanding
from query_understanding import (
    QueryUnderstandingPipeline,
    StructuredQueryObject,
    TaskType,
    Modality,
    ImageFormat,
    SensorType,
    ConversationSession
)

# Import Agent Registry
from database.db import init_db, SessionLocal
from services.registry_service import AgentRegistryService
from schemas.agent_schema import (
    AgentCapability,
    InputModality,
    AgentStatus,
    AgentResponse,
    AgentRegisterRequest
)

# Import SatQuery Routing
from satquery_routing import (
    SatQueryRouter,
    ExecutionPlan,
    RoutingStatus,
    RoutingRequirements,
    ExecutionStep,
    ExecutionStepType,
    FallbackRoute,
    CandidateScore
)


class SatQueryEngine:
    """
    Unified Agentic Vision-Language Assistant Engine for Remote Sensing.
    Orchestrates Query Understanding -> Agent Registry -> Router -> Execution Plan.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        cache_ttl_seconds: float = 60.0,
        auto_seed: bool = True
    ):
        """
        Initializes the SatQuery AI engine.
        Sets up the SQLite Agent Registry, Query Understanding pipeline, and Router.
        """
        init_db()
        self.db = SessionLocal()

        if auto_seed:
            AgentRegistryService.seed_mock_agents(self.db)

        self.router = SatQueryRouter(
            db_session=self.db,
            cache_ttl_seconds=cache_ttl_seconds
        )
        self.registry_client = self.router.cache.client if hasattr(self.router.cache, "client") else None
        self.query_pipeline = QueryUnderstandingPipeline()
        self.executor = ExecutionEngine(db_session=self.db)
        self.orchestrator = SatQueryOrchestrator(execution_engine=self.executor, db_session=self.db)
        self.aggregator = SatQueryAggregator()

    def process_query(

        self,
        raw_query: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[StructuredQueryObject, ExecutionPlan]:
        """
        End-to-End Processing Workflow:
        1. Query Understanding: Text sanitization, coreference resolution,
           modality & pair analysis, entity extraction, task classification,
           and ambiguity/compatibility checking.
        2. Routing: Translates SQO into requirements, discovers candidates from
           Agent Registry, filters, scores based on domain adaptation, and builds
           an auditable Execution Plan.

        Args:
            raw_query: Natural language user query.
            image_inputs: List of image metadata dicts (file_name, format, modality, etc.).
            project_context: Project/user context dict (project_id, user_role, etc.).
            session_id: Optional multi-turn conversation session ID.

        Returns:
            Tuple[StructuredQueryObject, ExecutionPlan]: The parsed query object and execution plan.
        """
        # Step 1: Query Understanding
        sqo = self.query_pipeline.process_query(
            raw_query=raw_query,
            image_inputs=image_inputs or [],
            project_context_dict=project_context,
            session_id=session_id
        )

        # Step 2: Agentic Routing
        plan = self.router.route(sqo)

        return sqo, plan

    def execute_plan(
        self,
        plan: ExecutionPlan,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        resolved_query: Optional[str] = None
    ) -> List[ExecutionResult]:
        """
        Executes an ExecutionPlan produced by the SatQueryRouter.
        Orchestrates specialist models, error recovery, and evidence collection.
        """
        return self.executor.execute_plan(
            plan=plan,
            image_inputs=image_inputs,
            resolved_query=resolved_query
        )

    def orchestrate_plan(
        self,
        plan: ExecutionPlan,
        sqo: Optional[StructuredQueryObject] = None,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        resolved_query: Optional[str] = None
    ) -> OrchestrationResult:
        """
        Executes an ExecutionPlan via the 17-stage Agentic Orchestration Engine.
        Manages dependency DAG, parallel/sequential scheduling, state sync, and fault tolerance.
        """
        req = OrchestrationRequest(
            execution_plan=plan,
            sqo=sqo,
            image_inputs=image_inputs or [],
            resolved_query=resolved_query
        )
        return self.orchestrator.orchestrate(req)

    def process_and_execute(
        self,
        raw_query: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[StructuredQueryObject, ExecutionPlan, List[ExecutionResult]]:
        """
        Complete end-to-end pipeline:
        Natural Language Query + Imagery -> SQO -> Router -> Specialist Model Execution -> Evidence.
        """
        sqo, plan = self.process_query(
            raw_query=raw_query,
            image_inputs=image_inputs,
            project_context=project_context,
            session_id=session_id
        )

        results = []
        if plan.routing_status.value in ["success", "degraded"]:
            results = self.execute_plan(
                plan=plan,
                image_inputs=image_inputs,
                resolved_query=sqo.effective_resolved_query
            )

        return sqo, plan, results

    def process_and_orchestrate(
        self,
        raw_query: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[StructuredQueryObject, ExecutionPlan, OrchestrationResult]:
        """
        Complete 17-Stage Orchestration Pipeline:
        Natural Language Query + Imagery -> SQO -> Router -> 17-Stage Orchestration -> Final Aggregated Result.
        """
        sqo, plan = self.process_query(
            raw_query=raw_query,
            image_inputs=image_inputs,
            project_context=project_context,
            session_id=session_id
        )

        orch_res = self.orchestrate_plan(
            plan=plan,
            sqo=sqo,
            image_inputs=image_inputs,
            resolved_query=sqo.effective_resolved_query
        )

        return sqo, plan, orch_res

    def process_and_aggregate(
        self,
        raw_query: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[StructuredQueryObject, ExecutionPlan, OrchestrationResult, AggregationResult]:
        """
        Complete End-to-End Pipeline with Aggregation:
        Natural Language Query + Imagery -> SQO -> Router -> Orchestrator -> Aggregator -> Final Evidence-Grounded Answer.
        """
        sqo, plan, orch_res = self.process_and_orchestrate(
            raw_query=raw_query,
            image_inputs=image_inputs,
            project_context=project_context,
            session_id=session_id
        )

        agg_res = self.aggregator.aggregate(
            raw_query=raw_query,
            input_data=orch_res,
            request_id=orch_res.request_id,
            image_inputs=image_inputs
        )

        return sqo, plan, orch_res, agg_res


    def get_registered_agents(
        self,
        capability: Optional[AgentCapability] = None,
        modality: Optional[InputModality] = None,
        status: Optional[AgentStatus] = AgentStatus.ACTIVE
    ) -> List[AgentResponse]:
        """Discovers specialist agents registered in the Agent Registry."""
        return self.registry_client.discover_agents(
            capability=capability,
            modality=modality,
            status=status
        )

    def register_custom_agent(self, request: AgentRegisterRequest) -> AgentResponse:
        """Registers a new specialist agent and invalidates the router cache."""
        agent = self.registry_client.register(request)
        self.router.cache.invalidate()
        return agent

    def register_change_model(self, custom_model_instance: Any):
        """
        Registers custom bi-temporal change model from Person 4 (Change Analysis Lead).
        Enables dynamic runtime model execution for bi-temporal change queries.
        """
        from execution.models.specialist_models import register_change_model as _reg_change
        _reg_change(custom_model_instance)

    def register_vlm_model(self, custom_model_instance: Any):
        """
        Registers custom RS-VLM model from Person 2 (VLM / BigEarthNet Lead).
        Enables dynamic runtime model execution for BigEarthNet-adapted visual question answering.
        """
        from execution.models.specialist_models import register_vlm_model as _reg_vlm
        _reg_vlm(custom_model_instance)

    def register_crossmodal_model(self, custom_model_instance: Any):
        """
        Registers custom Optical-SAR cross-modal model from Person 5 (Cross-Modal Fusion Lead).
        Enables dynamic runtime model execution for Optical-SAR cross-attention fusion queries.
        """
        from execution.models.specialist_models import register_crossmodal_model as _reg_xm
        _reg_xm(custom_model_instance)




    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieves conversation history for a given session ID."""
        return self.query_pipeline.get_session(session_id)

    def close(self):
        """Closes the active database connection."""
        if hasattr(self, 'db') and self.db:
            self.db.close()


def create_satquery_engine(**kwargs) -> SatQueryEngine:
    """Factory function to instantiate SatQueryEngine."""
    return SatQueryEngine(**kwargs)
