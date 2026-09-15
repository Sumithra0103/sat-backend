"""
Master Router Engine for SatQuery AI.
Unifies all 12-13 routing steps into an auditable, high-performance controller.
Flow:
Query Understanding (SQO)
        ↓
1. Receive Query Plan
        ↓
2. Validate Query Plan (Ambiguity / Incompatibility checks)
        ↓
3. Extract Routing Requirements
        ↓
4. Determine Required Capabilities & Modality
        ↓
5. Query Agent Registry (via Local TTL Cache)
        ↓
6. Get Candidate Agents
        ↓
7. Filter Candidates (Capabilities, Health, Modality, Formats, Latency)
        ↓
8. Rank / Score Candidates (Remote Sensing Adaptation, Benchmarks, Composite)
        ↓
9. Select Best Agent(s) (Primary + Fallbacks)
        ↓
10. Create Execution Plan (Preprocessing, Specialist, Evidence Grounding)
        ↓
11. Add Fallback / Retry Information
        ↓
12. Return Execution Plan → Orchestrator
"""

import uuid
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from schemas.agent_schema import (
    AgentCapability,
    InputModality,
    AgentStatus,
    AgentResponse
)
from client.registry_client import AgentRegistryClient
from satquery_routing.models import (
    ExecutionPlan,
    RoutingStatus,
    RouterExecutionTraceItem,
    RoutingRequirements
)
from satquery_routing.cache import LocalRegistryCache
from satquery_routing.requirements import RequirementExtractor
from satquery_routing.filter import CandidateFilter
from satquery_routing.scorer import AgentScorer
from satquery_routing.plan_builder import PlanBuilder


class SatQueryRouter:
    """
    Main Agentic Router for SatQuery AI.
    Routes structured queries to specialized remote sensing models.
    """

    def __init__(
        self,
        registry_client: Optional[AgentRegistryClient] = None,
        db_session: Optional[Session] = None,
        cache_ttl_seconds: float = 30.0
    ):
        if registry_client:
            self.client = registry_client
        elif db_session:
            self.client = AgentRegistryClient(db_session=db_session)
        else:
            raise ValueError("SatQueryRouter requires either a registry_client or a db_session.")

        self.cache = LocalRegistryCache(self.client, default_ttl_seconds=cache_ttl_seconds)

    def route(self, sqo: Any) -> ExecutionPlan:
        """
        Executes the complete 12-step routing pipeline.
        Returns the finalized ExecutionPlan with auditable trace.
        """
        trace: List[RouterExecutionTraceItem] = []

        def log_step(stage: str, action: str, status: str, details: Dict[str, Any] = None):
            trace.append(
                RouterExecutionTraceItem(
                    stage=stage,
                    action=action,
                    status=status,
                    details=details or {}
                )
            )

        # -------------------------------------------------------------
        # Step 1: Receive Query Plan
        # -------------------------------------------------------------
        query_id = getattr(sqo, "query_id", f"qry_{uuid.uuid4().hex[:8]}")
        raw_query = getattr(sqo, "raw_query", "")
        log_step("STEP_1_RECEIVE", "Receive Query Plan", "SUCCESS", {
            "query_id": query_id,
            "raw_query": raw_query
        })

        # -------------------------------------------------------------
        # Step 2: Validate Query Plan & Inputs
        # -------------------------------------------------------------
        ambiguity_rep = getattr(sqo, "ambiguity_report", None)
        if ambiguity_rep and getattr(ambiguity_rep, "is_ambiguous", False):
            issues = getattr(ambiguity_rep, "issues", ["Ambiguous query or missing inputs"])
            remediations = getattr(ambiguity_rep, "suggested_remediations", [])
            log_step("STEP_2_VALIDATE", "Check Ambiguity & Compatibility", "REJECTED", {
                "issues": issues,
                "remediations": remediations
            })

            # Create an early rejection plan
            dummy_req = RoutingRequirements(
                query_id=query_id,
                resolved_query=raw_query,
                primary_capability="unknown",
                required_modality="unknown",
                image_count=len(getattr(sqo, "image_inputs", [])),
                temporal_structure="UNKNOWN"
            )
            rejection_plan = ExecutionPlan(
                plan_id=f"plan_rej_{uuid.uuid4().hex[:8]}",
                query_id=query_id,
                routing_status=RoutingStatus.AMBIGUOUS_REJECTED,
                requirements=dummy_req,
                rejection_errors=issues,
                router_trace=trace
            )
            return rejection_plan

        log_step("STEP_2_VALIDATE", "Validate Query Plan", "SUCCESS", {
            "validation_status": "PASSED"
        })

        # -------------------------------------------------------------
        # Step 3 & 4: Extract Routing Requirements & Determine Capabilities
        # -------------------------------------------------------------
        req = RequirementExtractor.extract(sqo)
        log_step("STEP_3_4_REQUIREMENTS", "Extract Requirements & Capabilities", "SUCCESS", {
            "primary_capability": req.primary_capability,
            "required_modality": req.required_modality,
            "temporal_structure": req.temporal_structure,
            "benchmark_reference": req.benchmark_reference,
            "target_classes": req.target_classes
        })

        # -------------------------------------------------------------
        # Step 5 & 6: Query Registry & Get Candidates (via Local Cache)
        # -------------------------------------------------------------
        primary_cap_enum = None
        try:
            primary_cap_enum = AgentCapability(req.primary_capability)
        except ValueError:
            pass

        modality_enum = None
        try:
            modality_enum = InputModality(req.required_modality)
        except ValueError:
            pass

        # Query all active candidates matching modality or capability
        raw_candidates = self.cache.get_candidates(
            capability=primary_cap_enum,
            modality=modality_enum,
            status=AgentStatus.ACTIVE
        )

        # If strict search returned none, query broader active candidates for fallbacks
        if not raw_candidates:
            raw_candidates = self.cache.get_candidates(status=AgentStatus.ACTIVE)

        candidate_ids = [c.agent_id for c in raw_candidates]
        log_step("STEP_5_6_DISCOVERY", "Query Registry & Retrieve Candidates", "SUCCESS", {
            "candidate_count": len(raw_candidates),
            "candidate_ids": candidate_ids
        })

        # -------------------------------------------------------------
        # Step 7: Filter Candidates
        # -------------------------------------------------------------
        qualified_agents, disqualified_scores = CandidateFilter.filter_candidates(
            raw_candidates,
            req
        )

        log_step("STEP_7_FILTER", "Filter Candidates", "SUCCESS", {
            "qualified_count": len(qualified_agents),
            "qualified_ids": [a.agent_id for a in qualified_agents],
            "disqualified_count": len(disqualified_scores),
            "disqualified_reasons": {d.agent_id: d.rejection_reason for d in disqualified_scores}
        })

        # -------------------------------------------------------------
        # Step 8: Rank & Score Candidates
        # -------------------------------------------------------------
        ranked_scores = AgentScorer.rank_candidates(qualified_agents, req)
        all_evaluations = ranked_scores + disqualified_scores

        agent_map = {a.agent_id: a for a in qualified_agents}

        log_step("STEP_8_SCORE", "Score and Rank Candidates", "SUCCESS", {
            "ranked_scores": [
                {
                    "agent_id": s.agent_id,
                    "composite": s.composite_score,
                    "capability": s.capability_score,
                    "adaptation": s.adaptation_score,
                    "modality": s.modality_score
                } for s in ranked_scores
            ]
        })

        # -------------------------------------------------------------
        # Step 9, 10, 11: Select Agent, Build Execution Plan & Fallbacks
        # -------------------------------------------------------------
        image_inputs = getattr(sqo, "image_inputs", [])
        plan = PlanBuilder.build_plan(
            req=req,
            ranked_candidates=ranked_scores,
            agent_map=agent_map,
            image_inputs=image_inputs
        )
        plan.candidate_evaluations = all_evaluations

        # -------------------------------------------------------------
        # Step 12: Return Execution Plan
        # -------------------------------------------------------------
        log_step("STEP_12_PLAN_GENERATED", "Generate Execution Plan", "SUCCESS", {
            "plan_id": plan.plan_id,
            "selected_agent": plan.selected_agent_id,
            "steps_count": len(plan.steps),
            "fallback_count": len(plan.fallback_routes)
        })

        plan.router_trace = trace
        return plan
