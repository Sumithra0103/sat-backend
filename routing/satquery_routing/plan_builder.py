"""
Execution Plan Builder and Fallback Handler for SatQuery AI Router.
Steps 10 & 11 of Router workflow:
Constructs deterministic ExecutionPlan with:
1. Preprocessing steps (co-registration check, band normalization)
2. Specialist inference step with permitted task parameters
3. Evidence grounding / post-processing step
4. Fallback route configuration for automatic recovery
5. Auditable execution summary for evaluation compliance
"""

import uuid
from typing import List, Dict, Any, Optional
from schemas.agent_schema import AgentResponse
from satquery_routing.models import (
    ExecutionPlan,
    ExecutionStep,
    ExecutionStepType,
    FallbackRoute,
    CandidateScore,
    RoutingRequirements,
    AuditableExecutionSummary,
    RoutingStatus
)


class PlanBuilder:
    """
    Builds structured execution plans for the SatQuery AI Orchestrator.
    """

    @classmethod
    def configure_parameters(
        cls,
        agent: AgentResponse,
        req: RoutingRequirements
    ) -> Dict[str, Any]:
        """
        Extracts and binds permitted parameters based on agent's parameters_schema and query context.
        """
        configured = {}
        schema = agent.parameters_schema or {}

        # Default values from schema
        for param, default_val in schema.items():
            configured[param] = default_val

        # Context-dependent parameter overrides
        if req.target_classes and "output_masks" in schema:
            configured["output_masks"] = req.target_classes
        if req.target_classes and "text_threshold" in schema:
            configured["text_threshold"] = 0.30

        return configured

    @classmethod
    def build_plan(
        cls,
        req: RoutingRequirements,
        ranked_candidates: List[CandidateScore],
        agent_map: Dict[str, AgentResponse],
        image_inputs: List[Dict[str, Any]]
    ) -> ExecutionPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        if not ranked_candidates:
            # No candidates available
            return ExecutionPlan(
                plan_id=plan_id,
                query_id=req.query_id,
                routing_status=RoutingStatus.FAILED,
                requirements=req,
                candidate_evaluations=ranked_candidates,
                rejection_errors=["No registered specialist agents matched the requirements."]
            )

        primary_score = ranked_candidates[0]
        primary_agent = agent_map[primary_score.agent_id]
        primary_params = cls.configure_parameters(primary_agent, req)

        steps: List[ExecutionStep] = []

        # Step 1: Preprocessing
        prep_step_id = f"{plan_id}_prep"
        prep_params = {
            "image_count": req.image_count,
            "temporal_structure": req.temporal_structure,
            "verify_geospatial_bounds": True,
            "normalize_bands": True
        }
        if req.temporal_structure == "COREG_CROSS_MODAL_PAIR":
            prep_params["require_co_registration"] = True
            prep_params["resample_to_common_grid"] = True

        steps.append(
            ExecutionStep(
                step_id=prep_step_id,
                step_type=ExecutionStepType.PREPROCESSING,
                agent_id="system_preprocessor",
                agent_name="Geospatial Input Preprocessor & Validator",
                endpoint_url="internal://preprocessor",
                inputs=image_inputs,
                parameters=prep_params,
                timeout_seconds=10.0,
                depends_on=[]
            )
        )

        # Step 2: Specialist Inference
        infer_step_id = f"{plan_id}_infer"
        steps.append(
            ExecutionStep(
                step_id=infer_step_id,
                step_type=ExecutionStepType.SPECIALIST_INFERENCE,
                agent_id=primary_agent.agent_id,
                agent_name=primary_agent.name,
                endpoint_url=primary_agent.endpoint_url,
                inputs=[{"depends_on_step": prep_step_id}],
                parameters=primary_params,
                timeout_seconds=30.0,
                depends_on=[prep_step_id]
            )
        )

        # Step 3: Evidence Grounding & Postprocessing
        post_step_id = f"{plan_id}_post"
        steps.append(
            ExecutionStep(
                step_id=post_step_id,
                step_type=ExecutionStepType.EVIDENCE_GROUNDING,
                agent_id="system_grounder",
                agent_name="Evidence Grounding & Spatial Summarizer",
                endpoint_url="internal://evidence_grounder",
                inputs=[{"depends_on_step": infer_step_id}],
                parameters={
                    "target_classes": req.target_classes,
                    "generate_visual_evidence": True,
                    "calculate_confidence": True
                },
                timeout_seconds=15.0,
                depends_on=[infer_step_id]
            )
        )

        # Fallback routes
        fallback_routes: List[FallbackRoute] = []
        fallback_assigned = None
        if len(ranked_candidates) > 1:
            fallback_score = ranked_candidates[1]
            fallback_agent = agent_map[fallback_score.agent_id]
            fallback_params = cls.configure_parameters(fallback_agent, req)
            fallback_routes.append(
                FallbackRoute(
                    primary_agent_id=primary_agent.agent_id,
                    fallback_agent_id=fallback_agent.agent_id,
                    trigger_condition="on_failure_or_timeout_or_5xx",
                    fallback_endpoint_url=fallback_agent.endpoint_url,
                    fallback_parameters=fallback_params
                )
            )
            fallback_assigned = f"{fallback_agent.name} ({fallback_agent.agent_id})"

        # Auditable Execution Summary
        auditable_summary = AuditableExecutionSummary(
            query_id=req.query_id,
            selected_task=req.primary_capability,
            primary_agent_id=primary_agent.agent_id,
            primary_agent_name=primary_agent.name,
            models_and_tools=[s.agent_name for s in steps if s.agent_name],
            applied_parameters=primary_params,
            benchmark_dataset=req.benchmark_reference,
            input_configuration={
                "image_count": req.image_count,
                "modality": req.required_modality,
                "temporal_structure": req.temporal_structure,
                "formats": req.supported_formats
            },
            routing_confidence=primary_score.composite_score,
            fallback_assigned=fallback_assigned
        )

        return ExecutionPlan(
            plan_id=plan_id,
            query_id=req.query_id,
            routing_status=RoutingStatus.SUCCESS,
            requirements=req,
            candidate_evaluations=ranked_candidates,
            selected_agent_id=primary_agent.agent_id,
            steps=steps,
            fallback_routes=fallback_routes,
            auditable_summary=auditable_summary,
            rejection_errors=[]
        )
