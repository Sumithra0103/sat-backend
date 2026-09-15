"""
SatQuery AI - Traced Engine Integration
=======================================
High-level wrapper TracedSatQueryEngine that seamlessly connects SatQueryEngine
with TraceService, automating all 35 steps of trace recording, telemetry collection,
retry/fallback tracking, and persistence.
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union

from satquery_engine import SatQueryEngine
from trace.service import TraceService
from trace.models import TraceData, TraceStatus, SpanStatus
from trace.context import TraceContext


class TracedSatQueryEngine:
    """
    Wrapper around SatQueryEngine that automatically attaches TraceService lifecycle
    to every process_query, execute_plan, and process_and_execute call.
    """

    def __init__(
        self,
        engine: Optional[SatQueryEngine] = None,
        trace_service: Optional[TraceService] = None
    ):
        """Initializes the traced engine wrapper."""
        self.engine = engine or SatQueryEngine()
        self.trace_service = trace_service or TraceService()

    def process_and_execute(
        self,
        raw_query: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[Any, Any, List[Any], TraceData]:
        """
        Executes complete SatQuery end-to-end workflow with comprehensive trace recording.
        Steps 4 through 28 of the Trace specification are performed automatically.
        """
        req_id = request_id or f"Q_{uuid.uuid4().hex[:8]}"

        # Step 7 & 8: Start trace and record request
        trace = self.trace_service.start_trace(
            request_id=req_id,
            query=raw_query,
            image_inputs=image_inputs,
            session_id=session_id
        )

        try:
            # Step 1: Query Understanding
            sqo = self.engine.query_pipeline.process_query(
                raw_query=raw_query,
                image_inputs=image_inputs or [],
                project_context_dict=project_context,
                session_id=session_id
            )
            # Step 8: Record Query Understanding outputs
            self.trace_service.record_query_understanding(sqo)

            # Step 2: Routing
            plan = self.engine.router.route(sqo)
            # Step 9 & 10: Record Router decision and selected agents
            self.trace_service.record_router_decision(plan)

            # Step 3: Execution
            results = []
            if plan.routing_status.value in ["success", "degraded"]:
                results = self.execute_plan(
                    plan=plan,
                    image_inputs=image_inputs,
                    resolved_query=sqo.effective_resolved_query,
                    trace_id=trace.trace_id
                )

            # Step 25-28: Finalize & Persist Trace
            final_trace = self.trace_service.finalize_trace(trace_id=trace.trace_id)
            return sqo, plan, results, final_trace

        except Exception as e:
            # Record global engine failure
            span = self.trace_service.create_span(
                name="engine_exception",
                component="engine",
                trace_id=trace.trace_id
            )
            self.trace_service.record_error(
                span_id=span.span_id,
                error_message=str(e),
                trace_id=trace.trace_id
            )
            self.trace_service.close_span(span.span_id, trace_id=trace.trace_id)
            final_trace = self.trace_service.finalize_trace(
                status_override=TraceStatus.FAILED,
                trace_id=trace.trace_id
            )
            raise e

    def execute_plan(
        self,
        plan: Any,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        resolved_query: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> List[Any]:
        """
        Executes an ExecutionPlan with per-agent span tracing, retry tracking,
        and fallback recording.
        """
        results = self.engine.executor.execute_plan(
            plan=plan,
            image_inputs=image_inputs,
            resolved_query=resolved_query
        )

        for res in results:
            agent_id = getattr(res, "agent", "unknown_agent")
            span = self.trace_service.create_span(
                name=f"Execution step ({agent_id})",
                component="agent",
                agent_id=agent_id,
                input_references={"image_inputs": image_inputs, "resolved_query": resolved_query},
                trace_id=trace_id
            )
            self.trace_service.record_execution_result(
                span_id=span.span_id,
                result=res,
                trace_id=trace_id
            )

        return results
