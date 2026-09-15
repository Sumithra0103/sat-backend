"""
SatQuery AI - Trace Service
===========================
Implements TraceService managing full trace lifecycle, span tracking, routing decisons,
execution telemetries, retry/fallback handling, total duration calculation, and persistence.
Satisfies Steps 6 through 28 of the Trace Specification.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone

from trace.models import (
    TraceData,
    TraceSpan,
    TraceEvent,
    TraceStatus,
    SpanStatus
)
from trace.context import (
    generate_trace_id,
    generate_span_id,
    TraceContext
)


class TraceService:
    """
    Step 6: Core service managing trace recording, span execution telemetry,
    metrics calculation, persistence, and trace retrieval.
    """

    def __init__(self, persistence_dir: Optional[str] = None):
        """Initializes the TraceService with a default persistence directory."""
        if persistence_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            persistence_dir = str(base_dir / "evidence_artifacts" / "traces")
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self._memory_store: Dict[str, TraceData] = {}

    # Step 7 & 8: Implement start_trace() and record request/query
    def start_trace(
        self,
        request_id: str,
        query: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None
    ) -> TraceData:
        """
        Step 7: Initializes a new trace, creates TraceContext, records initial request details.
        Step 8: Records request query and image inputs.
        """
        trace_id = generate_trace_id()
        now = time.time()
        
        trace = TraceData(
            trace_id=trace_id,
            request_id=request_id,
            created_at=now,
            updated_at=now,
            iso_created_at=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            query=query,
            image_inputs=image_inputs or [],
            status=TraceStatus.RUNNING,
            metrics={"session_id": session_id} if session_id else {}
        )

        # Record initial event
        event = TraceEvent(
            event_id=f"evt_{generate_span_id()}",
            timestamp=now,
            event_type="trace_started",
            component="trace_service",
            status="success",
            message=f"Trace {trace_id} started for Request {request_id}",
            payload={"query": query, "image_count": len(image_inputs or [])}
        )
        trace.events.append(event)

        # Save into context and memory
        TraceContext.set_current_trace(trace)
        self._memory_store[trace_id] = trace
        return trace

    # Step 8: Record query understanding outputs
    def record_query_understanding(
        self,
        sqo: Any,
        trace_id: Optional[str] = None
    ) -> None:
        """Records Query Understanding pipeline results in the trace."""
        trace = self._get_active_trace(trace_id)
        if not trace:
            return

        now = time.time()
        sqo_dict = {}
        if hasattr(sqo, "model_dump"):
            sqo_dict = sqo.model_dump()
        elif hasattr(sqo, "dict"):
            sqo_dict = sqo.dict()
        elif isinstance(sqo, dict):
            sqo_dict = sqo

        trace.query_understanding = sqo_dict
        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=now,
                event_type="query_understanding_completed",
                component="query_understanding",
                status="success",
                message="Query understanding completed successfully",
                payload={"tasks": sqo_dict.get("tasks", []), "modalities": sqo_dict.get("modalities", [])}
            )
        )
        trace.updated_at = now

    # Step 9 & 10: Record Router decision and selected agents
    def record_router_decision(
        self,
        plan: Any,
        trace_id: Optional[str] = None
    ) -> None:
        """
        Step 9: Records Router execution plan decisions.
        Step 10: Records selected agents list.
        """
        trace = self._get_active_trace(trace_id)
        if not trace:
            return

        now = time.time()
        plan_dict = {}
        if hasattr(plan, "model_dump"):
            plan_dict = plan.model_dump()
        elif hasattr(plan, "dict"):
            plan_dict = plan.dict()
        elif isinstance(plan, dict):
            plan_dict = plan

        trace.router_decision = plan_dict

        # Extract selected agents list (Step 10)
        selected_agents = []
        steps = plan_dict.get("execution_steps", []) or plan_dict.get("steps", [])
        for step in steps:
            if isinstance(step, dict):
                agent_info = {
                    "step_id": step.get("step_id"),
                    "agent_id": step.get("agent_id"),
                    "agent_name": step.get("agent_name"),
                    "agent_version": step.get("agent_version", "1.0.0"),
                    "capability": step.get("capability") or step.get("step_type"),
                    "parameters": step.get("parameters", {})
                }
                selected_agents.append(agent_info)

        trace.selected_agents = selected_agents

        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=now,
                event_type="router_decision_completed",
                component="router",
                status="success",
                message=f"Router selected {len(selected_agents)} agent(s)",
                payload={"selected_agent_ids": [a.get("agent_id") for a in selected_agents]}
            )
        )
        trace.updated_at = now

    def record_aggregation(
        self,
        aggregation_result: Any,
        trace_id: Optional[str] = None
    ) -> None:
        """Records Aggregation Layer results and evidence bundle in the trace."""
        trace = self._get_active_trace(trace_id)
        if not trace:
            return

        now = time.time()
        agg_dict = {}
        if hasattr(aggregation_result, "model_dump"):
            agg_dict = aggregation_result.model_dump()
        elif hasattr(aggregation_result, "dict"):
            agg_dict = aggregation_result.dict()
        elif isinstance(aggregation_result, dict):
            agg_dict = aggregation_result

        trace.metrics["aggregation_confidence"] = agg_dict.get("aggregated_confidence", 0.0)
        trace.metrics["visual_evidence_count"] = len(agg_dict.get("visual_evidence_urls", []))
        
        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=now,
                event_type="aggregation_completed",
                component="aggregation",
                status=agg_dict.get("status", "success"),
                message=f"Aggregation completed with confidence {agg_dict.get('aggregated_confidence')}",
                payload={
                    "request_id": agg_dict.get("request_id"),
                    "confidence": agg_dict.get("aggregated_confidence"),
                    "final_answer_length": len(agg_dict.get("final_answer", "")),
                    "visual_artifacts": agg_dict.get("visual_evidence_urls", [])
                }
            )
        )
        trace.updated_at = now


    # Step 11, 13, 14, 15, 16: Create execution spans, record agent/version, inputs, parameters, start time
    def create_span(
        self,
        name: str,
        component: str = "agent",
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_version: str = "1.0.0",
        input_references: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> TraceSpan:
        """
        Step 11: Creates an execution span.
        Step 13: Records agent and version.
        Step 14: Records input references.
        Step 15: Records parameters.
        Step 16: Records start time.
        """
        trace = self._get_active_trace(trace_id)
        span_id = generate_span_id()
        parent_id = parent_span_id or TraceContext.current_parent_span_id()

        start_time = time.time()
        span = TraceSpan(
            span_id=span_id,
            parent_span_id=parent_id,
            name=name,
            component=component,
            agent_id=agent_id or name,
            agent_name=agent_name or name,
            agent_version=agent_version,
            start_time=start_time,
            status=SpanStatus.RUNNING,
            input_references=input_references or {},
            parameters=parameters or {}
        )

        if trace:
            trace.spans.append(span)
            trace.updated_at = start_time
            TraceContext.push_span_id(span_id)

        return span

    # Step 12, 17, 18, 19, 20: Consume ExecutionResult, calculate duration, record status, confidence, output refs
    def record_execution_result(
        self,
        span_id: str,
        result: Any,
        trace_id: Optional[str] = None
    ) -> Optional[TraceSpan]:
        """
        Step 12: Consumes ExecutionResult object.
        Step 17: Calculates duration.
        Step 18: Records status.
        Step 19: Records confidence.
        Step 20: Records output & evidence references.
        """
        trace = self._get_active_trace(trace_id)
        if not trace:
            return None

        span = self._find_span(trace, span_id)
        if not span:
            return None

        end_time = time.time()
        res_dict = {}
        if hasattr(result, "model_dump"):
            res_dict = result.model_dump()
        elif hasattr(result, "dict"):
            res_dict = result.dict()
        elif isinstance(result, dict):
            res_dict = result

        # Record end time & calculate duration (Step 16 & 17)
        span.end_time = end_time
        span.duration_ms = round((end_time - span.start_time) * 1000.0, 2)

        raw_status = res_dict.get("status", "success")
        status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        if status_str.lower() in ["success", "executionstatus.success"]:
            span.status = SpanStatus.SUCCESS
        elif status_str.lower() in ["degraded", "executionstatus.degraded"]:
            span.status = SpanStatus.SUCCESS
        else:
            span.status = SpanStatus.FAILURE

        # Record confidence (Step 19)
        span.confidence = res_dict.get("confidence_score") or res_dict.get("confidence", 1.0)

        # Record output references & evidence (Step 20)
        span.output_references = {
            "raw_output": res_dict.get("raw_output"),
            "execution_id": res_dict.get("execution_id"),
            "agent_id": res_dict.get("agent_id")
        }
        span.evidence_references = res_dict.get("evidence_list", [])

        # Retries & Fallbacks if captured in result
        if res_dict.get("retries_attempted"):
            span.retries = res_dict.get("retries_attempted")
        if res_dict.get("fallback_triggered"):
            span.fallbacks = 1
            if span.status == SpanStatus.SUCCESS:
                span.status = SpanStatus.FALLBACK

        # Record error if present (Step 21)
        if res_dict.get("error_message"):
            span.error = {
                "message": res_dict.get("error_message"),
                "status": str(raw_status)
            }

        # Step 24: Close span
        self.close_span(span_id, trace_id=trace.trace_id)
        return span

    # Step 21: Record errors
    def record_error(
        self,
        span_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> None:
        """Step 21: Records error in span and logs error trace event."""
        trace = self._get_active_trace(trace_id)
        if not trace:
            return

        span = self._find_span(trace, span_id)
        if span:
            span.status = SpanStatus.FAILURE
            span.error = {
                "message": error_message,
                "details": error_details or {},
                "timestamp": time.time()
            }

        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=time.time(),
                event_type="span_error",
                component=span.component if span else "agent",
                status="failure",
                message=f"Error in span {span_id}: {error_message}",
                payload={"span_id": span_id, "error": error_message, "details": error_details or {}}
            )
        )
        trace.updated_at = time.time()

    # Step 22: Record retries
    def record_retry(
        self,
        span_id: str,
        retry_count: int,
        error_message: str,
        trace_id: Optional[str] = None
    ) -> None:
        """Step 22: Records retry attempt for an execution span."""
        trace = self._get_active_trace(trace_id)
        if not trace:
            return

        span = self._find_span(trace, span_id)
        now = time.time()
        if span:
            span.retries = max(span.retries, retry_count)
            span.retry_history.append({
                "retry_number": retry_count,
                "timestamp": now,
                "error": error_message
            })
            span.status = SpanStatus.RETRYING

        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=now,
                event_type="span_retry",
                component=span.component if span else "agent",
                status="warning",
                message=f"Retry #{retry_count} for span {span_id}: {error_message}",
                payload={"span_id": span_id, "retry_count": retry_count, "error": error_message}
            )
        )
        trace.updated_at = now

    # Step 23: Record fallbacks
    def record_fallback(
        self,
        span_id: str,
        fallback_agent_id: str,
        reason: str,
        trace_id: Optional[str] = None
    ) -> None:
        """Step 23: Records fallback execution route activation."""
        trace = self._get_active_trace(trace_id)
        if not trace:
            return

        span = self._find_span(trace, span_id)
        now = time.time()
        if span:
            span.fallbacks += 1
            span.fallback_history.append({
                "fallback_agent_id": fallback_agent_id,
                "reason": reason,
                "timestamp": now
            })
            span.status = SpanStatus.FALLBACK

        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=now,
                event_type="span_fallback",
                component=span.component if span else "agent",
                status="warning",
                message=f"Fallback triggered for span {span_id} to agent {fallback_agent_id}",
                payload={"span_id": span_id, "fallback_agent_id": fallback_agent_id, "reason": reason}
            )
        )
        trace.updated_at = now

    # Step 24: Close spans
    def close_span(
        self,
        span_id: str,
        trace_id: Optional[str] = None
    ) -> Optional[TraceSpan]:
        """Step 24: Closes an execution span and pops from context stack."""
        trace = self._get_active_trace(trace_id)
        if not trace:
            return None

        span = self._find_span(trace, span_id)
        if not span:
            return None

        if span.end_time is None:
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000.0, 2)

        TraceContext.pop_span_id()
        trace.updated_at = time.time()
        return span

    # Step 25, 26, 27: Calculate total duration, determine final status, finalize trace
    def finalize_trace(
        self,
        status_override: Optional[TraceStatus] = None,
        trace_id: Optional[str] = None
    ) -> TraceData:
        """
        Step 25: Calculates total trace duration.
        Step 26: Determines final overall status (SUCCESS, PARTIAL_SUCCESS, FAILED, DEGRADED).
        Step 27: Finalizes trace object.
        """
        trace = self._get_active_trace(trace_id)
        if not trace:
            raise ValueError("No active trace found to finalize")

        now = time.time()
        trace.updated_at = now

        # Step 25: Calculate total wall-clock duration
        trace.total_duration_ms = round((now - trace.created_at) * 1000.0, 2)

        # Step 26: Determine final status if not explicitly overridden
        if status_override:
            trace.status = status_override
        else:
            if not trace.spans:
                trace.status = TraceStatus.SUCCESS
            else:
                span_statuses = [s.status for s in trace.spans]
                if all(s in [SpanStatus.SUCCESS, SpanStatus.SKIPPED] for s in span_statuses):
                    trace.status = TraceStatus.SUCCESS
                elif any(s in [SpanStatus.SUCCESS, SpanStatus.FALLBACK] for s in span_statuses) and any(s == SpanStatus.FAILURE for s in span_statuses):
                    trace.status = TraceStatus.PARTIAL_SUCCESS
                elif any(s == SpanStatus.FALLBACK for s in span_statuses):
                    trace.status = TraceStatus.DEGRADED
                elif all(s == SpanStatus.FAILURE for s in span_statuses):
                    trace.status = TraceStatus.FAILED
                else:
                    trace.status = TraceStatus.SUCCESS

        # Step 27: Record final metrics and event
        trace.metrics["span_count"] = len(trace.spans)
        trace.metrics["successful_spans"] = sum(1 for s in trace.spans if s.status == SpanStatus.SUCCESS)
        trace.metrics["failed_spans"] = sum(1 for s in trace.spans if s.status == SpanStatus.FAILURE)
        trace.metrics["fallback_spans"] = sum(1 for s in trace.spans if s.status == SpanStatus.FALLBACK)

        trace.events.append(
            TraceEvent(
                event_id=f"evt_{generate_span_id()}",
                timestamp=now,
                event_type="trace_finalized",
                component="trace_service",
                status="success" if trace.status in [TraceStatus.SUCCESS, TraceStatus.PARTIAL_SUCCESS] else "failure",
                message=f"Trace {trace.trace_id} finalized with status {trace.status.value}",
                payload={"total_duration_ms": trace.total_duration_ms, "final_status": trace.status.value}
            )
        )

        # Step 28: Persist trace
        self.persist_trace(trace=trace)
        return trace

    # Step 28: Persist trace
    def persist_trace(
        self,
        trace: Optional[TraceData] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """Step 28: Saves the trace JSON to disk in evidence_artifacts/traces/<trace_id>.json."""
        target_trace = trace or TraceContext.get_current_trace()
        if not target_trace:
            raise ValueError("No trace object available to persist")

        out_path = Path(output_dir) if output_dir else self.persistence_dir
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{target_trace.trace_id}.json"

        # Serialize Pydantic object cleanly
        if hasattr(target_trace, "model_dump_json"):
            data_str = target_trace.model_dump_json(indent=2)
        else:
            data_str = target_trace.json(indent=2)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data_str)

        return str(file_path)

    # Step 29 support: Trace retrieval
    def get_trace(self, trace_id: str) -> Optional[TraceData]:
        """Retrieves trace by ID from memory store or disk."""
        if trace_id in self._memory_store:
            return self._memory_store[trace_id]

        file_path = self.persistence_dir / f"{trace_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
                trace = TraceData(**data_dict)
                self._memory_store[trace_id] = trace
                return trace

        return None

    def list_traces(self) -> List[Dict[str, Any]]:
        """Lists metadata of all available traces."""
        traces = []
        for file_path in self.persistence_dir.glob("trc_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    traces.append({
                        "trace_id": d.get("trace_id"),
                        "request_id": d.get("request_id"),
                        "query": d.get("query"),
                        "status": d.get("status"),
                        "total_duration_ms": d.get("total_duration_ms"),
                        "span_count": len(d.get("spans", [])),
                        "created_at": d.get("iso_created_at")
                    })
            except Exception:
                continue

        return sorted(traces, key=lambda x: x.get("created_at", ""), reverse=True)

    # Internal helpers
    def _get_active_trace(self, trace_id: Optional[str] = None) -> Optional[TraceData]:
        if trace_id:
            return self.get_trace(trace_id)
        return TraceContext.get_current_trace()

    def _find_span(self, trace: TraceData, span_id: str) -> Optional[TraceSpan]:
        for span in trace.spans:
            if span.span_id == span_id:
                return span
        return None
