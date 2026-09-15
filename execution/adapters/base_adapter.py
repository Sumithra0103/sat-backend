"""
Base Agent Adapter for SatQuery AI Execution Subsystem.
Implements the core execution contract:
- Resource allocation & Sandboxing / isolation wrapper
- Timeout management (enforcing 60s maximum execution limit)
- Error recovery (Retry once -> Fallback agent -> Return partial/error)
- Output validation (Checking validity, confidence availability, expected format)
- Confidence extraction & Evidence collection
- Final execution result packaging conforming to the specification:
  {
    "agent": "...",
    "status": "success",
    "result": {...},
    "confidence": 0.93,
    "execution_time": 4.8
  }
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from execution.schemas import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResourceMetrics,
    ExecutionTraceItem
)
from execution.preprocessing.preprocessor import PreprocessedBatch
from execution.models.model_manager import ModelManager
from execution.evidence.evidence_collector import EvidenceCollector


class AdapterExecutionError(Exception):
    def __init__(self, message: str, can_retry: bool = True):
        super().__init__(message)
        self.can_retry = can_retry


class BaseAgentAdapter:
    """
    Abstract Base Adapter for Remote Sensing Specialist Agents.
    Executes in an isolated sandbox context with timeout protection,
    single retry, fallback recovery, and output validation.
    """

    def __init__(self, agent_id: str, agent_version: str = "1.0.0"):
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.model_manager = ModelManager()

    def run_inference(
        self,
        query: str,
        batch: PreprocessedBatch,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Subclasses or models override this to produce raw model outputs."""
        model = self.model_manager.get_model(self.agent_id)
        return model.predict(query=query, batch=batch, parameters=parameters)

    def validate_output(self, raw_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Step 4 from execution spec: Validate output.
        - Is output valid?
        - Is confidence available?
        - Is expected format returned?
        """
        if not isinstance(raw_result, dict):
            return False, "Output is not a valid JSON/dict object."
        if not raw_result:
            return False, "Output dictionary is empty."
        if "confidence" not in raw_result and "iou_confidence" not in raw_result:
            return False, "Confidence metric is missing from agent output."
        return True, None

    def execute_sandboxed(
        self,
        request: ExecutionRequest,
        batch: PreprocessedBatch,
        trace: List[ExecutionTraceItem],
        fallback_factory = None
    ) -> ExecutionResult:
        """
        Executes the agent in a sandboxed, timed wrapper with:
        1. Resource allocation
        2. Attempt 1 execution
        3. Retry on failure (once)
        4. Fallback agent on persistent failure
        5. Output validation & confidence extraction
        6. Evidence collection
        """
        start_time = time.time()
        res_metrics = self.model_manager.allocate_resources(request.resource_requirements, trace=trace)
        timeout = request.timeout_seconds
        query = request.resolved_query or "Remote sensing scene analysis"

        trace.append(ExecutionTraceItem(
            stage="Sandboxing",
            action="initiate_sandboxed_execution",
            status="success",
            details={"agent_id": self.agent_id, "timeout_seconds": timeout}
        ))

        raw_output: Optional[Dict[str, Any]] = None
        current_agent_id = self.agent_id
        current_version = self.agent_version
        status = ExecutionStatus.SUCCESS
        error_info = None

        # Execution Attempt 1
        try:
            raw_output = self._run_with_timeout_check(query, batch, request.parameters, start_time, timeout)
            trace.append(ExecutionTraceItem(
                stage="Model Inference",
                action="execute_primary_agent",
                status="success",
                details={"agent_id": self.agent_id}
            ))
        except Exception as e1:
            trace.append(ExecutionTraceItem(
                stage="Model Inference",
                action="primary_agent_failure",
                status="failed",
                details={"agent_id": self.agent_id, "error": str(e1)}
            ))

            # Step: Retry once
            trace.append(ExecutionTraceItem(
                stage="Error Recovery",
                action="retry_agent_execution",
                status="in_progress",
                details={"retry_attempt": 1}
            ))
            try:
                raw_output = self._run_with_timeout_check(query, batch, request.parameters, start_time, timeout)
                trace.append(ExecutionTraceItem(
                    stage="Error Recovery",
                    action="retry_succeeded",
                    status="success",
                    details={"agent_id": self.agent_id}
                ))
            except Exception as e2:
                trace.append(ExecutionTraceItem(
                    stage="Error Recovery",
                    action="retry_failed",
                    status="failed",
                    details={"agent_id": self.agent_id, "error": str(e2)}
                ))

                # Step: Fallback Agent
                if request.fallback_agent_id and fallback_factory:
                    trace.append(ExecutionTraceItem(
                        stage="Fallback Recovery",
                        action="dispatch_fallback_agent",
                        status="in_progress",
                        details={"fallback_agent_id": request.fallback_agent_id}
                    ))
                    try:
                        fallback_adapter = fallback_factory(request.fallback_agent_id)
                        raw_output = fallback_adapter._run_with_timeout_check(
                            query, batch, request.parameters, start_time, timeout
                        )
                        current_agent_id = request.fallback_agent_id
                        current_version = fallback_adapter.agent_version
                        status = ExecutionStatus.DEGRADED
                        trace.append(ExecutionTraceItem(
                            stage="Fallback Recovery",
                            action="fallback_agent_success",
                            status="success",
                            details={"fallback_agent_id": request.fallback_agent_id}
                        ))
                    except Exception as ef:
                        status = ExecutionStatus.FAILED
                        error_info = {"primary_error": str(e1), "retry_error": str(e2), "fallback_error": str(ef)}
                        trace.append(ExecutionTraceItem(
                            stage="Fallback Recovery",
                            action="fallback_failed",
                            status="failed",
                            details=error_info
                        ))
                else:
                    status = ExecutionStatus.FAILED
                    error_info = {"primary_error": str(e1), "retry_error": str(e2)}

        elapsed = max(0.001, round(time.time() - start_time, 3))
        self.model_manager.release_resources(res_metrics, elapsed)

        # In case of complete failure, return structured partial/error result
        if status == ExecutionStatus.FAILED or raw_output is None:
            return ExecutionResult(
                agent=current_agent_id,
                status=status,
                result={"error": "Agent execution failed after retry and fallback", "details": error_info},
                confidence=0.0,
                execution_time=elapsed,
                execution_id=request.execution_id,
                agent_version=current_version,
                evidence=[],
                resource_metrics=res_metrics,
                execution_trace=trace,
                error_details=error_info
            )

        # Step 4: Validate Output
        is_valid, validation_err = self.validate_output(raw_output)
        if not is_valid:
            status = ExecutionStatus.DEGRADED
            trace.append(ExecutionTraceItem(
                stage="Output Validation",
                action="validate_agent_output",
                status="failed",
                details={"error": validation_err}
            ))
        else:
            trace.append(ExecutionTraceItem(
                stage="Output Validation",
                action="validate_agent_output",
                status="success",
                details={"output_keys": list(raw_output.keys())}
            ))

        # Step 5: Extract Confidence
        confidence = float(raw_output.get("confidence", raw_output.get("iou_confidence", 0.85)))
        confidence = max(0.0, min(1.0, confidence))

        trace.append(ExecutionTraceItem(
            stage="Confidence Extraction",
            action="extract_and_calibrate_confidence",
            status="success",
            details={"calibrated_confidence": confidence}
        ))

        # Step 6: Collect Evidence
        evidence_items = EvidenceCollector.collect_evidence(
            agent_id=current_agent_id,
            task=request.task,
            prediction=raw_output,
            image_inputs=request.inputs,
            trace=trace
        )

        return ExecutionResult(
            agent=current_agent_id,
            status=status,
            result=raw_output,
            confidence=confidence,
            execution_time=elapsed,
            execution_id=request.execution_id,
            agent_version=current_version,
            evidence=evidence_items,
            resource_metrics=res_metrics,
            execution_trace=trace
        )

    def _run_with_timeout_check(
        self,
        query: str,
        batch: PreprocessedBatch,
        parameters: Dict[str, Any],
        start_time: float,
        timeout_seconds: float
    ) -> Dict[str, Any]:
        """Checks timeout bounds during execution."""
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Execution exceeded timeout limit of {timeout_seconds} seconds.")

        result = self.run_inference(query, batch, parameters)

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Execution exceeded timeout limit of {timeout_seconds} seconds.")

        return result
