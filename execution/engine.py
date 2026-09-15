"""
Execution Engine for SatQuery AI.
Central orchestrator executing the 19-stage execution pipeline:
Execution Request -> Agent Registry Lookup -> Request Validation -> Input Validation ->
Geospatial Validation -> Image Loading -> Preprocessing -> Adapter Factory ->
Model Manager & Resource Allocation -> Sandboxed Execution (with Retry & Fallback) ->
Output Validation -> Confidence Extraction -> Evidence Collection -> Execution Result.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session

from execution.schemas import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ExecutionTraceItem,
    ResourceRequirements,
    GeospatialMetadata
)
from execution.validation.request_validator import RequestValidator, RequestValidationError
from execution.validation.input_validator import InputValidator, InputValidationError
from execution.validation.geospatial_validator import GeospatialValidator, GeospatialValidationError
from execution.preprocessing.image_loader import ImageLoader, LoadedRaster
from execution.preprocessing.preprocessor import ImagePreprocessor
from execution.adapters.adapter_factory import AdapterFactory
from execution.models.model_manager import ModelManager


class ExecutionEngine:
    """
    Unified Execution Engine for SatQuery AI.
    Executes specialist remote sensing tasks dispatched by the Router.
    """

    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.model_manager = ModelManager()

    def execute_request(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Executes a single ExecutionRequest through the full 19-stage pipeline.
        """
        start_time = time.time()
        trace: List[ExecutionTraceItem] = []

        trace.append(ExecutionTraceItem(
            stage="Execution Request",
            action="receive_request",
            status="success",
            details={
                "execution_id": request.execution_id,
                "agent_id": request.agent_id,
                "task": request.task,
                "input_count": len(request.inputs)
            }
        ))

        # Stage 2: Agent Registry Lookup
        agent_spec = self._lookup_agent_in_registry(request.agent_id, trace)

        # Stage 3: Request & Parameter Validation
        try:
            RequestValidator.validate_request(request, agent_spec, trace)
        except RequestValidationError as e:
            return self._build_error_result(
                request, str(e), e.errors, "Request Validation", trace, start_time
            )

        # Stage 4: Input Validation
        is_benchmark = bool(
            "benchmark" in request.task.lower() or
            any("png" in str(img.get("format", "")).lower() for img in request.inputs)
        )
        try:
            InputValidator.validate_inputs(request.inputs, request.task, is_benchmark, trace)
        except InputValidationError as e:
            return self._build_error_result(
                request, str(e), e.errors, "Input Validation", trace, start_time
            )

        # Stage 5 & 6: Image Loading & Geospatial Metadata Extraction
        loaded_rasters: List[LoadedRaster] = []
        geospatial_meta_list: List[GeospatialMetadata] = []
        try:
            for img_dict in request.inputs:
                raster = ImageLoader.load_image(img_dict, trace)
                loaded_rasters.append(raster)
                geospatial_meta_list.append(raster.metadata)
        except Exception as e:
            return self._build_error_result(
                request, f"Image loading failed: {str(e)}", [str(e)], "Image Loading", trace, start_time
            )

        # Stage 5: Geospatial Validation
        try:
            GeospatialValidator.validate_geospatial(geospatial_meta_list, request.task, trace)
        except GeospatialValidationError as e:
            return self._build_error_result(
                request, str(e), e.errors, "Geospatial Validation", trace, start_time
            )

        # Stage 7: Preprocessing (Band selection, Normalization, Tiling, Tensor formatting)
        try:
            preprocessed_batch = ImagePreprocessor.preprocess_rasters(
                loaded_rasters,
                task=request.task,
                parameters=request.parameters,
                trace=trace
            )
        except Exception as e:
            return self._build_error_result(
                request, f"Preprocessing failed: {str(e)}", [str(e)], "Preprocessing", trace, start_time
            )

        # Stage 8: Adapter Factory
        try:
            adapter = AdapterFactory.get_adapter(request.agent_id, agent_spec)
            trace.append(ExecutionTraceItem(
                stage="Adapter Factory",
                action="instantiate_adapter",
                status="success",
                details={"adapter_class": adapter.__class__.__name__, "agent_id": request.agent_id}
            ))
        except Exception as e:
            return self._build_error_result(
                request, f"Failed to instantiate adapter: {str(e)}", [str(e)], "Adapter Factory", trace, start_time
            )

        # Fallback factory helper closure
        def fallback_factory(fb_id: str):
            fb_spec = self._lookup_agent_in_registry(fb_id)
            return AdapterFactory.get_adapter(fb_id, fb_spec)

        # Stages 9 to 14: Sandboxed Specialist Execution (with Timeout, Retry, Fallback)
        result = adapter.execute_sandboxed(
            request=request,
            batch=preprocessed_batch,
            trace=trace,
            fallback_factory=fallback_factory
        )

        trace.append(ExecutionTraceItem(
            stage="Execution Completed",
            action="return_result_to_orchestrator",
            status=result.status.value,
            details={
                "agent": result.agent,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "evidence_count": len(result.evidence)
            }
        ))

        return result

    def execute_plan(
        self,
        plan: Any,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        resolved_query: Optional[str] = None
    ) -> List[ExecutionResult]:
        """
        Executes all specialist steps of a SatQueryRouter ExecutionPlan.
        """
        results: List[ExecutionResult] = []

        if not hasattr(plan, "steps") or not plan.steps:
            return results

        # Determine fallback agent ID from plan if present
        fallback_agent_id = None
        if hasattr(plan, "fallback_routes") and plan.fallback_routes:
            fallback_agent_id = plan.fallback_routes[0].fallback_agent_id

        # Identify query string
        query_text = resolved_query
        if not query_text and hasattr(plan, "requirements"):
            query_text = plan.requirements.resolved_query

        # Extract images from plan requirements or parameter
        inputs = image_inputs or []
        if not inputs and hasattr(plan, "auditable_summary") and plan.auditable_summary:
            config = plan.auditable_summary.input_configuration
            # if images dict provided in config
            if isinstance(config, dict) and "images" in config:
                inputs = config["images"]

        # If inputs still empty, construct minimal placeholder inputs
        if not inputs and hasattr(plan, "requirements"):
            count = plan.requirements.image_count or 1
            mod = plan.requirements.required_modality or "OPTICAL"
            inputs = [{
                "image_id": f"img_{i}",
                "file_name": f"remote_sensing_input_{i}.tif",
                "format": "GeoTIFF",
                "modality": mod
            } for i in range(count)]

        for step in plan.steps:
            # Only execute specialist inference steps through agents
            step_type_val = step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)
            if "specialist" in step_type_val.lower() or "inference" in step_type_val.lower():
                step_agent_id = step.agent_id or plan.selected_agent_id
                step_params = step.parameters or {}

                task_name = (
                    plan.auditable_summary.selected_task
                    if (hasattr(plan, "auditable_summary") and plan.auditable_summary)
                    else "remote_sensing_analysis"
                )

                req = ExecutionRequest(
                    execution_id=f"exec_{step.step_id}_{uuid.uuid4().hex[:6]}",
                    agent_id=step_agent_id,
                    task=task_name,
                    inputs=inputs,
                    parameters=step_params,
                    timeout_seconds=step.timeout_seconds,
                    fallback_agent_id=fallback_agent_id,
                    resolved_query=query_text
                )

                res = self.execute_request(req)
                results.append(res)

        return results

    def _lookup_agent_in_registry(
        self,
        agent_id: str,
        trace: Optional[List[ExecutionTraceItem]] = None
    ) -> Optional[Dict[str, Any]]:
        """Queries the SQLite Agent Registry for agent specifications."""
        agent_spec = None
        if self.db_session:
            try:
                from database.models import AgentModel
                model = self.db_session.query(AgentModel).filter(AgentModel.agent_id == agent_id).first()
                if model:
                    agent_spec = {
                        "agent_id": model.agent_id,
                        "name": model.name,
                        "version": model.version,
                        "capabilities": model.capabilities,
                        "input_modalities": model.input_modalities,
                        "parameters_schema": model.parameters_schema,
                        "metadata": model.agent_metadata,
                        "status": model.status
                    }
            except Exception:
                pass

        if trace is not None:
            trace.append(ExecutionTraceItem(
                stage="Agent Registry",
                action="retrieve_agent_configuration",
                status="success" if agent_spec else "fallback_default",
                details={"agent_id": agent_id, "found": bool(agent_spec)}
            ))

        return agent_spec

    def _build_error_result(
        self,
        request: ExecutionRequest,
        message: str,
        errors: List[str],
        stage: str,
        trace: List[ExecutionTraceItem],
        start_time: float
    ) -> ExecutionResult:
        elapsed = max(0.001, round(time.time() - start_time, 3))
        return ExecutionResult(
            agent=request.agent_id,
            status=ExecutionStatus.FAILED,
            result={"error": message, "validation_errors": errors, "stage": stage},
            confidence=0.0,
            execution_time=elapsed,
            execution_id=request.execution_id,
            agent_version=request.agent_version,
            evidence=[],
            execution_trace=trace,
            error_details={"stage": stage, "errors": errors}
        )
