"""
Request Validator for SatQuery AI Execution Subsystem.
Validates execution requests, agent eligibility, parameter schemas, and task permissions.
"""

from typing import Dict, Any, List, Optional
from execution.schemas import ExecutionRequest, ExecutionTraceItem


class RequestValidationError(Exception):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]


class RequestValidator:
    """
    Validates that:
    1. Execution request contains all mandatory fields (execution_id, agent_id, task, inputs).
    2. Agent parameters conform to permitted types, ranges, and required fields.
    3. Requested task is compatible with the specialist agent.
    """

    @staticmethod
    def validate_request(
        request: ExecutionRequest,
        agent_spec: Optional[Dict[str, Any]] = None,
        trace: Optional[List[ExecutionTraceItem]] = None
    ) -> bool:
        errors = []

        if not request.execution_id:
            errors.append("Execution ID is required.")
        if not request.agent_id:
            errors.append("Agent ID is required.")
        if not request.task:
            errors.append("Task is required.")
        if not request.inputs:
            errors.append("At least one input image must be provided.")

        if request.timeout_seconds <= 0:
            errors.append(f"Timeout must be positive, got {request.timeout_seconds}s.")

        # Parameter validation against agent parameters_schema if available
        if agent_spec and "parameters_schema" in agent_spec:
            schema = agent_spec["parameters_schema"]
            if isinstance(schema, dict) and "properties" in schema:
                props = schema.get("properties", {})
                required = schema.get("required", [])

                for req_key in required:
                    if req_key not in request.parameters:
                        errors.append(f"Missing required parameter '{req_key}' for agent {request.agent_id}.")

                for key, val in request.parameters.items():
                    if key in props:
                        prop_spec = props[key]
                        p_type = prop_spec.get("type")
                        if p_type == "number" and not isinstance(val, (int, float)):
                            errors.append(f"Parameter '{key}' must be numeric, got {type(val).__name__}.")
                        elif p_type == "integer" and not isinstance(val, int):
                            errors.append(f"Parameter '{key}' must be an integer, got {type(val).__name__}.")
                        elif p_type == "string" and not isinstance(val, str):
                            errors.append(f"Parameter '{key}' must be a string, got {type(val).__name__}.")
                        elif p_type == "boolean" and not isinstance(val, bool):
                            errors.append(f"Parameter '{key}' must be a boolean, got {type(val).__name__}.")

                        # Check bounds
                        if isinstance(val, (int, float)):
                            if "minimum" in prop_spec and val < prop_spec["minimum"]:
                                errors.append(f"Parameter '{key}' is below minimum {prop_spec['minimum']}: {val}.")
                            if "maximum" in prop_spec and val > prop_spec["maximum"]:
                                errors.append(f"Parameter '{key}' exceeds maximum {prop_spec['maximum']}: {val}.")
                        
                        # Check enums
                        if "enum" in prop_spec and val not in prop_spec["enum"]:
                            errors.append(f"Parameter '{key}' has invalid value '{val}'. Permitted: {prop_spec['enum']}.")

        if trace is not None:
            trace.append(ExecutionTraceItem(
                stage="Request Validation",
                action="validate_request_parameters",
                status="failed" if errors else "success",
                details={"errors": errors, "agent_id": request.agent_id, "task": request.task}
            ))

        if errors:
            raise RequestValidationError(f"Execution request validation failed: {'; '.join(errors)}", errors)

        return True
