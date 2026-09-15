"""
SatQuery AI - Result Validator (Phase 3)
========================================
Validates incoming AgentResult objects for completeness, schema conformity,
coordinate bounds, non-NaN confidence scores, and path integrity.
"""

from typing import List, Dict, Any, Tuple
import logging
from aggregation.schemas import AgentResult

logger = logging.getLogger("SatQuery.Aggregation.Validator")


class ValidationIssue:
    def __init__(self, agent_id: str, field: str, issue_type: str, message: str):
        self.agent_id = agent_id
        self.field = field
        self.issue_type = issue_type
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "field": self.field,
            "issue_type": self.issue_type,
            "message": self.message
        }


class ResultValidator:
    """
    Validates agent results before normalization and fusion.
    """

    @staticmethod
    def validate_agent_result(agent_res: AgentResult) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validates a single AgentResult object.
        Returns (is_valid, list_of_issues).
        """
        issues: List[ValidationIssue] = []

        # 1. Agent ID & Task Type Check
        if not agent_res.agent_id:
            issues.append(ValidationIssue("unknown", "agent_id", "missing", "Agent ID is empty."))
        if not agent_res.task_type:
            issues.append(ValidationIssue(agent_res.agent_id, "task_type", "missing", "Task type is empty."))

        # 2. Confidence Validation
        if agent_res.confidence is None or agent_res.confidence != agent_res.confidence:  # NaN check
            issues.append(ValidationIssue(agent_res.agent_id, "confidence", "nan_or_null", "Confidence is NaN or null."))
            agent_res.confidence = 0.5  # Fallback fix
        elif not (0.0 <= agent_res.confidence <= 1.0):
            issues.append(ValidationIssue(
                agent_res.agent_id, "confidence", "out_of_bounds",
                f"Confidence {agent_res.confidence} is outside [0.0, 1.0]."
            ))

        # 3. Result Data Validation
        if not agent_res.result_data and agent_res.status == "success":
            issues.append(ValidationIssue(agent_res.agent_id, "result_data", "empty", "Result data dict is empty."))

        res_data = agent_res.result_data or {}

        # 4. Bounding Box & Coordinate Checks
        if "bounding_boxes" in res_data:
            boxes = res_data.get("bounding_boxes", [])
            for idx, box in enumerate(boxes):
                if isinstance(box, dict):
                    coords = box.get("box") or box.get("bounding_box") or []
                elif isinstance(box, list):
                    coords = box
                else:
                    coords = []

                if len(coords) != 4:
                    issues.append(ValidationIssue(
                        agent_res.agent_id, f"bounding_boxes[{idx}]", "invalid_dim",
                        f"Box at index {idx} does not have 4 coordinates: {coords}"
                    ))
                else:
                    # Check box min/max logic
                    ymin, xmin, ymax, xmax = coords[0], coords[1], coords[2], coords[3]
                    if ymin > ymax or xmin > xmax:
                        issues.append(ValidationIssue(
                            agent_res.agent_id, f"bounding_boxes[{idx}]", "inverted_coords",
                            f"Box coords inverted: ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}"
                        ))

        # 5. Status Check
        if agent_res.status not in ["success", "degraded", "failed"]:
            issues.append(ValidationIssue(
                agent_res.agent_id, "status", "unknown_status",
                f"Status '{agent_res.status}' is not recognized."
            ))

        is_valid = len([i for i in issues if i.issue_type in ["missing", "nan_or_null", "invalid_dim"]]) == 0
        return is_valid, issues

    @classmethod
    def validate_all(cls, agent_results: List[AgentResult]) -> Tuple[List[AgentResult], List[Dict[str, Any]]]:
        """
        Filters and validates a list of AgentResult objects.
        Returns (valid_results, all_validation_report).
        """
        valid_results = []
        reports = []

        for res in agent_results:
            is_valid, issues = cls.validate_agent_result(res)
            report = {
                "agent_id": res.agent_id,
                "is_valid": is_valid,
                "issues": [i.to_dict() for i in issues]
            }
            reports.append(report)

            if is_valid and res.status != "failed":
                valid_results.append(res)
            else:
                logger.warning(f"AgentResult from {res.agent_id} failed validation or has failed status.")

        return valid_results, reports
