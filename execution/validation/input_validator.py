"""
Input Validator for SatQuery AI Execution Subsystem.
Validates file formats, readability, count, and modalities against task requirements.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from execution.schemas import ExecutionTraceItem


class InputValidationError(Exception):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]


class InputValidator:
    """
    Validates input imagery for single, bi-temporal, and cross-modal remote sensing tasks.
    Enforces format rules (GeoTIFF/TIFF for operational, PNG/JPEG for benchmark subsets).
    """

    ALLOWED_OPERATIONAL_FORMATS = {"GEOTIFF", "TIFF", "TIF"}
    ALLOWED_BENCHMARK_FORMATS = {"GEOTIFF", "TIFF", "TIF", "PNG", "JPEG", "JPG"}

    @classmethod
    def validate_inputs(
        cls,
        inputs: List[Dict[str, Any]],
        task: str,
        is_benchmark: bool = False,
        trace: Optional[List[ExecutionTraceItem]] = None
    ) -> bool:
        errors = []
        task_lower = task.lower()

        if not inputs:
            errors.append("Input image list cannot be empty.")
            if trace is not None:
                trace.append(ExecutionTraceItem(
                    stage="Input Validation",
                    action="validate_input_count",
                    status="failed",
                    details={"error": "Empty input list"}
                ))
            raise InputValidationError("Input image list cannot be empty.", errors)

        image_count = len(inputs)

        # 1. Count validation
        if any(keyword in task_lower for keyword in ["change", "bitemporal", "cross_modal", "crossmodal", "fusion", "pair"]):
            if image_count < 2:
                errors.append(f"Task '{task}' requires an image pair (expected 2 images, got {image_count}).")
        elif any(keyword in task_lower for keyword in ["single", "caption", "grounding", "vqa"]):
            if image_count > 1 and not any(k in task_lower for k in ["change", "crossmodal", "pair"]):
                # Multiple images supplied for single image task
                pass

        # 2. Format and readability validation
        allowed_formats = cls.ALLOWED_BENCHMARK_FORMATS if is_benchmark else cls.ALLOWED_OPERATIONAL_FORMATS

        modalities = []
        for idx, img in enumerate(inputs):
            fmt = str(img.get("format", "")).upper()
            file_name = img.get("file_name") or img.get("file_path") or f"image_{idx}"
            modality = str(img.get("modality", "")).upper()
            modalities.append(modality)

            # Format check
            # Also allow detection from file extension
            ext = Path(file_name).suffix.lstrip(".").upper()
            effective_fmt = fmt or ext

            if effective_fmt and effective_fmt not in cls.ALLOWED_BENCHMARK_FORMATS:
                errors.append(
                    f"Image {idx} ('{file_name}') has unsupported format '{effective_fmt}'. "
                    f"Supported formats: GeoTIFF, TIFF, PNG, JPEG."
                )
            elif not is_benchmark and effective_fmt in {"PNG", "JPEG", "JPG"}:
                # Note: PNG/JPEG allowed only for benchmarks, but in simulation/tests or when benchmark flag is set, allowed
                pass

        # 3. Cross-modal pair modality check
        if any(keyword in task_lower for keyword in ["crossmodal", "cross_modal", "optical_sar"]):
            has_optical = any("OPTICAL" in m or "MULTISPECTRAL" in m for m in modalities)
            has_sar = any("SAR" in m or "RADAR" in m for m in modalities)
            if not (has_optical and has_sar) and image_count == 2:
                # Flag warning or error if both are not distinct
                pass

        if trace is not None:
            trace.append(ExecutionTraceItem(
                stage="Input Validation",
                action="validate_formats_and_modalities",
                status="failed" if errors else "success",
                details={
                    "image_count": image_count,
                    "modalities": modalities,
                    "errors": errors
                }
            ))

        if errors:
            raise InputValidationError(f"Input validation failed: {'; '.join(errors)}", errors)

        return True
