"""
Ambiguity & Compatibility Detector for SatQuery AI.
Verifies that input image count, modalities (Optical/SAR), formats, and permissions
match the requirements of the classified remote-sensing task.
"""

from typing import Tuple, List, Dict, Any

from .models import (
    TaskType,
    ImageMetadata,
    Modality,
    ExtractedEntities,
    ProjectContext,
    InputCompatibility,
    AmbiguityReport,
    ExecutionTraceStep
)


class AmbiguityDetector:
    """
    Validates input compatibility against task requirements and flags ambiguities or missing inputs.
    """

    def validate(
        self,
        task_type: TaskType,
        images: List[ImageMetadata],
        entities: ExtractedEntities,
        temporal_structure: str,
        detected_modalities: List[str],
        formats: List[str],
        sensors: List[str],
        project_context: ProjectContext
    ) -> Tuple[InputCompatibility, AmbiguityReport, ExecutionTraceStep]:
        """
        Validates task requirements against image metadata and user permissions.
        """
        issues: List[str] = []
        missing_inputs: List[str] = []
        suggested_remediations: List[str] = []

        is_compatible = True
        status = "VALID"

        # 1. Bi-Temporal Change Detection Validation
        if task_type in (TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING, TaskType.BI_TEMPORAL_CHANGE_QUANTIFICATION):
            if len(images) < 2:
                is_compatible = False
                status = "INCOMPATIBLE"
                issue_msg = f"Task '{task_type.value}' requires a bi-temporal image pair (>=2 images at different dates), but only {len(images)} image was provided."
                issues.append(issue_msg)
                missing_inputs.append("Second temporal image (t2)")
                suggested_remediations.append("Upload a second image of the same geographic region acquired at a different date.")

        # 2. Cross-Modal Optical-SAR Fusion Validation
        if task_type == TaskType.CROSS_MODAL_OPTICAL_SAR_FUSION:
            has_optical = any(m in ("OPTICAL", "MULTISPECTRAL") for m in detected_modalities)
            has_sar = "SAR" in detected_modalities

            if not (has_optical and has_sar):
                is_compatible = False
                status = "INCOMPATIBLE"
                missing_mod = "SAR" if not has_sar else "Optical/Multispectral"
                issue_msg = f"Optical-SAR Fusion task requires co-registered Optical and SAR imagery, but missing '{missing_mod}' modality."
                issues.append(issue_msg)
                missing_inputs.append(f"{missing_mod} image")
                suggested_remediations.append(f"Upload a co-registered {missing_mod} image (e.g. RISAT SAR or Cartosat-2S Optical).")

        # 3. Text-Guided Grounding Validation
        if task_type == TaskType.TEXT_GUIDED_GROUNDING:
            if not entities.target_classes and not entities.is_coreference_resolved:
                status = "AMBIGUOUS"
                issue_msg = "Grounding query requested highlighting/segmentation, but target object class was not specified."
                issues.append(issue_msg)
                missing_inputs.append("Target object class (e.g., 'water body', 'built-up area')")
                suggested_remediations.append("Specify which feature or land-cover class to highlight in the query.")

        # 4. Image Count Validation
        if len(images) == 0:
            is_compatible = False
            status = "INCOMPATIBLE"
            issues.append("No satellite imagery provided for analysis.")
            missing_inputs.append("Satellite image (GeoTIFF/TIFF/PNG/JPEG)")
            suggested_remediations.append("Upload at least one GeoTIFF/TIFF/PNG/JPEG satellite image to proceed.")

        # 5. Permission Check
        if not project_context.can_execute_heavy_models and task_type in (TaskType.CROSS_MODAL_OPTICAL_SAR_FUSION, TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING):
            issues.append(f"User role '{project_context.user_role.value}' does not have execution permission for heavy multi-modal models.")
            suggested_remediations.append("Request project administrator to elevate execution permissions.")

        is_ambiguous = len(issues) > 0

        compatibility = InputCompatibility(
            is_compatible=is_compatible,
            status=status,
            detected_modalities=detected_modalities,
            image_count=len(images),
            formats=formats,
            sensors=sensors,
            temporal_structure=temporal_structure
        )

        ambiguity_report = AmbiguityReport(
            is_ambiguous=is_ambiguous,
            issues=issues,
            missing_inputs=missing_inputs,
            suggested_remediations=suggested_remediations
        )

        trace_msg = (
            f"Compatibility Status: '{status}'. "
            f"Ambiguous/Issues: {is_ambiguous} ({len(issues)} issue(s) detected)."
        )

        trace = ExecutionTraceStep(
            step="AmbiguityDetector",
            status="SUCCESS" if is_compatible and not is_ambiguous else "WARNING" if is_ambiguous and is_compatible else "ERROR",
            details=trace_msg
        )

        return compatibility, ambiguity_report, trace
