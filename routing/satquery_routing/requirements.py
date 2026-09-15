"""
Requirement Extraction and Capability Mapping for SatQuery AI Router.
Steps 3 & 4 of Router workflow:
- Extracts structured routing requirements from StructuredQueryObject (SQO).
- Maps high-level tasks to exact AgentCapability and InputModality enums.
"""

from typing import List, Dict, Any, Tuple
from schemas.agent_schema import AgentCapability, InputModality, InputFormat
from satquery_routing.models import RoutingRequirements


def _get_val(obj: Any, key: str, default: str = "") -> str:
    """Helper to safely extract string/enum value from either dict or object."""
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    if hasattr(val, "value"):
        val = val.value
    return str(val) if val is not None else default


class RequirementExtractor:
    """
    Translates Query Understanding outputs into formal routing constraints.
    """

    @staticmethod
    def map_task_to_capability(task_str: str) -> Tuple[AgentCapability, List[AgentCapability]]:
        """
        Maps a task string to a primary AgentCapability and secondary capabilities.
        """
        t = task_str.lower().strip()
        if "grounding" in t:
            return AgentCapability.REGION_GROUNDING, [AgentCapability.SINGLE_IMAGE_VQA]
        elif "change_vqa" in t or "quantif" in t:
            return AgentCapability.CHANGE_VQA, [AgentCapability.BITEMPORAL_CHANGE_DETECTION]
        elif "change" in t or "bitemporal" in t:
            return AgentCapability.BITEMPORAL_CHANGE_DETECTION, [AgentCapability.CHANGE_VQA]
        elif "cross_modal" in t or "fusion" in t:
            return AgentCapability.CROSS_MODAL_FUSION, [AgentCapability.SINGLE_IMAGE_VQA]
        elif "caption" in t or "describe" in t:
            return AgentCapability.SCENE_CAPTIONING, [AgentCapability.SINGLE_IMAGE_VQA]
        else:
            return AgentCapability.SINGLE_IMAGE_VQA, []

    @staticmethod
    def map_inputs_to_modality(
        image_inputs: List[Any],
        temporal_structure: str,
        detected_modalities: List[str] = None
    ) -> InputModality:
        """
        Determines the required input modality from image metadata and temporal structure.
        Works with both dicts and ImageMetadata objects.
        """
        modalities = [_get_val(img, "modality").upper() for img in image_inputs]
        if not modalities and detected_modalities:
            modalities = [str(m).upper() for m in detected_modalities]

        has_optical = any(m in ("OPTICAL", "MULTISPECTRAL") for m in modalities)
        has_sar = "SAR" in modalities

        temp_upper = (temporal_structure or "").upper()
        if "CROSS_MODAL" in temp_upper or (has_optical and has_sar):
            return InputModality.CROSS_MODAL_OPTICAL_SAR

        img_count = len(image_inputs)
        if img_count >= 2 or "BI_TEMPORAL" in temp_upper or "BITEMPORAL" in temp_upper:
            if has_sar and not has_optical:
                return InputModality.BITEMPORAL_SAR
            return InputModality.BITEMPORAL_OPTICAL
        elif img_count == 1:
            if has_sar and not has_optical:
                return InputModality.SAR
            return InputModality.OPTICAL

        # Fallback based on detected modalities
        if has_sar and not has_optical:
            return InputModality.SAR
        return InputModality.OPTICAL

    @staticmethod
    def normalize_temporal_structure(temporal_struct: str, image_count: int) -> str:
        """Normalizes temporal structure names across query understanding and router."""
        t = (temporal_struct or "").upper()
        if "CROSS_MODAL" in t:
            return "COREG_CROSS_MODAL_PAIR"
        elif "BI_TEMPORAL" in t or "BITEMPORAL" in t:
            return "BITEMPORAL_PAIR"
        elif "MULTI_TEMPORAL" in t:
            return "MULTI_TEMPORAL"
        elif image_count == 2:
            return "BITEMPORAL_PAIR"
        return "SINGLE_IMAGE"

    @classmethod
    def extract(cls, sqo: Any) -> RoutingRequirements:
        """
        Extracts RoutingRequirements model from SQO.
        Supports both object attribute access and dictionary access.
        """
        query_id = getattr(sqo, "query_id", "unknown_query")
        resolved_query = getattr(sqo, "effective_resolved_query", getattr(sqo, "raw_query", getattr(sqo, "original_query", "")))

        # Task classification
        task_cls = getattr(sqo, "task_classification", None)
        task_str = ""
        benchmark_ref = None
        if task_cls:
            primary_task_obj = getattr(task_cls, "primary_task", "")
            task_str = primary_task_obj.value if hasattr(primary_task_obj, "value") else str(primary_task_obj)
            benchmark_ref = getattr(task_cls, "benchmark_reference", None)

        primary_cap, sec_caps = cls.map_task_to_capability(task_str)

        # Image inputs & format
        image_inputs = getattr(sqo, "image_inputs", [])
        input_comp = getattr(sqo, "input_compatibility", None)
        raw_temp_struct = getattr(input_comp, "temporal_structure", "SINGLE_DATE") if input_comp else "SINGLE_DATE"
        detected_mods = getattr(input_comp, "detected_modalities", []) if input_comp else []

        temporal_struct = cls.normalize_temporal_structure(raw_temp_struct, len(image_inputs))
        req_modality = cls.map_inputs_to_modality(image_inputs, raw_temp_struct, detected_mods)

        # Formats
        formats = set()
        for img in image_inputs:
            fmt = _get_val(img, "format").lower()
            if fmt in ("geotiff", "tiff"):
                formats.add("geotiff")
                formats.add("tiff")
            elif fmt in ("png", "jpeg", "jpg"):
                formats.add("jpeg" if fmt == "jpg" else fmt)
        if not formats and input_comp:
            for fmt in getattr(input_comp, "formats", []):
                fmt_lower = str(fmt).lower()
                if fmt_lower in ("geotiff", "tiff"):
                    formats.add("geotiff")
                    formats.add("tiff")
                elif fmt_lower in ("png", "jpeg", "jpg"):
                    formats.add("jpeg" if fmt_lower == "jpg" else fmt_lower)
        if not formats:
            formats.add("geotiff")

        # Extracted entities
        extracted_entities = getattr(sqo, "extracted_entities", None)
        target_classes = getattr(extracted_entities, "target_classes", []) if extracted_entities else []

        requires_cloud_resilience = "cloud" in resolved_query.lower() or req_modality == InputModality.CROSS_MODAL_OPTICAL_SAR

        return RoutingRequirements(
            query_id=query_id,
            resolved_query=resolved_query,
            primary_capability=primary_cap.value,
            secondary_capabilities=[c.value for c in sec_caps],
            required_modality=req_modality.value,
            supported_formats=list(formats),
            benchmark_reference=benchmark_ref,
            target_classes=target_classes,
            image_count=len(image_inputs),
            temporal_structure=temporal_struct,
            max_acceptable_latency_ms=2500,
            requires_cloud_resilience=requires_cloud_resilience,
            requires_gpu=False
        )
