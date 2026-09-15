"""
Query Classifier for SatQuery AI.
Classifies query into mandatory Remote Sensing task categories and specifies
required specialist tools, benchmark references, and permitted parameters.
"""

from typing import Tuple, List, Dict, Any

from .models import (
    TaskType,
    ExtractedEntities,
    TaskClassification,
    RoutingMetadata,
    ExecutionTraceStep
)


class QueryClassifier:
    """
    Classifies queries into defined TaskTypes and maps to specialist tools & benchmarks.
    """

    TASK_TOOL_MAPPING = {
        TaskType.SINGLE_IMAGE_CAPTIONING: {
            "agent_family": "CaptioningSpecialist",
            "tools": ["RSVQA_Captioner", "VRSBench_Scene_Describer"],
            "benchmark": "VRSBench / RSVQA",
            "capabilities": ["single_image_captioning", "land_cover_description"]
        },
        TaskType.SINGLE_IMAGE_VQA: {
            "agent_family": "VQASpecialist",
            "tools": ["RSVQA_Model", "VRSBench_VQA_Model"],
            "benchmark": "RSVQA",
            "capabilities": ["single_image_vqa"]
        },
        TaskType.TEXT_GUIDED_GROUNDING: {
            "agent_family": "GroundingSpecialist",
            "tools": ["VRSBench_Grounding_Model", "RS_Visual_Grounding_DINO"],
            "benchmark": "VRSBench",
            "capabilities": ["text_guided_grounding", "bounding_box_generation", "mask_segmentation"]
        },
        TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING: {
            "agent_family": "ChangeAnalysisSpecialist",
            "tools": ["CDVQA_Model", "BiTemporal_Change_Detector"],
            "benchmark": "CDVQA",
            "capabilities": ["bitemporal_change_description", "spatial_change_mapping"]
        },
        TaskType.BI_TEMPORAL_CHANGE_QUANTIFICATION: {
            "agent_family": "ChangeAnalysisSpecialist",
            "tools": ["CDVQA_Model", "Change_Trend_Quantifier"],
            "benchmark": "CDVQA",
            "capabilities": ["bitemporal_change_quantification", "trend_classification"]
        },
        TaskType.CROSS_MODAL_OPTICAL_SAR_FUSION: {
            "agent_family": "OpticalSARFusionSpecialist",
            "tools": ["BigEarthNet_Fusion_Model", "Optical_SAR_Information_Extractor"],
            "benchmark": "BigEarthNet / ISRO SAC",
            "capabilities": ["optical_sar_joint_extraction", "cross_modal_reasoning"]
        },
        TaskType.UNKNOWN: {
            "agent_family": "GeneralRSAssistant",
            "tools": ["RS_General_VLM"],
            "benchmark": "General",
            "capabilities": ["general_vqa"]
        }
    }

    def classify(
        self,
        query: str,
        entities: ExtractedEntities,
        temporal_structure: str,
        detected_modalities: List[str]
    ) -> Tuple[TaskClassification, RoutingMetadata, ExecutionTraceStep]:
        """
        Classifies task based on entities, query semantics, and input context.
        """
        query_lower = query.lower()
        actions = set(entities.spatial_actions)

        # 1. Classification Logic
        primary_task = TaskType.UNKNOWN
        confidence = 0.85

        # Check Cross-Modal Fusion
        if "cross_modal_fuse" in actions or "together" in query_lower or temporal_structure == "CROSS_MODAL":
            primary_task = TaskType.CROSS_MODAL_OPTICAL_SAR_FUSION
            confidence = 0.98
        # Check Change Quantification
        elif "change_quantify" in actions or any(k in query_lower for k in ["increased", "decreased", "remained unchanged"]):
            primary_task = TaskType.BI_TEMPORAL_CHANGE_QUANTIFICATION
            confidence = 0.96
        # Check Change Understanding
        elif "change_detect" in actions or "what changed" in query_lower or temporal_structure in ("BI_TEMPORAL", "MULTI_TEMPORAL"):
            primary_task = TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING
            confidence = 0.95
        # Check Grounding / Highlighting
        elif "highlight_ground" in actions or any(k in query_lower for k in ["highlight", "segment", "locate", "outline"]):
            primary_task = TaskType.TEXT_GUIDED_GROUNDING
            confidence = 0.94
        # Check Captioning / Land-cover Description
        elif "describe_caption" in actions or "land-cover" in query_lower or "major objects" in query_lower:
            primary_task = TaskType.SINGLE_IMAGE_CAPTIONING
            confidence = 0.93
        # Fallback Single Image VQA
        else:
            primary_task = TaskType.SINGLE_IMAGE_VQA
            confidence = 0.88

        # 2. Extract Tool & Benchmark Mapping
        mapping = self.TASK_TOOL_MAPPING[primary_task]

        classification = TaskClassification(
            primary_task=primary_task,
            confidence=confidence,
            required_specialist_tools=mapping["tools"],
            benchmark_reference=mapping["benchmark"]
        )

        # 3. Configure Permitted Parameters
        permitted_params = {
            "confidence_threshold": 0.5,
            "return_visual_evidence": True,
            "output_format": "JSON_PLUS_VISUAL_OVERLAY"
        }
        if primary_task == TaskType.TEXT_GUIDED_GROUNDING:
            permitted_params["target_entity"] = entities.target_classes[0] if entities.target_classes else "object"
            permitted_params["bbox_format"] = "xyxy_normalized"
        elif primary_task in (TaskType.BI_TEMPORAL_CHANGE_UNDERSTANDING, TaskType.BI_TEMPORAL_CHANGE_QUANTIFICATION):
            permitted_params["spatial_change_map"] = True
            permitted_params["diff_metric"] = "normalized_difference_index"

        routing = RoutingMetadata(
            target_agent_family=mapping["agent_family"],
            required_model_capabilities=mapping["capabilities"],
            permitted_parameters=permitted_params
        )

        trace = ExecutionTraceStep(
            step="QueryClassifier",
            status="SUCCESS",
            details=f"Classified as '{primary_task.value}' with confidence {confidence:.2f}. Agent Family: '{mapping['agent_family']}'."
        )

        return classification, routing, trace
