"""
SatQuery AI - Result Normalizer (Phase 4)
=========================================
Normalizes text, bounding box coordinates [ymin, xmin, ymax, xmax], confidence bounds,
and domain adaptation weights into unified NormalizedResult instances.
"""

from typing import List, Dict, Any, Optional
import math

from aggregation.schemas import (
    AgentResult,
    NormalizedResult,
    SpatialEvidence,
    TemporalEvidence,
    ModalEvidence,
    SpatialType
)


class ResultNormalizer:
    """
    Translates raw AgentResult outputs into standardized NormalizedResult schema objects.
    """

    DOMAIN_WEIGHT_MAP = {
        "BigEarthNet": 1.25,
        "CDVQA": 1.30,
        "VRSBench": 1.20,
        "RSVQA": 1.20,
        "ISRO-SAC": 1.35,
        "default": 1.00
    }

    @classmethod
    def normalize_box(cls, box: List[float], img_width: int = 512, img_height: int = 512) -> List[float]:
        """
        Ensures bounding box format is normalized [ymin, xmin, ymax, xmax] in range [0.0, 1.0].
        Converts pixel coordinates if coords > 1.0.
        """
        if len(box) != 4:
            return [0.0, 0.0, 1.0, 1.0]

        # Check if bounds are in pixel coordinates (> 1.0)
        c0, c1, c2, c3 = box[0], box[1], box[2], box[3]
        if any(c > 1.0 for c in [c0, c1, c2, c3]):
            # If formatted as [x, y, w, h] or [xmin, ymin, xmax, ymax]
            # Standard pixel check:
            ymin = float(c0) / img_height if c0 > 1.0 else float(c0)
            xmin = float(c1) / img_width if c1 > 1.0 else float(c1)
            ymax = float(c2) / img_height if c2 > 1.0 else float(c2)
            xmax = float(c3) / img_width if c3 > 1.0 else float(c3)
        else:
            ymin, xmin, ymax, xmax = float(c0), float(c1), float(c2), float(c3)

        # Clamp into [0.0, 1.0]
        ymin = max(0.0, min(1.0, ymin))
        xmin = max(0.0, min(1.0, xmin))
        ymax = max(0.0, min(1.0, ymax))
        xmax = max(0.0, min(1.0, xmax))

        # Ensure min <= max
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        return [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]

    @classmethod
    def get_domain_weight(cls, domain_adaptation: Optional[str]) -> float:
        """Determines domain adaptation weight multiplier."""
        if not domain_adaptation:
            return cls.DOMAIN_WEIGHT_MAP["default"]
        
        for key, weight in cls.DOMAIN_WEIGHT_MAP.items():
            if key.lower() in domain_adaptation.lower():
                return weight
        return cls.DOMAIN_WEIGHT_MAP["default"]

    @classmethod
    def normalize(cls, agent_result: AgentResult) -> NormalizedResult:
        """
        Transforms single AgentResult to NormalizedResult.
        """
        res_data = agent_result.result_data or {}
        
        # 1. Text & VQA normalization
        vqa_answer = None
        captions = []
        if "answer" in res_data:
            vqa_answer = str(res_data["answer"]).strip()
        if "caption" in res_data:
            captions.append(str(res_data["caption"]).strip())
        if "captions" in res_data:
            captions.extend([str(c).strip() for c in res_data["captions"]])

        # 2. Extract Spatial Evidence
        spatial_list: List[SpatialEvidence] = []

        # Parse bounding boxes
        boxes = res_data.get("bounding_boxes", []) or res_data.get("grounded_regions", [])
        for idx, b in enumerate(boxes):
            label = "object"
            box_coords = [0.0, 0.0, 1.0, 1.0]
            conf = agent_result.confidence

            if isinstance(b, dict):
                label = b.get("label", b.get("class", "object"))
                box_coords = b.get("box") or b.get("bounding_box") or box_coords
                conf = b.get("confidence", conf)
            elif isinstance(b, list):
                box_coords = b

            norm_box = cls.normalize_box(box_coords)
            spatial_list.append(SpatialEvidence(
                label=label,
                spatial_type=SpatialType.BOUNDING_BOX,
                bounding_box=norm_box,
                confidence=min(1.0, max(0.0, conf)),
                source_agent=agent_result.agent_id
            ))

        # Parse segmentation masks or change maps
        if "change_map" in res_data:
            change_map_path = str(res_data["change_map"])
            spatial_list.append(SpatialEvidence(
                label="change_map",
                spatial_type=SpatialType.HEATMAP,
                mask_path=change_map_path,
                confidence=agent_result.confidence,
                source_agent=agent_result.agent_id
            ))

        # 3. Extract Temporal Evidence
        temporal_list: List[TemporalEvidence] = []
        if "change_type" in res_data or "change_summary" in res_data or agent_result.task_type in ["change_detection", "change_vqa"]:
            change_type = res_data.get("change_type", res_data.get("change_summary", "detected_change"))
            change_mag = float(res_data.get("change_magnitude", res_data.get("changed_area_ratio", 0.15)))
            temporal_list.append(TemporalEvidence(
                change_type=str(change_type),
                change_magnitude=min(1.0, max(0.0, change_mag)),
                changed_area_sq_km=res_data.get("changed_area_sq_km"),
                change_map_path=res_data.get("change_map"),
                pre_state=res_data.get("pre_state"),
                post_state=res_data.get("post_state"),
                confidence=agent_result.confidence,
                source_agent=agent_result.agent_id
            ))

        # 4. Extract Cross-Modal Evidence
        modal_list: List[ModalEvidence] = []
        if agent_result.task_type in ["optical_sar", "cross_modal_analysis"] or "optical_sar_features" in res_data:
            feat_name = res_data.get("feature_name", "optical_sar_synergy")
            opt_find = res_data.get("optical_finding", "Optical spectral land-cover visual features")
            sar_find = res_data.get("sar_finding", "SAR radar backscatter structural features")
            synergy = res_data.get("synergy_description", "Joint optical-SAR cross-validation")

            modal_list.append(ModalEvidence(
                feature_name=str(feat_name),
                optical_finding=str(opt_find),
                sar_finding=str(sar_find),
                synergy_description=str(synergy),
                confidence=agent_result.confidence,
                source_agent=agent_result.agent_id
            ))

        # Calibrated confidence with domain adaptation boost
        domain_weight = cls.get_domain_weight(agent_result.domain_adaptation)
        calibrated_conf = min(1.0, agent_result.confidence * min(1.15, domain_weight))

        norm_text = f"VQA: {vqa_answer}" if vqa_answer else ""
        if captions:
            norm_text += f" | Captions: {'; '.join(captions)}"

        return NormalizedResult(
            agent_id=agent_result.agent_id,
            task_type=agent_result.task_type,
            normalized_text=norm_text.strip(" |"),
            vqa_answer=vqa_answer,
            captions=captions,
            spatial_evidence=spatial_list,
            temporal_evidence=temporal_list,
            modal_evidence=modal_list,
            raw_confidence=agent_result.confidence,
            calibrated_confidence=calibrated_conf,
            domain_adaptation_weight=domain_weight
        )
