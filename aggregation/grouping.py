"""
SatQuery AI - Entity Grouping & Duplicate Detector (Phases 5 & 6)
================================================================
Groups spatial and textual evidence across agents, calculates IoU overlap,
and detects/removes duplicates.
"""

from typing import List, Dict, Any, Tuple
import math
from aggregation.schemas import SpatialEvidence, NormalizedResult


def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) for two boxes formatted as [ymin, xmin, ymax, xmax].
    """
    if not boxA or not boxB or len(boxA) != 4 or len(boxB) != 4:
        return 0.0

    yA_min, xA_min, yA_max, xA_max = boxA
    yB_min, xB_min, yB_max, xB_max = boxB

    # Determine intersection rectangle
    inter_ymin = max(yA_min, yB_min)
    inter_xmin = max(xA_min, xB_min)
    inter_ymax = min(yA_max, yB_max)
    inter_xmax = min(xA_max, xB_max)

    inter_height = max(0.0, inter_ymax - inter_ymin)
    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_area = inter_height * inter_width

    if inter_area == 0.0:
        return 0.0

    # Area of each box
    areaA = (yA_max - yA_min) * (xA_max - xA_min)
    areaB = (yB_max - yB_min) * (xB_max - xB_min)

    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def compute_text_similarity(str1: str, str2: str) -> float:
    """
    Computes word-based Jaccard similarity between two strings.
    """
    if not str1 or not str2:
        return 0.0
    
    words1 = set(str1.lower().strip().split())
    words2 = set(str2.lower().strip().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


class EntityGrouper:
    """
    Clusters spatial evidence and textual answers into logical groups.
    """

    @classmethod
    def group_spatial_evidence(
        cls,
        spatial_items: List[SpatialEvidence],
        iou_threshold: float = 0.50
    ) -> List[List[SpatialEvidence]]:
        """
        Groups spatial evidence boxes using IoU graph connectivity.
        """
        if not spatial_items:
            return []

        clusters: List[List[SpatialEvidence]] = []
        visited = [False] * len(spatial_items)

        for i in range(len(spatial_items)):
            if visited[i]:
                continue
            
            cluster = [spatial_items[i]]
            visited[i] = True

            for j in range(i + 1, len(spatial_items)):
                if visited[j]:
                    continue
                
                boxA = spatial_items[i].bounding_box
                boxB = spatial_items[j].bounding_box
                
                if boxA and boxB and compute_box_iou(boxA, boxB) >= iou_threshold:
                    cluster.append(spatial_items[j])
                    visited[j] = True

            clusters.append(cluster)

        return clusters


class DuplicateDetector:
    """
    Identifies redundant predictions and produces deduplicated evidence lists.
    """

    @classmethod
    def deduplicate_spatial_evidence(
        cls,
        spatial_items: List[SpatialEvidence],
        iou_threshold: float = 0.50
    ) -> Tuple[List[SpatialEvidence], int]:
        """
        Performs Non-Maximum Suppression (NMS) style deduplication on spatial evidence.
        Returns (deduplicated_items, duplicates_removed_count).
        """
        if not spatial_items:
            return [], 0

        # Sort by confidence descending
        sorted_items = sorted(spatial_items, key=lambda x: x.confidence, reverse=True)
        kept_items: List[SpatialEvidence] = []
        duplicates_count = 0

        for item in sorted_items:
            if not item.bounding_box:
                kept_items.append(item)
                continue

            is_duplicate = False
            for kept in kept_items:
                if kept.bounding_box and compute_box_iou(item.bounding_box, kept.bounding_box) >= iou_threshold:
                    is_duplicate = True
                    duplicates_count += 1
                    break

            if not is_duplicate:
                kept_items.append(item)

        return kept_items, duplicates_count

    @classmethod
    def deduplicate_text_answers(
        cls,
        vqa_answers: List[str],
        similarity_threshold: float = 0.80
    ) -> List[str]:
        """
        Deduplicates close textual VQA answers.
        """
        unique_answers: List[str] = []
        for ans in vqa_answers:
            if not ans:
                continue
            is_dup = False
            for u_ans in unique_answers:
                if compute_text_similarity(ans, u_ans) >= similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique_answers.append(ans)

        return unique_answers
