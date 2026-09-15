"""
SatQuery AI - Conflict Resolver (Phase 7)
========================================
Resolves contradictory predictions between specialist agents (VQA choices,
spatial locations, land cover classes, change directions) using weighted consensus.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
from aggregation.schemas import NormalizedResult, SpatialEvidence, TemporalEvidence
from aggregation.grouping import compute_text_similarity

logger = logging.getLogger("SatQuery.Aggregation.ConflictResolver")


class ConflictResolver:
    """
    Identifies and resolves conflicts across multiple agent results.
    """

    @classmethod
    def resolve_vqa_conflicts(
        cls,
        normalized_results: List[NormalizedResult]
    ) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
        """
        Resolves conflicts in VQA answers across multiple agents using weighted voting.
        Returns (winning_answer, conflict_audit_log, conflict_occurred).
        """
        vqa_candidates: Dict[str, float] = {}
        candidate_sources: Dict[str, List[str]] = {}
        conflict_log: List[Dict[str, Any]] = []

        vqa_results = [r for r in normalized_results if r.vqa_answer]

        if not vqa_results:
            return None, [], False

        # Check if there are differing answers
        answers = [r.vqa_answer for r in vqa_results if r.vqa_answer]
        has_conflict = len(set(a.lower().strip() for a in answers)) > 1

        for res in vqa_results:
            ans = res.vqa_answer.strip()
            # Score = calibrated_confidence * domain_adaptation_weight
            weight = res.calibrated_confidence * res.domain_adaptation_weight

            # Match with existing candidates by similarity
            matched_key = None
            for key in vqa_candidates:
                if compute_text_similarity(ans, key) > 0.70:
                    matched_key = key
                    break

            if matched_key:
                vqa_candidates[matched_key] += weight
                candidate_sources[matched_key].append(res.agent_id)
            else:
                vqa_candidates[ans] = weight
                candidate_sources[ans] = [res.agent_id]

        # Select highest weighted candidate
        winning_answer = max(vqa_candidates.items(), key=lambda x: x[1])[0]

        if has_conflict:
            conflict_log.append({
                "type": "vqa_conflict",
                "competing_answers": list(vqa_candidates.keys()),
                "weights": {k: round(v, 4) for k, v in vqa_candidates.items()},
                "sources": candidate_sources,
                "winner": winning_answer,
                "resolution_strategy": "weighted_domain_adaptation_voting"
            })
            logger.info(f"VQA Conflict resolved. Winner: '{winning_answer}' among {list(vqa_candidates.keys())}")

        return winning_answer, conflict_log, has_conflict

    @classmethod
    def resolve_temporal_conflicts(
        cls,
        temporal_evidences: List[TemporalEvidence]
    ) -> Tuple[List[TemporalEvidence], List[Dict[str, Any]], bool]:
        """
        Resolves conflicts in temporal change predictions (e.g. built_up_increase vs built_up_decrease).
        """
        if not temporal_evidences or len(temporal_evidences) <= 1:
            return temporal_evidences, [], False

        # Group by change class family
        change_groups: Dict[str, List[TemporalEvidence]] = {}
        for ev in temporal_evidences:
            key = ev.change_type.lower()
            if key not in change_groups:
                change_groups[key] = []
            change_groups[key].append(ev)

        has_conflict = len(change_groups) > 1
        conflict_log = []
        resolved_evidences = []

        if has_conflict:
            group_weights = {}
            for key, ev_list in change_groups.items():
                w = sum(e.confidence for e in ev_list)
                group_weights[key] = w

            winning_key = max(group_weights.items(), key=lambda x: x[1])[0]
            resolved_evidences = change_groups[winning_key]

            conflict_log.append({
                "type": "temporal_change_conflict",
                "competing_types": list(group_weights.keys()),
                "weights": {k: round(v, 4) for k, v in group_weights.items()},
                "winner": winning_key,
                "resolution_strategy": "weighted_confidence_voting"
            })
        else:
            resolved_evidences = temporal_evidences

        return resolved_evidences, conflict_log, has_conflict
