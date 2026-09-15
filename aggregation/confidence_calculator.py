"""
SatQuery AI - Confidence Calculator (Phase 8)
=============================================
Calculates composite aggregated confidence scores combining calibrated agent confidences,
domain adaptation bonuses, consensus agreement, and cross-modal synergy.
"""

from typing import List, Dict, Any
import math
from aggregation.schemas import NormalizedResult, SpatialEvidence, TemporalEvidence, ModalEvidence


class ConfidenceCalculator:
    """
    Computes unified composite confidence for aggregated responses.
    """

    @classmethod
    def calculate_aggregated_confidence(
        cls,
        normalized_results: List[NormalizedResult],
        fused_spatial: List[SpatialEvidence],
        fused_temporal: List[TemporalEvidence],
        fused_modal: List[ModalEvidence],
        has_conflicts: bool = False
    ) -> float:
        """
        Computes composite confidence C in range [0.0, 1.0]:
        C = w1 * C_agents + w2 * W_domain + w3 * S_consensus + w4 * M_synergy - Penalty_conflict
        """
        if not normalized_results:
            return 0.50

        # 1. Base Agent Confidence
        agent_confidences = [r.calibrated_confidence for r in normalized_results]
        c_agents = sum(agent_confidences) / len(agent_confidences)

        # 2. Domain Adaptation Bonus
        domain_weights = [r.domain_adaptation_weight for r in normalized_results]
        avg_domain_weight = sum(domain_weights) / len(domain_weights)
        w_domain_bonus = min(0.15, (avg_domain_weight - 1.0) * 0.5)

        # 3. Consensus Agreement Score
        s_consensus = 0.10 if len(normalized_results) > 1 and not has_conflicts else 0.0

        # 4. Multi-Modal & Temporal Synergy Bonus
        m_synergy = 0.05 if (fused_modal or fused_temporal) else 0.0

        # 5. Conflict Penalty
        penalty_conflict = 0.08 if has_conflicts else 0.0

        # Combine terms: base confidence gets 85% weight, domain bonus up to 15%
        raw_composite = (0.85 * c_agents) + w_domain_bonus + s_consensus + m_synergy - penalty_conflict


        # Clamp into [0.0, 1.0]
        final_confidence = min(1.0, max(0.0, raw_composite))

        return round(final_confidence, 4)
