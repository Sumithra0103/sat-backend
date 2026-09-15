"""
SatQuery AI - Answer Synthesizer (Phase 13)
===========================================
Generates clear, evidence-grounded natural language responses backed by
fused spatial, temporal, and cross-modal evidence items.
"""

from typing import List, Dict, Any, Optional
import logging
from aggregation.schemas import UnifiedEvidenceBundle, NormalizedResult

logger = logging.getLogger("SatQuery.Aggregation.AnswerSynthesizer")


class AnswerSynthesizer:
    """
    Synthesizes natural language answer grounded in evidence.
    """

    @classmethod
    def synthesize_answer(
        cls,
        raw_query: str,
        bundle: UnifiedEvidenceBundle,
        winning_vqa: Optional[str],
        normalized_results: List[NormalizedResult],
        confidence: float
    ) -> str:
        """
        Synthesizes cohesive answer text.
        """
        paragraphs: List[str] = []

        # 1. Primary Direct Answer
        if winning_vqa:
            paragraphs.append(f"**Direct Answer:** {winning_vqa}")
        elif bundle.land_cover_summary.get("captions"):
            caps = bundle.land_cover_summary["captions"]
            paragraphs.append(f"**Scene Overview:** {caps[0]}")

        # 2. Change Analysis Summary (if temporal evidence present)
        if bundle.temporal_evidence:
            change_lines = []
            for item in bundle.temporal_evidence:
                change_desc = (
                    f"Between the bi-temporal observations, **{item.change_type.replace('_', ' ')}** was detected "
                    f"over an estimated area of {item.changed_area_sq_km} sq km (relative magnitude: {int(item.change_magnitude * 100)}%). "
                    f"Baseline state '{item.pre_state}' transitioned to '{item.post_state}'."
                )
                change_lines.append(change_desc)
            paragraphs.append(f"**Multitemporal Change Analysis:**\n" + "\n".join(f"- {line}" for line in change_lines))

        # 3. Optical-SAR Cross-Modal Synergy (if modal evidence present)
        if bundle.modal_evidence:
            modal_lines = []
            for item in bundle.modal_evidence:
                modal_desc = (
                    f"**{item.feature_name.replace('_', ' ').title()}**: {item.synergy_description}. "
                    f"Optical observation confirmed {item.optical_finding.lower()}, while SAR radar confirmed {item.sar_finding.lower()}."
                )
                modal_lines.append(modal_desc)
            paragraphs.append(f"**Joint Optical-SAR Analysis:**\n" + "\n".join(f"- {line}" for line in modal_lines))

        # 4. Spatial Evidence & Region Grounding
        if bundle.spatial_evidence:
            sp_boxes = [e for e in bundle.spatial_evidence if e.bounding_box]
            if sp_boxes:
                grounding_tokens = [
                    f"`{b.label}` at box [{', '.join(str(c) for c in b.bounding_box)}] (Confidence: {int(b.confidence * 100)}%)"
                    for b in sp_boxes[:5]
                ]
                paragraphs.append(f"**Grounded Regions:** Identified {len(sp_boxes)} spatial region(s):\n" + "\n".join(f"- {t}" for t in grounding_tokens))

        # 5. Evidence & Confidence Grounding Footnote
        conf_pct = int(confidence * 100)
        paragraphs.append(
            f"*Auditable Evidence Summary: Response synthesized from {len(normalized_results)} specialist agent(s) "
            f"with a composite calibrated confidence of {conf_pct}%.*"
        )

        return "\n\n".join(paragraphs)
