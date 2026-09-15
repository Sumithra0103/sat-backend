"""
SatQuery AI - Evidence Builder (Phase 12)
=========================================
Assembles unified evidence bundles containing spatial, temporal, cross-modal evidence,
conflict logs, land cover summaries, and visual asset references.
"""

from typing import List, Dict, Any, Optional
import uuid

from aggregation.schemas import (
    NormalizedResult,
    SpatialEvidence,
    TemporalEvidence,
    ModalEvidence,
    UnifiedEvidenceBundle
)


class UnifiedEvidenceBuilder:
    """
    Constructs the cohesive UnifiedEvidenceBundle for downstream answer synthesis and API response.
    """

    @classmethod
    def build_bundle(
        cls,
        request_id: str,
        spatial_evidence: List[SpatialEvidence],
        temporal_evidence: List[TemporalEvidence],
        modal_evidence: List[ModalEvidence],
        conflict_log: List[Dict[str, Any]],
        normalized_results: List[NormalizedResult],
        fused_visual_paths: Optional[List[str]] = None
    ) -> UnifiedEvidenceBundle:
        """
        Builds UnifiedEvidenceBundle instance.
        """
        # Aggregate land cover summary if present in normalized results
        land_cover_summary: Dict[str, Any] = {}
        captions_collected = []

        for res in normalized_results:
            if res.captions:
                captions_collected.extend(res.captions)

        if captions_collected:
            land_cover_summary["captions"] = list(set(captions_collected))

        if spatial_evidence:
            land_cover_summary["detected_objects_count"] = len(spatial_evidence)
            land_cover_summary["spatial_labels"] = list(set(e.label for e in spatial_evidence))

        if temporal_evidence:
            land_cover_summary["change_types"] = [t.change_type for t in temporal_evidence]
            land_cover_summary["avg_change_magnitude"] = round(
                sum(t.change_magnitude for t in temporal_evidence) / len(temporal_evidence), 4
            )

        fused_change_path = next((t.change_map_path for t in temporal_evidence if t.change_map_path), None)

        return UnifiedEvidenceBundle(
            request_id=request_id,
            spatial_evidence=spatial_evidence,
            temporal_evidence=temporal_evidence,
            modal_evidence=modal_evidence,
            resolved_conflicts=conflict_log,
            fused_change_map_path=fused_change_path,
            fused_annotated_image_paths=fused_visual_paths or [],
            land_cover_summary=land_cover_summary
        )
