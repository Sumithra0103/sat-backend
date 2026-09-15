"""
SatQuery AI - Aggregation Subsystem Unit & Integration Tests (Phase 18)
=======================================================================
Tests all 15 aggregation phases:
- Validator & Normalizer
- Grouping & Duplicate Detection (IoU & NMS)
- Conflict Resolution
- Confidence Calculation
- Evidence Fusion (Spatial, Temporal, Optical-SAR)
- Answer Synthesizer
- Visual Generator
- SatQueryAggregator Master Engine
"""

import os
import unittest
from pathlib import Path


from aggregation.schemas import (
    AgentResult,
    NormalizedResult,
    SpatialEvidence,
    TemporalEvidence,
    ModalEvidence,
    SpatialType,
    AggregationResult
)
from aggregation.validator import ResultValidator
from aggregation.normalizer import ResultNormalizer
from aggregation.grouping import EntityGrouper, DuplicateDetector, compute_box_iou
from aggregation.conflict_resolver import ConflictResolver
from aggregation.confidence_calculator import ConfidenceCalculator
from aggregation.evidence_fusion import SpatialEvidenceFusion, TemporalEvidenceFusion, OpticalSarEvidenceFusion
from aggregation.evidence_builder import UnifiedEvidenceBuilder
from aggregation.synthesizer import AnswerSynthesizer
from aggregation.visual_generator import VisualEvidenceGenerator
from aggregation.aggregator import SatQueryAggregator


def test_result_validator():
    valid_res = AgentResult(
        agent_id="test_agent_1",
        task_type="vqa",
        confidence=0.92,
        result_data={"answer": "Urban residential area"}
    )
    is_valid, issues = ResultValidator.validate_agent_result(valid_res)
    assert is_valid is True
    assert len(issues) == 0


def test_result_normalizer_and_box_clamping():
    res = AgentResult(
        agent_id="grounding_agent",
        task_type="grounding",
        confidence=0.88,
        domain_adaptation="BigEarthNet",
        result_data={
            "bounding_boxes": [
                {"label": "water_body", "box": [10, 20, 100, 200]},  # Pixel coords
                {"label": "built_up", "box": [0.1, 0.2, 0.5, 0.6]}   # Normalized coords
            ]
        }
    )
    norm = ResultNormalizer.normalize(res)
    assert norm.calibrated_confidence > 0.88  # Boosted by domain adaptation weight
    assert len(norm.spatial_evidence) == 2
    box0 = norm.spatial_evidence[0].bounding_box
    assert 0.0 <= box0[0] <= 1.0


def test_box_iou_and_duplicate_detector():
    boxA = [0.1, 0.1, 0.5, 0.5]
    boxB = [0.1, 0.1, 0.5, 0.5]  # Identical box
    iou = compute_box_iou(boxA, boxB)
    assert abs(iou - 1.0) < 0.01


    boxC = [0.6, 0.6, 0.9, 0.9]  # Disjoint box
    assert compute_box_iou(boxA, boxC) == 0.0

    items = [
        SpatialEvidence(label="water", spatial_type=SpatialType.BOUNDING_BOX, bounding_box=boxA, confidence=0.9, source_agent="a1"),
        SpatialEvidence(label="water", spatial_type=SpatialType.BOUNDING_BOX, bounding_box=boxB, confidence=0.8, source_agent="a2"),
        SpatialEvidence(label="building", spatial_type=SpatialType.BOUNDING_BOX, bounding_box=boxC, confidence=0.95, source_agent="a3")
    ]
    deduped, count = DuplicateDetector.deduplicate_spatial_evidence(items, iou_threshold=0.50)
    assert count == 1
    assert len(deduped) == 2


def test_conflict_resolver_vqa():
    norm1 = NormalizedResult(
        agent_id="agent_opt",
        task_type="vqa",
        vqa_answer="Water body increased in area",
        raw_confidence=0.90,
        calibrated_confidence=0.95,
        domain_adaptation_weight=1.30
    )
    norm2 = NormalizedResult(
        agent_id="agent_generic",
        task_type="vqa",
        vqa_answer="Water body remained unchanged",
        raw_confidence=0.60,
        calibrated_confidence=0.60,
        domain_adaptation_weight=1.00
    )
    winning_ans, conflict_log, has_conflict = ConflictResolver.resolve_vqa_conflicts([norm1, norm2])
    assert has_conflict is True
    assert winning_ans == "Water body increased in area"
    assert len(conflict_log) == 1


def test_confidence_calculator():
    norm1 = NormalizedResult(
        agent_id="agent_1", task_type="vqa", raw_confidence=0.90, calibrated_confidence=0.95, domain_adaptation_weight=1.25
    )
    norm2 = NormalizedResult(
        agent_id="agent_2", task_type="grounding", raw_confidence=0.85, calibrated_confidence=0.85, domain_adaptation_weight=1.0
    )
    conf = ConfidenceCalculator.calculate_aggregated_confidence(
        normalized_results=[norm1, norm2],
        fused_spatial=[],
        fused_temporal=[],
        fused_modal=[],
        has_conflicts=False
    )
    assert 0.80 <= conf <= 1.0


def test_full_satquery_aggregator():
    aggregator = SatQueryAggregator()

    raw_agent_results = [
        AgentResult(
            agent_id="rs_vqa_expert",
            task_type="vqa",
            confidence=0.92,
            domain_adaptation="BigEarthNet",
            result_data={"answer": "Significant increase in built-up urban infrastructure"}
        ),
        AgentResult(
            agent_id="cdvqa_change_expert",
            task_type="change_detection",
            confidence=0.89,
            domain_adaptation="CDVQA",
            result_data={
                "change_type": "built_up_increase",
                "change_magnitude": 0.28,
                "changed_area_sq_km": 3.4,
                "pre_state": "Vegetation and open soil",
                "post_state": "Paved roads and commercial buildings"
            }
        ),
        AgentResult(
            agent_id="optical_sar_fusion_model",
            task_type="optical_sar",
            confidence=0.95,
            domain_adaptation="ISRO-SAC",
            result_data={
                "feature_name": "sar_double_bounce_urban_verification",
                "optical_finding": "Optical red/NIR spectral reflectance indicates impervious built surfaces",
                "sar_finding": "SAR HH/HV backscatter exhibits strong double-bounce reflections from structures",
                "synergy_description": "Cross-modal Optical-SAR joint verification confirms newly constructed buildings under cloud-free and cloudy conditions"
            }
        )
    ]

    agg_result: AggregationResult = aggregator.aggregate(
        raw_query="What changed between these two dates and where did the change occur?",
        input_data=raw_agent_results,
        request_id="test_req_1001"
    )

    assert agg_result.status == "success"
    assert agg_result.aggregated_confidence >= 0.85
    assert "Direct Answer" in agg_result.final_answer or "Multitemporal Change Analysis" in agg_result.final_answer
    assert len(agg_result.visual_evidence_urls) > 0
    assert agg_result.metrics.total_agents_collected == 3
