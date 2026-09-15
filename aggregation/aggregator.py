"""
SatQuery AI - Master Aggregator Engine (Phase 15)
=================================================
Master SatQueryAggregator orchestrates all 14 aggregation phases:
Collect -> Validate -> Normalize -> Deduplicate -> Resolve Conflicts ->
Fuse Evidence -> Calculate Confidence -> Synthesize Answer -> Render Visuals ->
Return AggregationResult with auditable summary metrics.
"""

import time
from typing import List, Dict, Any, Optional, Union
import logging

from execution.schemas import ExecutionResult
from orchestration.schemas import OrchestrationResult
from aggregation.schemas import (
    AgentResult,
    NormalizedResult,
    SpatialEvidence,
    TemporalEvidence,
    ModalEvidence,
    UnifiedEvidenceBundle,
    AggregationResult,
    AggregationMetrics
)
from aggregation.validator import ResultValidator
from aggregation.normalizer import ResultNormalizer
from aggregation.grouping import DuplicateDetector
from aggregation.conflict_resolver import ConflictResolver
from aggregation.confidence_calculator import ConfidenceCalculator
from aggregation.evidence_fusion import (
    SpatialEvidenceFusion,
    TemporalEvidenceFusion,
    OpticalSarEvidenceFusion
)
from aggregation.evidence_builder import UnifiedEvidenceBuilder
from aggregation.synthesizer import AnswerSynthesizer
from aggregation.visual_generator import VisualEvidenceGenerator

logger = logging.getLogger("SatQuery.Aggregation.Aggregator")


class SatQueryAggregator:
    """
    Master Aggregation Controller for SatQuery AI.
    Converts multi-agent execution outputs into evidence-grounded final responses.
    """

    def __init__(self, visual_output_dir: Optional[str] = None):
        self.visual_generator = VisualEvidenceGenerator(output_dir=visual_output_dir)

    def aggregate(
        self,
        raw_query: str,
        input_data: Union[OrchestrationResult, List[ExecutionResult], List[AgentResult], List[Dict[str, Any]]],
        request_id: Optional[str] = None,
        image_inputs: Optional[List[Dict[str, Any]]] = None
    ) -> AggregationResult:
        """
        Executes complete 15-phase aggregation pipeline.
        """
        start_time = time.time()
        req_id = request_id or f"req_agg_{int(time.time())}"

        # -------------------------------------------------------------
        # Phase 1: Collect & Standardize Input into AgentResult list
        # -------------------------------------------------------------
        agent_results: List[AgentResult] = []

        if isinstance(input_data, OrchestrationResult):
            for res in input_data.completed_agent_results:
                agent_results.append(AgentResult(
                    agent_id=res.agent,
                    task_type=res.result.get("task", "remote_sensing_task"),
                    status=res.status.value,
                    confidence=res.confidence,
                    execution_time=res.execution_time,
                    result_data=res.result,
                    evidence_items=[e.model_dump() for e in res.evidence],
                    domain_adaptation=res.result.get("domain_adaptation"),
                    modality=res.result.get("modality"),
                    execution_id=res.execution_id
                ))
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, AgentResult):
                    agent_results.append(item)
                elif isinstance(item, ExecutionResult):
                    agent_results.append(AgentResult(
                        agent_id=item.agent,
                        task_type=item.result.get("task", "remote_sensing_task"),
                        status=item.status.value,
                        confidence=item.confidence,
                        execution_time=item.execution_time,
                        result_data=item.result,
                        evidence_items=[e.model_dump() for e in item.evidence],
                        domain_adaptation=item.result.get("domain_adaptation"),
                        modality=item.result.get("modality"),
                        execution_id=item.execution_id
                    ))
                elif isinstance(item, dict):
                    agent_results.append(AgentResult(
                        agent_id=item.get("agent", item.get("agent_id", "agent_unknown")),
                        task_type=item.get("task_type", item.get("task", "vqa")),
                        status=item.get("status", "success"),
                        confidence=float(item.get("confidence", 0.85)),
                        execution_time=float(item.get("execution_time", 0.1)),
                        result_data=item.get("result", item.get("result_data", {})),
                        domain_adaptation=item.get("domain_adaptation")
                    ))

        metrics = AggregationMetrics(total_agents_collected=len(agent_results))

        # -------------------------------------------------------------
        # Phase 3: Validate Agent Results
        # -------------------------------------------------------------
        valid_agent_results, validation_reports = ResultValidator.validate_all(agent_results)
        metrics.valid_agents_count = len(valid_agent_results)

        # Fallback if no valid results
        if not valid_agent_results:
            logger.warning("No valid agent results passed validation.")
            empty_bundle = UnifiedEvidenceBuilder.build_bundle(
                request_id=req_id,
                spatial_evidence=[],
                temporal_evidence=[],
                modal_evidence=[],
                conflict_log=[],
                normalized_results=[]
            )
            return AggregationResult(
                request_id=req_id,
                status="degraded",
                final_answer="Unable to extract conclusive remote sensing evidence from input query.",
                aggregated_confidence=0.0,
                evidence_bundle=empty_bundle,
                metrics=metrics
            )

        # -------------------------------------------------------------
        # Phase 4: Normalize Results
        # -------------------------------------------------------------
        normalized_results: List[NormalizedResult] = [
            ResultNormalizer.normalize(res) for res in valid_agent_results
        ]

        # -------------------------------------------------------------
        # Phase 7: Resolve Conflicts
        # -------------------------------------------------------------
        winning_vqa, vqa_conflicts, vqa_has_conflict = ConflictResolver.resolve_vqa_conflicts(normalized_results)

        # -------------------------------------------------------------
        # Phases 9, 10 & 11: Fuse Evidence
        # -------------------------------------------------------------
        fused_spatial = SpatialEvidenceFusion.fuse_spatial_items(normalized_results)
        fused_temporal = TemporalEvidenceFusion.fuse_temporal_items(normalized_results)
        fused_modal = OpticalSarEvidenceFusion.fuse_modal_items(normalized_results)

        # Resolve temporal conflicts if needed
        fused_temporal, temp_conflicts, temp_has_conflict = ConflictResolver.resolve_temporal_conflicts(fused_temporal)

        conflict_log = vqa_conflicts + temp_conflicts
        has_conflicts = vqa_has_conflict or temp_has_conflict
        metrics.conflicts_detected = len(conflict_log)
        metrics.conflicts_resolved = len(conflict_log)

        metrics.spatial_features_fused = len(fused_spatial)
        metrics.temporal_features_fused = len(fused_temporal)
        metrics.cross_modal_features_fused = len(fused_modal)

        # -------------------------------------------------------------
        # Phase 8: Calculate Composite Confidence
        # -------------------------------------------------------------
        aggregated_confidence = ConfidenceCalculator.calculate_aggregated_confidence(
            normalized_results=normalized_results,
            fused_spatial=fused_spatial,
            fused_temporal=fused_temporal,
            fused_modal=fused_modal,
            has_conflicts=has_conflicts
        )

        # -------------------------------------------------------------
        # Phase 14: Generate Visual Evidence Artifacts
        # -------------------------------------------------------------
        visual_urls: List[str] = []
        base_img_path = None
        if image_inputs and len(image_inputs) > 0:
            base_img_path = image_inputs[0].get("file_name") or image_inputs[0].get("file_path")

        if fused_spatial:
            sp_url = self.visual_generator.draw_bounding_boxes(
                spatial_evidence=fused_spatial,
                input_image_path=base_img_path,
                output_filename=f"grounded_{req_id}.png"
            )
            visual_urls.append(sp_url)

        if fused_temporal:
            temp_url = self.visual_generator.draw_change_map(
                temporal_evidence=fused_temporal,
                output_filename=f"change_{req_id}.png"
            )
            visual_urls.append(temp_url)

        if fused_modal or (image_inputs and len(image_inputs) > 1):
            modal_url = self.visual_generator.draw_optical_sar_panel(
                modal_evidence=fused_modal,
                output_filename=f"cross_modal_{req_id}.png"
            )
            visual_urls.append(modal_url)

        # -------------------------------------------------------------
        # Phase 12: Build Unified Evidence Bundle
        # -------------------------------------------------------------
        bundle = UnifiedEvidenceBuilder.build_bundle(
            request_id=req_id,
            spatial_evidence=fused_spatial,
            temporal_evidence=fused_temporal,
            modal_evidence=fused_modal,
            conflict_log=conflict_log,
            normalized_results=normalized_results,
            fused_visual_paths=visual_urls
        )

        # -------------------------------------------------------------
        # Phase 13: Synthesize Final Grounded Answer
        # -------------------------------------------------------------
        final_answer = AnswerSynthesizer.synthesize_answer(
            raw_query=raw_query,
            bundle=bundle,
            winning_vqa=winning_vqa,
            normalized_results=normalized_results,
            confidence=aggregated_confidence
        )

        # -------------------------------------------------------------
        # Phase 15: Build Final Aggregation Result & Auditable Summary
        # -------------------------------------------------------------
        metrics.aggregation_time_seconds = round(time.time() - start_time, 4)

        auditable_summary = {
            "request_id": req_id,
            "raw_query": raw_query,
            "agents_evaluated": [r.agent_id for r in valid_agent_results],
            "task_types": list(set(r.task_type for r in valid_agent_results)),
            "conflicts_log": conflict_log,
            "confidence_score": aggregated_confidence,
            "visual_artifacts_count": len(visual_urls),
            "execution_summary_trace": {
                "spatial_evidence_count": len(fused_spatial),
                "temporal_evidence_count": len(fused_temporal),
                "modal_evidence_count": len(fused_modal)
            }
        }

        return AggregationResult(
            request_id=req_id,
            status="success",
            final_answer=final_answer,
            aggregated_confidence=aggregated_confidence,
            evidence_bundle=bundle,
            visual_evidence_urls=visual_urls,
            auditable_summary=auditable_summary,
            metrics=metrics
        )
