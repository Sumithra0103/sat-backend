"""
RS-Change Specialist Agent Adapter for SatQuery AI Execution Subsystem.
Adapted on SECOND (Semantic Change Detection) and CDVQA benchmarks for bi-temporal change detection,
spatial change mapping, and change-based VQA.
"""

from typing import Dict, Any
from execution.adapters.base_adapter import BaseAgentAdapter
from execution.preprocessing.preprocessor import PreprocessedBatch
from execution.evidence.visualizer import EvidenceVisualizer


class RSChangeAdapter(BaseAgentAdapter):
    """Specialist adapter for Bi-Temporal Change Detection and Change-VQA."""

    def __init__(self, agent_id: str = "rs-change-agent", agent_version: str = "1.5.0"):
        super().__init__(agent_id=agent_id, agent_version=agent_version)

    def run_inference(
        self,
        query: str,
        batch: PreprocessedBatch,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.model_manager.get_model(self.agent_id)
        raw_pred = model.predict(query=query, batch=batch, parameters=parameters)

        # Generate spatial change map artifact
        change_map_path = EvidenceVisualizer.create_change_map_artifact(
            filename=raw_pred["change_map"],
            changed_regions=raw_pred["changed_regions"]
        )

        return {
            "changed_regions": raw_pred["changed_regions"],
            "change_map": change_map_path,
            "change_description": raw_pred["change_description"],
            "change_trend": raw_pred["change_trend"],
            "changed_area_km2": raw_pred["changed_area_km2"],
            "confidence": raw_pred["confidence"],
            "model": model.model_name,
            "training_adaptation": model.training_dataset,
            "backbone": model.backbone
        }
