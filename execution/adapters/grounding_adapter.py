"""
RS-Grounding Specialist Agent Adapter for SatQuery AI Execution Subsystem.
Adapted on VRSBench benchmark for text-guided bounding boxes, masks, and GeoJSON.
"""

from typing import Dict, Any
from execution.adapters.base_adapter import BaseAgentAdapter
from execution.preprocessing.preprocessor import PreprocessedBatch
from execution.evidence.visualizer import EvidenceVisualizer


class RSGroundingAdapter(BaseAgentAdapter):
    """Specialist adapter for Text-Guided Region Grounding."""

    def __init__(self, agent_id: str = "rs-grounding-agent", agent_version: str = "2.0.1"):
        super().__init__(agent_id=agent_id, agent_version=agent_version)

    def run_inference(
        self,
        query: str,
        batch: PreprocessedBatch,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.model_manager.get_model(self.agent_id)
        raw_pred = model.predict(query=query, batch=batch, parameters=parameters)

        # Export GeoJSON feature to evidence directory
        geojson_path = EvidenceVisualizer.export_geojson(
            filename=f"grounded_{raw_pred['grounded_entity']}_region.geojson",
            feature_dict=raw_pred["geojson_feature"]
        )

        return {
            "grounded_entity": raw_pred["grounded_entity"],
            "bounding_box_normalized": raw_pred["bounding_box_normalized"],
            "geojson_feature": raw_pred["geojson_feature"],
            "geojson_file": geojson_path,
            "label": raw_pred["label"],
            "confidence": raw_pred["iou_confidence"],
            "iou_confidence": raw_pred["iou_confidence"],
            "model": model.model_name,
            "training_adaptation": model.training_dataset,
            "backbone": model.backbone
        }
