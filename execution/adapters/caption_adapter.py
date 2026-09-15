"""
RS-Caption Specialist Agent Adapter for SatQuery AI Execution Subsystem.
Adapted on BigEarthNet-MM dataset.
"""

from typing import Dict, Any
from execution.adapters.base_adapter import BaseAgentAdapter
from execution.preprocessing.preprocessor import PreprocessedBatch


class RSCaptionAdapter(BaseAgentAdapter):
    """Specialist adapter for Scene Captioning & Land-cover description."""

    def __init__(self, agent_id: str = "rs-caption-agent", agent_version: str = "1.0.0"):
        super().__init__(agent_id=agent_id, agent_version=agent_version)

    def run_inference(
        self,
        query: str,
        batch: PreprocessedBatch,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.model_manager.get_model(self.agent_id)
        raw_pred = model.predict(query=query, batch=batch, parameters=parameters)
        return {
            "caption": raw_pred["scene_description"],
            "landcover_breakdown": raw_pred["landcover_distribution_percent"],
            "detail_level": raw_pred["detail_level"],
            "confidence": raw_pred["confidence"],
            "model": model.model_name,
            "training_adaptation": model.training_dataset,
            "backbone": model.backbone
        }
