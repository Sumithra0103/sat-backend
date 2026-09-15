"""
RS-CrossModal Specialist Agent Adapter for SatQuery AI Execution Subsystem.
Adapted on BigEarthNet-MM and ISRO/SAC Cartosat-2S + RISAT SAR benchmark pairs.
"""

from typing import Dict, Any
from execution.adapters.base_adapter import BaseAgentAdapter
from execution.preprocessing.preprocessor import PreprocessedBatch


class RSCrossModalAdapter(BaseAgentAdapter):
    """Specialist adapter for Optical-SAR Cross-Modal Joint Analysis."""

    def __init__(self, agent_id: str = "rs-crossmodal-agent", agent_version: str = "1.1.0"):
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
            "joint_analysis": raw_pred["joint_analysis_summary"],
            "classified_features": raw_pred["identified_classes"],
            "cloud_masking_applied": raw_pred["cloud_masking_applied"],
            "fusion_method": raw_pred["fusion_method"],
            "confidence": raw_pred["confidence"],
            "model": model.model_name,
            "training_adaptation": model.training_dataset,
            "backbone": model.backbone
        }
