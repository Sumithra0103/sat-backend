"""
RS-VQA Specialist Agent Adapter for SatQuery AI Execution Subsystem.
Adapted on BigEarthNet.txt and RSVQA benchmarks.
"""

from typing import Dict, Any
from execution.adapters.base_adapter import BaseAgentAdapter
from execution.preprocessing.preprocessor import PreprocessedBatch


class RSVQAAdapter(BaseAgentAdapter):
    """Specialist adapter for Remote Sensing Visual Question Answering."""

    def __init__(self, agent_id: str = "rs-vqa-agent", agent_version: str = "1.2.0"):
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
            "answer": raw_pred["answer"],
            "confidence": raw_pred["confidence"],
            "model": model.model_name,
            "training_adaptation": model.training_dataset,
            "backbone": model.backbone,
            "evaluated_channels": raw_pred["spectral_bands_evaluated"]
        }
