"""
Adapter Factory for SatQuery AI Execution Subsystem.
Resolves and instantiates specialist agent adapters from agent ID and registry specs.
"""

from typing import Dict, Any, Optional
from execution.adapters.base_adapter import BaseAgentAdapter
from execution.adapters.vqa_adapter import RSVQAAdapter
from execution.adapters.caption_adapter import RSCaptionAdapter
from execution.adapters.grounding_adapter import RSGroundingAdapter
from execution.adapters.change_adapter import RSChangeAdapter
from execution.adapters.crossmodal_adapter import RSCrossModalAdapter


class AdapterFactory:
    """Factory creating specialist agent adapters."""

    @classmethod
    def get_adapter(
        cls,
        agent_id: str,
        agent_spec: Optional[Dict[str, Any]] = None
    ) -> BaseAgentAdapter:
        aid = agent_id.lower()
        version = agent_spec.get("version", "1.0.0") if agent_spec else "1.0.0"

        if "vqa" in aid:
            return RSVQAAdapter(agent_id=agent_id, agent_version=version)
        elif "caption" in aid:
            return RSCaptionAdapter(agent_id=agent_id, agent_version=version)
        elif "grounding" in aid:
            return RSGroundingAdapter(agent_id=agent_id, agent_version=version)
        elif "change" in aid:
            return RSChangeAdapter(agent_id=agent_id, agent_version=version)
        elif "crossmodal" in aid or "fusion" in aid:
            return RSCrossModalAdapter(agent_id=agent_id, agent_version=version)
        else:
            return RSVQAAdapter(agent_id=agent_id, agent_version=version)
