"""
Candidate Filtering and Validation Engine for SatQuery AI Router.
Steps 6 & 7 of Router workflow:
Filters candidate agents retrieved from the Registry based on:
1. Capability matching (primary vs secondary)
2. Operational health status (Active vs Degraded vs Inactive)
3. Modality compatibility (Single Optical/SAR vs Bi-temporal vs Cross-modal)
4. Format compatibility (GeoTIFF/TIFF vs PNG/JPEG)
5. Latency and resource limits
"""

from typing import List, Tuple, Optional
from schemas.agent_schema import AgentResponse, AgentStatus
from satquery_routing.models import RoutingRequirements, CandidateScore


class CandidateFilter:
    """
    Applies multi-stage filtering on candidate agents.
    Separates qualifying candidates from disqualified ones, recording clear audit reasons.
    """

    @staticmethod
    def evaluate_candidate(
        agent: AgentResponse,
        req: RoutingRequirements
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates a single agent against requirements.
        Returns (is_passed, rejection_reason).
        """
        # 1. Health Status Filter
        if agent.status in (AgentStatus.INACTIVE, AgentStatus.MAINTENANCE):
            return False, f"Agent status is '{agent.status.value}' (unhealthy/maintenance)."

        # 2. Modality Filter
        agent_modalities = [m.value.lower() for m in agent.input_modalities]
        req_mod = req.required_modality.lower()
        if req_mod not in agent_modalities:
            # Special case: single optical image query can be handled by an agent that supports optical
            return False, f"Modality mismatch: agent supports {agent_modalities}, requires '{req_mod}'."

        # 3. Capability Filter
        agent_caps = [c.value.lower() for c in agent.capabilities]
        primary_cap = req.primary_capability.lower()
        sec_caps = [c.lower() for c in req.secondary_capabilities]

        has_primary = primary_cap in agent_caps
        has_secondary = any(sc in agent_caps for sc in sec_caps)

        if not (has_primary or has_secondary):
            return False, f"Capability mismatch: agent has {agent_caps}, needs '{primary_cap}'."

        # 4. Format Compatibility Filter
        agent_formats = [f.value.lower() for f in agent.supported_formats]
        format_match = any(fmt.lower() in agent_formats for fmt in req.supported_formats)
        if not format_match:
            return False, f"Format mismatch: agent supports {agent_formats}, input has {req.supported_formats}."

        # 5. Latency threshold check
        agent_latency = agent.metadata.get("latency_ms", 100)
        if agent_latency > req.max_acceptable_latency_ms:
            return False, f"Latency {agent_latency}ms exceeds max acceptable {req.max_acceptable_latency_ms}ms."

        return True, None

    @classmethod
    def filter_candidates(
        cls,
        candidates: List[AgentResponse],
        req: RoutingRequirements
    ) -> Tuple[List[AgentResponse], List[CandidateScore]]:
        """
        Partitions candidate agents into qualified agents and disqualified candidate scores with reasons.
        """
        qualified: List[AgentResponse] = []
        evaluations: List[CandidateScore] = []

        for agent in candidates:
            passed, reason = cls.evaluate_candidate(agent, req)
            if passed:
                qualified.append(agent)
            else:
                evaluations.append(
                    CandidateScore(
                        agent_id=agent.agent_id,
                        name=agent.name,
                        capability_score=0.0,
                        modality_score=0.0,
                        adaptation_score=0.0,
                        latency_score=0.0,
                        health_score=0.0,
                        composite_score=0.0,
                        rejection_reason=reason
                    )
                )

        return qualified, evaluations
