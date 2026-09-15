"""
Agent Scoring and Ranking Engine for SatQuery AI Router.
Step 8 of Router workflow:
Scores qualified candidate agents using a weighted multi-criteria function:
1. Capability match score (35%)
2. Remote sensing domain adaptation & benchmark alignment (25%)
3. Modality alignment score (20%)
4. Operational health score (10%)
5. Latency & performance score (10%)
"""

from typing import List
from schemas.agent_schema import AgentResponse, AgentStatus
from satquery_routing.models import RoutingRequirements, CandidateScore


class AgentScorer:
    """
    Ranks qualified agents, prioritizing remote-sensing specialized models
    adapted to BigEarthNet, VRSBench, RSVQA, CDVQA, and ISRO/SAC.
    """

    # Weights for multi-criteria scoring
    W_CAPABILITY = 0.35
    W_ADAPTATION = 0.25
    W_MODALITY = 0.20
    W_HEALTH = 0.10
    W_LATENCY = 0.10

    @classmethod
    def score_agent(cls, agent: AgentResponse, req: RoutingRequirements) -> CandidateScore:
        # 1. Capability Score
        agent_caps = [c.value.lower() for c in agent.capabilities]
        primary_cap = req.primary_capability.lower()
        if primary_cap in agent_caps:
            cap_score = 1.0
        elif any(sc.lower() in agent_caps for sc in req.secondary_capabilities):
            cap_score = 0.70
        else:
            cap_score = 0.40

        # 2. Modality Score
        agent_mods = [m.value.lower() for m in agent.input_modalities]
        req_mod = req.required_modality.lower()
        if req_mod in agent_mods:
            mod_score = 1.0
        else:
            mod_score = 0.50

        # 3. Remote Sensing Domain Adaptation Score
        meta = agent.metadata or {}
        training_set = str(meta.get("training_dataset", "")).lower()
        backbone = str(meta.get("backbone", "")).lower()
        institution = str(meta.get("institution", "")).lower()

        # Benchmarks: BigEarthNet, VRSBench, RSVQA, CDVQA, ISRO/SAC
        rs_keywords = ["bigearthnet", "vrsbench", "rsvqa", "cdvqa", "isro", "sac", "levir", "remoteclip", "siamese"]
        adaptation_matches = sum(1 for kw in rs_keywords if kw in training_set or kw in backbone or kw in institution)

        if adaptation_matches >= 2:
            adapt_score = 1.0
        elif adaptation_matches == 1:
            adapt_score = 0.85
        else:
            adapt_score = 0.50  # Generic model penalty

        # Benchmark reference match boost
        if req.benchmark_reference and req.benchmark_reference.lower() in (training_set + backbone + institution):
            adapt_score = min(1.0, adapt_score + 0.10)

        # Cloud resilience boost for cross-modal queries
        if req.requires_cloud_resilience and meta.get("cloud_resilience", False):
            adapt_score = min(1.0, adapt_score + 0.05)

        # 4. Health Score
        if agent.status == AgentStatus.ACTIVE:
            health_score = 1.0
        elif agent.status == AgentStatus.DEGRADED:
            health_score = 0.60
        else:
            health_score = 0.10

        # 5. Latency Score
        latency_ms = meta.get("latency_ms", 100)
        max_lat = max(req.max_acceptable_latency_ms, 500)
        lat_ratio = max(0.0, min(1.0, latency_ms / max_lat))
        latency_score = round(1.0 - (0.5 * lat_ratio), 4)

        # Composite Score Calculation
        composite = (
            cls.W_CAPABILITY * cap_score +
            cls.W_ADAPTATION * adapt_score +
            cls.W_MODALITY * mod_score +
            cls.W_HEALTH * health_score +
            cls.W_LATENCY * latency_score
        )
        composite = round(min(1.0, max(0.0, composite)), 4)

        return CandidateScore(
            agent_id=agent.agent_id,
            name=agent.name,
            capability_score=cap_score,
            modality_score=mod_score,
            adaptation_score=adapt_score,
            latency_score=latency_score,
            health_score=health_score,
            composite_score=composite,
            rejection_reason=None
        )

    @classmethod
    def rank_candidates(
        cls,
        qualified_agents: List[AgentResponse],
        req: RoutingRequirements
    ) -> List[CandidateScore]:
        """
        Scores and ranks all qualified agents in descending order of composite score.
        """
        scores = [cls.score_agent(agent, req) for agent in qualified_agents]
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores
