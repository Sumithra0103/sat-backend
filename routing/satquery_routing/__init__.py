"""
SatQuery AI Agentic Routing Package.
Combines Query Understanding and Agent Registry to generate auditable execution plans for the Orchestrator.
"""

from satquery_routing.router import SatQueryRouter
from satquery_routing.models import (
    ExecutionPlan,
    ExecutionStep,
    ExecutionStepType,
    FallbackRoute,
    CandidateScore,
    RoutingRequirements,
    AuditableExecutionSummary,
    RoutingStatus,
    RouterExecutionTraceItem
)
from satquery_routing.cache import LocalRegistryCache

__all__ = [
    "SatQueryRouter",
    "ExecutionPlan",
    "ExecutionStep",
    "ExecutionStepType",
    "FallbackRoute",
    "CandidateScore",
    "RoutingRequirements",
    "AuditableExecutionSummary",
    "RoutingStatus",
    "RouterExecutionTraceItem",
    "LocalRegistryCache"
]
