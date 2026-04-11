"""Multi-agent source discovery pipeline.

Architecture:
  Orchestrator (plan-mode) -> N parallel GenericFinders (act-mode) -> Aggregator

The orchestrator analyzes the user's topic and produces search strategies.
Each finder executes one strategy using search + validation tools.
The aggregator merges, deduplicates, ranks, and returns final sources.
"""

from news_service.agents.source_discovery.models import (
    DiscoveryPlan,
    ScoredSource,
    SourceDiscoveryResult,
)
from news_service.agents.source_discovery.pipeline import run_source_discovery

__all__ = [
    "DiscoveryPlan",
    "ScoredSource",
    "SourceDiscoveryResult",
    "run_source_discovery",
]
