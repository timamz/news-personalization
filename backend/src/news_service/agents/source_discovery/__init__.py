"""Single-agent source discovery pipeline.

Architecture:
  Discovery Agent (looped ADK) -> N parallel GenericFinders (act-mode)

The discovery agent analyzes the user's topic, runs parallel finders via
run_parallel_search(), reviews results, optionally refines, and finalizes.
Each finder executes one strategy using search + validation tools.
Deduplication and ranking happen inside the agent's tool.
"""

from news_service.agents.source_discovery.models import (
    ScoredSource,
    SourceDiscoveryResult,
)
from news_service.agents.source_discovery.pipeline import run_source_discovery

__all__ = [
    "ScoredSource",
    "SourceDiscoveryResult",
    "run_source_discovery",
]
