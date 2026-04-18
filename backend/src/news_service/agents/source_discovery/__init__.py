"""Single-agent source discovery pipeline.

Architecture:
  Discovery Agent (looped ADK) -> N parallel Source Finders (ReAct mode)

The discovery agent analyzes the user's topic and spawns one finder per
search strategy via the spawn_finder tool (ADK runs them in parallel).
It reviews the deduped scored pool, may inspect candidates or spawn more
finders, and finalizes via submit_selection or abort. Each finder runs
its own ReAct loop with search + validation tools. Deduplication and
ranking happen inside the orchestrator's tool layer.
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
