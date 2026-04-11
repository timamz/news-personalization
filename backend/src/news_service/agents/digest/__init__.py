"""Multi-stage digest generation pipeline.

Architecture:
  CandidateFetch (DB query) -> Planner -> LoopAgent([Composer, Judge]) -> Reflector

The planner creates a digest outline from user_spec and candidates.
The composer writes the digest. The judge scores it and either passes
or requests revision. The reflector reviews pipeline health and
self-heals sources/preferences.
"""

from news_service.agents.digest.pipeline import generate_digest

__all__ = ["generate_digest"]
