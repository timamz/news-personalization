"""Multi-stage digest generation pipeline.

Architecture:
  CandidateFetch (DB query) -> Writer (ADK agent) + Judge loop -> Reflector

The writer plans, optionally researches (fetches articles, searches web),
and composes the digest. The judge scores it and either passes or requests
revision. The reflector reviews pipeline health and self-heals
sources/preferences.
"""

from news_service.agents.digest.pipeline import generate_digest

__all__ = ["generate_digest"]
