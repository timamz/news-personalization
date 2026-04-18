"""Multi-stage digest generation pipeline.

Architecture:
  CandidateFetch (DB query) -> Writer (ADK agent) + Judge loop -> Reflector

The writer plans, optionally researches (web search only -- items already
carry full article bodies from ingest-time enrichment), and composes the
digest. The judge scores it and either passes or requests revision. The
reflector curates the subscription's source pool when health triggers
fire (drift / staleness / contribution streak / REVISE verdict).
"""

from news_service.agents.digest.pipeline import generate_digest

__all__ = ["generate_digest"]
