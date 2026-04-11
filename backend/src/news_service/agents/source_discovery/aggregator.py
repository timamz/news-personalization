"""Source Aggregator — merges, deduplicates, and ranks results from all finders."""

import logging

from .models import ScoredSource, SourceDiscoveryResult

logger = logging.getLogger(__name__)


def aggregate_sources(
    finder_results: list[list[ScoredSource]],
    *,
    max_sources: int,
) -> SourceDiscoveryResult:
    """Merge results from all finders, deduplicate by URL, rank by score.

    No LLM call — pure data processing.
    """
    seen_urls: set[str] = set()
    merged: list[ScoredSource] = []

    for sources in finder_results:
        for source in sources:
            normalized_url = source.url.rstrip("/").lower()
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            merged.append(source)

    merged.sort(key=lambda s: s.relevance_score, reverse=True)
    selected = merged[:max_sources]

    logger.info(
        "Aggregator: %d total from finders, %d unique, %d selected (top by score)",
        sum(len(r) for r in finder_results),
        len(merged),
        len(selected),
    )

    return SourceDiscoveryResult(sources=selected)
