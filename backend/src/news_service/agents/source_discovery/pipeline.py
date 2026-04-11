"""Source discovery pipeline: Orchestrator -> N parallel Finders -> Aggregator.

This is the main entry point for source discovery. It:
1. Calls the orchestrator to produce search strategies (plan-mode)
2. Spawns one GenericFinder per strategy in parallel (act-mode, fan-out)
3. Aggregates all results (fan-in), deduplicates, ranks by relevance
"""

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings

from .aggregator import aggregate_sources
from .finder import run_finder
from .models import SourceDiscoveryResult
from .orchestrator import plan_discovery

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_source_discovery(
    *,
    session: AsyncSession,
    raw_prompt: str,
    prompt_embedding: list[float],
    exclude_urls: list[str] | None = None,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> SourceDiscoveryResult:
    """Run the full multi-agent source discovery pipeline.

    Steps:
    1. Orchestrator analyzes the topic and produces 2-5 search strategies
    2. One GenericFinder runs per strategy, all in parallel
    3. Aggregator merges, deduplicates, and ranks results

    Returns a SourceDiscoveryResult with scored, validated sources.
    """
    if exclude_urls is None:
        exclude_urls = []

    if status_queue is not None:
        status_queue.put_nowait({"event": "status", "status_key": "status_planning_discovery"})

    plan = await plan_discovery(raw_prompt)

    logger.info(
        "Discovery plan for '%s': %d strategies",
        raw_prompt[:60],
        len(plan.strategies),
    )

    if status_queue is not None:
        status_queue.put_nowait({"event": "status", "status_key": "status_searching_sources"})

    finder_tasks = [
        run_finder(
            strategy=strategy,
            session=session,
            prompt_embedding=prompt_embedding,
            exclude_urls=exclude_urls,
            status_queue=status_queue,
        )
        for strategy in plan.strategies
    ]

    finder_results = await asyncio.gather(*finder_tasks, return_exceptions=True)

    successful_results: list[list] = []
    for i, result in enumerate(finder_results):
        if isinstance(result, Exception):
            logger.exception(
                "Finder failed for strategy '%s': %s",
                plan.strategies[i][:60],
                result,
            )
        else:
            successful_results.append(result)

    if not successful_results:
        logger.warning("All finders failed for topic: %s", raw_prompt[:60])
        return SourceDiscoveryResult(sources=[])

    return aggregate_sources(
        successful_results,
        max_sources=settings.source_target_count,
    )
