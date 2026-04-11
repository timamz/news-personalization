"""Tests for the multi-agent source discovery pipeline."""

import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.agents.source_discovery import (
    DiscoveryPlan,
    ScoredSource,
    SourceDiscoveryResult,
)
from news_service.agents.source_discovery.aggregator import aggregate_sources

logging.disable(logging.CRITICAL)


def test_scored_source_model_accepts_valid_data() -> None:
    source = ScoredSource(
        url=f"https://{uuid.uuid4().hex[:8]}.com/feed",
        title=f"Источник-{uuid.uuid4().hex[:6]}",
        source_kind="rss",
        relevance_score=0.85,
    )
    assert source.relevance_score == 0.85, "ScoredSource did not preserve relevance score"
    assert source.source_kind == "rss", "ScoredSource did not preserve source kind"


def test_source_discovery_result_model_accepts_source_list() -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            title="Тест",
            source_kind="telegram_channel",
            relevance_score=0.72,
        )
    ]
    result = SourceDiscoveryResult(sources=sources)
    assert len(result.sources) == 1, "SourceDiscoveryResult did not preserve sources list"


def test_discovery_plan_model_accepts_strategies() -> None:
    plan = DiscoveryPlan(
        strategies=[
            f"Find arxiv RSS feeds about ML {uuid.uuid4().hex[:4]}",
            f"Find Reddit subreddits about AI {uuid.uuid4().hex[:4]}",
        ]
    )
    assert len(plan.strategies) == 2, "DiscoveryPlan did not preserve strategies"


def test_discovery_plan_rejects_empty_strategies() -> None:
    with pytest.raises(ValueError):
        DiscoveryPlan(strategies=[])


def test_aggregator_deduplicates_by_url() -> None:
    url = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    source_a = ScoredSource(url=url, source_kind="rss", relevance_score=0.8)
    source_b = ScoredSource(url=url, source_kind="rss", relevance_score=0.9)

    result = aggregate_sources([[source_a], [source_b]], max_sources=10)

    assert len(result.sources) == 1, "aggregator did not deduplicate sources with same URL"


def test_aggregator_deduplicates_urls_case_insensitive() -> None:
    base = f"https://{uuid.uuid4().hex[:8]}.com/Feed"
    source_a = ScoredSource(url=base, source_kind="rss", relevance_score=0.8)
    source_b = ScoredSource(url=base.lower(), source_kind="rss", relevance_score=0.9)

    result = aggregate_sources([[source_a], [source_b]], max_sources=10)

    assert len(result.sources) == 1, "aggregator did not deduplicate case-different URLs"


def test_aggregator_sorts_by_relevance_descending() -> None:
    low = ScoredSource(
        url=f"https://{uuid.uuid4().hex[:8]}.com", source_kind="rss", relevance_score=0.3
    )
    high = ScoredSource(
        url=f"https://{uuid.uuid4().hex[:8]}.com", source_kind="rss", relevance_score=0.9
    )
    mid = ScoredSource(
        url=f"https://{uuid.uuid4().hex[:8]}.com", source_kind="rss", relevance_score=0.6
    )

    result = aggregate_sources([[low, mid], [high]], max_sources=10)

    scores = [s.relevance_score for s in result.sources]
    assert scores == sorted(scores, reverse=True), (
        "aggregator did not sort sources by relevance descending"
    )


def test_aggregator_respects_max_sources() -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="rss",
            relevance_score=0.5 + i * 0.1,
        )
        for i in range(10)
    ]

    result = aggregate_sources([sources], max_sources=3)

    assert len(result.sources) == 3, "aggregator did not respect max_sources limit"


def test_aggregator_returns_empty_for_no_results() -> None:
    result = aggregate_sources([], max_sources=5)
    assert len(result.sources) == 0, "aggregator did not return empty for no finder results"


@pytest.mark.asyncio
async def test_pipeline_calls_orchestrator_and_finders(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.plan_discovery",
        new=AsyncMock(
            return_value=DiscoveryPlan(strategies=["Find RSS feeds", "Find Reddit subs"])
        ),
    )

    source_url = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(
            return_value=[ScoredSource(url=source_url, source_kind="rss", relevance_score=0.85)]
        ),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        raw_prompt=f"AI research {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) > 0, "pipeline returned no sources"
    assert result.sources[0].url == source_url, "pipeline did not return expected source URL"


@pytest.mark.asyncio
async def test_pipeline_handles_finder_failure_gracefully(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.plan_discovery",
        new=AsyncMock(return_value=DiscoveryPlan(strategies=["Strategy A", "Strategy B"])),
    )

    source_url = f"https://{uuid.uuid4().hex[:8]}.com"
    call_count = 0

    async def _mock_finder(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Finder A failed")
        return [ScoredSource(url=source_url, source_kind="rss", relevance_score=0.7)]

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        raw_prompt=f"Технологии {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) == 1, "pipeline did not return sources from surviving finder"
