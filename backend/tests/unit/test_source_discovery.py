"""Tests for the single-agent source discovery pipeline."""

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.agents.source_discovery import (
    ScoredSource,
    SourceDiscoveryResult,
)
from news_service.agents.source_discovery.pipeline import (
    _deduplicate,
    _format_summary,
)

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


def test_deduplicate_removes_sources_with_same_url() -> None:
    url = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    source_a = ScoredSource(url=url, source_kind="rss", relevance_score=0.8)
    source_b = ScoredSource(url=url, source_kind="rss", relevance_score=0.9)

    result = _deduplicate([source_a, source_b])

    assert len(result) == 1, "deduplicate did not remove sources with same URL"


def test_deduplicate_is_case_insensitive() -> None:
    base = f"https://{uuid.uuid4().hex[:8]}.com/Feed"
    source_a = ScoredSource(url=base, source_kind="rss", relevance_score=0.8)
    source_b = ScoredSource(url=base.lower(), source_kind="rss", relevance_score=0.9)

    result = _deduplicate([source_a, source_b])

    assert len(result) == 1, "deduplicate did not treat case-different URLs as duplicates"


def test_deduplicate_normalizes_trailing_slashes() -> None:
    base = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    source_a = ScoredSource(url=base, source_kind="rss", relevance_score=0.8)
    source_b = ScoredSource(url=base + "/", source_kind="rss", relevance_score=0.9)

    result = _deduplicate([source_a, source_b])

    assert len(result) == 1, "deduplicate did not normalize trailing slashes"


def test_deduplicate_keeps_first_occurrence() -> None:
    url = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    first = ScoredSource(url=url, source_kind="rss", relevance_score=0.8, title="first")
    second = ScoredSource(url=url, source_kind="rss", relevance_score=0.9, title="second")

    result = _deduplicate([first, second])

    assert result[0].title == "first", "deduplicate did not keep the first occurrence"


def test_deduplicate_preserves_unique_sources() -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="rss",
            relevance_score=0.5 + i * 0.1,
        )
        for i in range(5)
    ]

    result = _deduplicate(sources)

    assert len(result) == 5, "deduplicate removed unique sources"


def test_deduplicate_returns_empty_for_empty_input() -> None:
    result = _deduplicate([])
    assert len(result) == 0, "deduplicate did not return empty list for empty input"


def test_format_summary_shows_no_sources_for_empty_list() -> None:
    summary = _format_summary([])
    assert "No sources found" in summary, "format_summary did not indicate empty results"


def test_format_summary_includes_source_count() -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="rss",
            relevance_score=0.75,
        ),
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="telegram_channel",
            relevance_score=0.6,
        ),
    ]
    summary = _format_summary(sources)
    assert "Found 2 sources" in summary, "format_summary did not include correct source count"


def test_format_summary_includes_type_breakdown() -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="rss",
            relevance_score=0.75,
        ),
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="reddit_subreddit",
            relevance_score=0.6,
        ),
    ]
    summary = _format_summary(sources)
    assert "rss" in summary, "format_summary did not include rss type"
    assert "reddit_subreddit" in summary, "format_summary did not include reddit type"


@pytest.mark.asyncio
async def test_pipeline_returns_sources_from_finders(mocker) -> None:
    source_url = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    found_sources = [ScoredSource(url=source_url, source_kind="rss", relevance_score=0.85)]

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=found_sources),
    )

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        """Simulate the ADK agent calling run_parallel_search then submit_results."""
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Find RSS feeds about tech")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"AI research {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) > 0, "pipeline returned no sources"
    assert result.sources[0].url == source_url, "pipeline did not return expected source URL"


@pytest.mark.asyncio
async def test_pipeline_handles_finder_failure_gracefully(mocker) -> None:
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

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Strategy A\nStrategy B")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Технологии {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) == 1, "pipeline did not return sources from surviving finder"


@pytest.mark.asyncio
async def test_pipeline_deduplicates_across_rounds(mocker) -> None:
    shared_url = f"https://{uuid.uuid4().hex[:8]}.com/feed"
    unique_url = f"https://{uuid.uuid4().hex[:8]}.com/other"
    round_counter = 0

    async def _mock_finder(**kwargs):
        nonlocal round_counter
        round_counter += 1
        if round_counter <= 1:
            return [ScoredSource(url=shared_url, source_kind="rss", relevance_score=0.8)]
        return [
            ScoredSource(url=shared_url, source_kind="rss", relevance_score=0.8),
            ScoredSource(url=unique_url, source_kind="rss", relevance_score=0.6),
        ]

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Round 1 strategy")
        await tools_by_name["run_parallel_search"]("Round 2 strategy")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"ML papers {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) == 2, "pipeline did not deduplicate across rounds"


@pytest.mark.asyncio
async def test_pipeline_sorts_results_by_relevance_descending(mocker) -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="rss",
            relevance_score=score,
        )
        for score in [0.3, 0.9, 0.6]
    ]

    async def _mock_finder(**kwargs):
        return sources

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Find sources")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Тестовый запрос {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    scores = [s.relevance_score for s in result.sources]
    assert scores == sorted(scores, reverse=True), (
        "pipeline did not sort results by relevance descending"
    )


@pytest.mark.asyncio
async def test_pipeline_respects_source_target_count(mocker) -> None:
    sources = [
        ScoredSource(
            url=f"https://{uuid.uuid4().hex[:8]}.com",
            source_kind="rss",
            relevance_score=0.5 + i * 0.05,
        )
        for i in range(20)
    ]

    async def _mock_finder(**kwargs):
        return sources

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.settings.source_target_count",
        3,
    )

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Find everything")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Narrow topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) == 3, "pipeline did not respect source_target_count limit"


@pytest.mark.asyncio
async def test_pipeline_returns_empty_when_all_finders_fail(mocker) -> None:
    async def _mock_finder(**kwargs):
        raise RuntimeError("Total failure")

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Doomed strategy")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Невозможный запрос {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    assert len(result.sources) == 0, "pipeline returned sources when all finders failed"


@pytest.mark.asyncio
async def test_pipeline_sends_status_events_to_queue(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[]),
    )

    async def _fake_run_agent_text(*, agent, message, user_id="system"):
        tools_by_name = {t.__name__: t for t in agent.tools}
        await tools_by_name["run_parallel_search"]("Strategy 1")
        await tools_by_name["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_fake_run_agent_text),
    )

    status_queue: asyncio.Queue[dict] = asyncio.Queue()

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Запрос со статусами {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
        status_queue=status_queue,
    )

    events = []
    while not status_queue.empty():
        events.append(status_queue.get_nowait())

    status_keys = [e["status_key"] for e in events]
    assert "status_planning_discovery" in status_keys, "pipeline did not emit planning status event"
    assert "status_searching_sources" in status_keys, "pipeline did not emit searching status event"
