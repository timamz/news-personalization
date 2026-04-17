"""Tests for the source discovery pipeline."""

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.agents.source_discovery import ScoredSource
from news_service.agents.source_discovery.pipeline import _deduplicate, _format_summary

logging.disable(logging.CRITICAL)


def _scored(url: str, score: float = 0.8, **kw) -> ScoredSource:
    return ScoredSource(
        url=url, source_kind=kw.pop("source_kind", "rss"), relevance_score=score, **kw
    )


@pytest.mark.parametrize(
    ("url_a", "url_b"),
    [
        ("https://a.test/feed", "https://a.test/feed"),
        ("https://a.test/Feed", "https://a.test/feed"),
        ("https://a.test/feed", "https://a.test/feed/"),
    ],
    ids=["exact", "case_insensitive", "trailing_slash"],
)
def test_deduplicate_treats_variants_of_same_url_as_duplicates(url_a: str, url_b: str) -> None:
    result = _deduplicate([_scored(url_a, 0.8), _scored(url_b, 0.9)])
    assert len(result) == 1, "deduplicate did not collapse duplicate URL variants"


def test_deduplicate_keeps_first_occurrence_and_preserves_distinct_urls() -> None:
    url = f"https://{uuid.uuid4().hex[:8]}.test/feed"
    first = _scored(url, 0.8, title="first")
    second = _scored(url, 0.9, title="second")
    distinct = [
        _scored(f"https://{uuid.uuid4().hex[:8]}.test/{i}", 0.5 + i * 0.1) for i in range(3)
    ]

    result = _deduplicate([first, second, *distinct])
    assert result[0].title == "first" and len(result) == 4, (
        "deduplicate did not keep the first occurrence while preserving distinct URLs"
    )


def test_format_summary_shows_count_and_type_breakdown_or_empty_notice() -> None:
    assert "No sources found" in _format_summary([])
    summary = _format_summary(
        [
            _scored(f"https://{uuid.uuid4().hex[:8]}.test/a", source_kind="rss"),
            _scored(f"https://{uuid.uuid4().hex[:8]}.test/b", source_kind="reddit_subreddit"),
        ]
    )
    assert "Found 2 sources" in summary and "rss" in summary and "reddit_subreddit" in summary


def _fake_adk_runner(strategies: int = 1):
    async def fake(*, agent, message, user_id="system"):
        tools = {t.__name__: t for t in agent.tools}
        for i in range(strategies):
            await tools["run_parallel_search"](f"strategy-{i}")
        await tools["submit_results"]()
        return "Done"

    return AsyncMock(side_effect=fake)


@pytest.mark.asyncio
async def test_pipeline_returns_sources_and_sorts_by_relevance_descending(mocker) -> None:
    sources = [_scored(f"https://{uuid.uuid4().hex[:8]}.test", score=s) for s in (0.3, 0.9, 0.6)]
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=sources),
    )
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=_fake_adk_runner(),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )

    scores = [s.relevance_score for s in result.sources]
    assert scores == sorted(scores, reverse=True) and len(result.sources) == 3


@pytest.mark.asyncio
async def test_pipeline_survives_partial_finder_failure(mocker) -> None:
    surviving_url = f"https://{uuid.uuid4().hex[:8]}.test/feed"
    call = 0

    async def _mock_finder(**_):
        nonlocal call
        call += 1
        if call == 1:
            raise RuntimeError("finder failed")
        return [_scored(surviving_url, 0.7)]

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )

    async def _runner(*, agent, message, user_id="system"):
        tools = {t.__name__: t for t in agent.tools}
        await tools["run_parallel_search"]("Strategy A\nStrategy B")
        await tools["submit_results"]()
        return "Done"

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=AsyncMock(side_effect=_runner),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    assert len(result.sources) == 1, (
        "pipeline did not recover and return the surviving finder's source"
    )


@pytest.mark.asyncio
async def test_pipeline_deduplicates_across_rounds(mocker) -> None:
    shared = f"https://{uuid.uuid4().hex[:8]}.test/feed"
    unique = f"https://{uuid.uuid4().hex[:8]}.test/other"
    round_counter = 0

    async def _mock_finder(**_):
        nonlocal round_counter
        round_counter += 1
        if round_counter <= 1:
            return [_scored(shared, 0.8)]
        return [_scored(shared, 0.8), _scored(unique, 0.6)]

    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=_mock_finder),
    )
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=_fake_adk_runner(strategies=2),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    assert len(result.sources) == 2, "pipeline did not deduplicate shared sources across rounds"


@pytest.mark.asyncio
async def test_pipeline_respects_source_target_count(mocker) -> None:
    sources = [
        _scored(f"https://{uuid.uuid4().hex[:8]}.test", score=0.5 + i * 0.05) for i in range(20)
    ]
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=sources),
    )
    mocker.patch("news_service.agents.source_discovery.pipeline.settings.source_target_count", 3)
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=_fake_adk_runner(),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    assert len(result.sources) == 3


@pytest.mark.asyncio
async def test_pipeline_returns_empty_when_all_finders_fail(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=RuntimeError("total failure")),
    )
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=_fake_adk_runner(),
    )

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    assert result.sources == []


@pytest.mark.asyncio
async def test_pipeline_emits_status_events_through_the_queue(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_agent_text",
        new=_fake_adk_runner(),
    )
    queue: asyncio.Queue[dict] = asyncio.Queue()

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
        status_queue=queue,
    )

    keys = []
    while not queue.empty():
        keys.append(queue.get_nowait()["status_key"])
    assert "status_planning_discovery" in keys and "status_searching_sources" in keys
