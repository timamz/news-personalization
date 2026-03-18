import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.source_discovery import ScoredSource, SourceDiscoveryResult
from news_service.services import coverage


def _make_db_feed(url: str = "https://example.com/rss.xml") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        url=url,
        title="Example Feed",
        subscriber_count=0,
        is_active=True,
        source_description="test",
        source_description_embedding=[0.1] * 10,
    )


def _make_scored_source(
    url: str = "https://example.com/feed",
    title: str = "Example Feed",
    source_kind: str = "rss",
    score: float = 0.8,
) -> ScoredSource:
    return ScoredSource(url=url, title=title, source_kind=source_kind, relevance_score=score)


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_registers_agent_results(mocker) -> None:
    """Agent-returned sources are registered in DB via _register_or_reuse_source."""
    session = AsyncMock()
    source_a = _make_scored_source("https://a.com/feed", score=0.9)
    source_b = _make_scored_source("https://b.com/feed", score=0.7)
    agent_result = SourceDiscoveryResult(sources=[source_a, source_b])

    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(return_value=agent_result),
    )

    feed_a = _make_db_feed("https://a.com/feed")
    feed_b = _make_db_feed("https://b.com/feed")
    register_mock = AsyncMock(side_effect=[feed_a, feed_b])
    mocker.patch.object(coverage, "_register_or_reuse_source", register_mock)

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert len(result) == 2
    assert result[0] == feed_a
    assert result[1] == feed_b
    assert register_mock.await_count == 2


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_returns_empty_on_agent_failure(mocker) -> None:
    """Returns empty list when the discovery agent raises."""
    session = AsyncMock()
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(side_effect=RuntimeError("agent failed")),
    )

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert result == []


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_returns_empty_on_no_sources(mocker) -> None:
    """Returns empty list when agent finds no sources."""
    session = AsyncMock()
    agent_result = SourceDiscoveryResult(sources=[])
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(return_value=agent_result),
    )

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert result == []


@pytest.mark.asyncio
async def test_register_or_reuse_source_reuses_existing_feed(mocker) -> None:
    """Existing feed is reused with incremented subscriber count."""
    session = AsyncMock()
    existing = _make_db_feed("https://example.com/feed")
    existing.subscriber_count = 2

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing
    session.execute = AsyncMock(return_value=mock_result)
    mocker.patch.object(coverage, "_ensure_feed_profile", new=AsyncMock())

    source = _make_scored_source("https://example.com/feed")
    feed = await coverage._register_or_reuse_source(session, source)

    assert feed == existing
    assert existing.subscriber_count == 3
    assert existing.is_active is True


@pytest.mark.asyncio
async def test_register_or_reuse_source_creates_new_feed(mocker) -> None:
    """New feed is created when URL not in DB."""
    session = AsyncMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()

    mocker.patch.object(
        coverage, "_build_feed_profile", new=AsyncMock(return_value=("desc", [0.1] * 10))
    )

    source = _make_scored_source("https://new.com/feed", title="New Feed")
    feed = await coverage._register_or_reuse_source(session, source)

    assert feed is not None
    session.add.assert_called_once()
    session.flush.assert_awaited_once()
