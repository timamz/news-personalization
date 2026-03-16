import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.discovery import DiscoveredSourceItem
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


def _make_discovered(
    url: str = "https://example.com/rss.xml",
    title: str = "Example Feed",
    source_kind: str = "rss",
) -> DiscoveredSourceItem:
    return DiscoveredSourceItem(url=url, title=title, source_kind=source_kind)


@pytest.fixture(autouse=True)
def _patch_feed_source_kind(mocker):
    mocker.patch.object(coverage, "_feed_source_kind", return_value="rss")


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_ranks_by_score(mocker) -> None:
    """Top-K candidates by content relevance score are selected."""
    session = AsyncMock()
    feed_a = _make_db_feed("https://a.com/feed")
    feed_b = _make_db_feed("https://b.com/feed")

    mocker.patch.object(
        coverage, "find_similar_feeds", new=AsyncMock(return_value=[feed_a, feed_b])
    )
    mocker.patch.object(coverage, "_discover_all", new=AsyncMock(return_value=[]))
    # feed_b scores higher
    mocker.patch.object(
        coverage,
        "score_candidate",
        new=AsyncMock(side_effect=[(0.3, ["text"]), (0.8, ["text"])]),
    )
    mocker.patch.object(coverage, "_ensure_feed_profile", new=AsyncMock())

    mocker.patch.object(coverage.settings, "source_target_count", 1)

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert len(result) == 1
    assert result[0] == feed_b
    assert feed_b.subscriber_count == 1
    assert feed_a.subscriber_count == 0


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_merges_db_and_web(mocker) -> None:
    """DB and web candidates are merged, deduped, and ranked together."""
    session = AsyncMock()
    db_feed = _make_db_feed("https://existing.com/feed")

    web_item = _make_discovered("https://new.com/feed", "New Feed", "rss")
    registered_feed = _make_db_feed("https://new.com/feed")

    mocker.patch.object(coverage, "find_similar_feeds", new=AsyncMock(return_value=[db_feed]))
    mocker.patch.object(coverage, "_discover_all", new=AsyncMock(return_value=[web_item]))
    # web candidate scores higher
    mocker.patch.object(
        coverage,
        "score_candidate",
        new=AsyncMock(side_effect=[(0.2, ["text"]), (0.9, ["text"])]),
    )
    mocker.patch.object(coverage, "_ensure_feed_profile", new=AsyncMock())
    mocker.patch.object(coverage, "_register_feed", new=AsyncMock(return_value=registered_feed))
    mocker.patch.object(coverage.settings, "source_target_count", 2)

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert len(result) == 2
    # Higher-scored web candidate should be first
    assert result[0] == registered_feed
    assert result[1] == db_feed


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_deduplicates_by_url(mocker) -> None:
    """Same URL from DB and web discovery is not scored twice."""
    session = AsyncMock()
    db_feed = _make_db_feed("https://example.com/feed")
    web_item = _make_discovered("https://example.com/feed")

    mocker.patch.object(coverage, "find_similar_feeds", new=AsyncMock(return_value=[db_feed]))
    mocker.patch.object(coverage, "_discover_all", new=AsyncMock(return_value=[web_item]))
    score_mock = mocker.patch.object(
        coverage, "score_candidate", new=AsyncMock(return_value=(0.5, ["text"]))
    )
    mocker.patch.object(coverage, "_ensure_feed_profile", new=AsyncMock())
    mocker.patch.object(coverage.settings, "source_target_count", 8)

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert len(result) == 1
    # Only scored once despite appearing in both DB and web
    score_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_handles_no_candidates(mocker) -> None:
    """Returns empty list when no candidates found."""
    session = AsyncMock()
    mocker.patch.object(coverage, "find_similar_feeds", new=AsyncMock(return_value=[]))
    mocker.patch.object(coverage, "_discover_all", new=AsyncMock(return_value=[]))

    result = await coverage.ensure_prompt_coverage(session, "test", [0.1])

    assert result == []
