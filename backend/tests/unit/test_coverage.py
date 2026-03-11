import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.discovery import DiscoveredSourceItem
from news_service.services import coverage


@pytest.mark.asyncio
async def test_ensure_topic_coverage_deduplicates_discovered_urls(mocker) -> None:
    session = AsyncMock()
    find_similar_feeds = mocker.patch.object(
        coverage,
        "find_similar_feeds",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        coverage,
        "discover_rss_feeds",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["ai"],
                    title="Example Feed",
                    source_kind="rss",
                ),
                DiscoveredSourceItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["ai"],
                    title="Example Feed Duplicate",
                    source_kind="rss",
                ),
            ]
        ),
    )
    mocker.patch.object(coverage, "discover_telegram_channels", new=AsyncMock(return_value=[]))
    registered_feed = SimpleNamespace(id=uuid.uuid4(), url="https://example.com/rss.xml")
    register_feed = mocker.patch.object(
        coverage,
        "_register_feed",
        new=AsyncMock(return_value=registered_feed),
    )

    result = await coverage.ensure_topic_coverage(session, ["ai"], [0.1])

    assert result == [registered_feed]
    find_similar_feeds.assert_awaited_once_with(session, [0.1], limit=3)
    register_feed.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_topic_coverage_skips_discovered_url_already_selected(mocker) -> None:
    session = AsyncMock()
    existing_feed = SimpleNamespace(
        id=uuid.uuid4(),
        url="https://example.com/rss.xml",
        subscriber_count=0,
    )
    find_similar_feeds = mocker.patch.object(
        coverage,
        "find_similar_feeds",
        new=AsyncMock(return_value=[existing_feed]),
    )
    mocker.patch.object(
        coverage,
        "discover_rss_feeds",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["science"],
                    title="Example Feed",
                    source_kind="rss",
                )
            ]
        ),
    )
    mocker.patch.object(coverage, "discover_telegram_channels", new=AsyncMock(return_value=[]))
    register_feed = mocker.patch.object(coverage, "_register_feed", new=AsyncMock())

    result = await coverage.ensure_topic_coverage(session, ["ai", "science"], [0.1])

    assert result == [existing_feed]
    assert existing_feed.subscriber_count == 1
    find_similar_feeds.assert_awaited_once_with(session, [0.1], limit=3)
    register_feed.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_topic_coverage_merges_rss_and_telegram_discovery(mocker) -> None:
    session = AsyncMock()
    mocker.patch.object(
        coverage,
        "find_similar_feeds",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        coverage,
        "discover_rss_feeds",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["badminton"],
                    title="RSS Feed",
                    source_kind="rss",
                )
            ]
        ),
    )
    mocker.patch.object(
        coverage,
        "discover_telegram_channels",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://t.me/s/badmintonnews",
                    topic_tags=["badminton"],
                    title="Telegram @badmintonnews",
                    source_kind="telegram_channel",
                )
            ]
        ),
    )
    rss_feed = SimpleNamespace(id=uuid.uuid4(), url="https://example.com/rss.xml")
    telegram_feed = SimpleNamespace(id=uuid.uuid4(), url="https://t.me/s/badmintonnews")
    register_feed = mocker.patch.object(
        coverage,
        "_register_feed",
        new=AsyncMock(side_effect=[rss_feed, telegram_feed]),
    )

    result = await coverage.ensure_topic_coverage(session, ["badminton"], [0.1])

    assert result == [rss_feed, telegram_feed]
    assert register_feed.await_count == 2
