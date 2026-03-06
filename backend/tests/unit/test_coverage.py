import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.discovery import DiscoveredFeedItem
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
        "discover_feeds",
        new=AsyncMock(
            return_value=[
                DiscoveredFeedItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["ai"],
                    title="Example Feed",
                ),
                DiscoveredFeedItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["ai"],
                    title="Example Feed Duplicate",
                ),
            ]
        ),
    )
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
        "discover_feeds",
        new=AsyncMock(
            return_value=[
                DiscoveredFeedItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["science"],
                    title="Example Feed",
                )
            ]
        ),
    )
    register_feed = mocker.patch.object(coverage, "_register_feed", new=AsyncMock())

    result = await coverage.ensure_topic_coverage(session, ["ai", "science"], [0.1])

    assert result == [existing_feed]
    assert existing_feed.subscriber_count == 1
    find_similar_feeds.assert_awaited_once_with(session, [0.1], limit=3)
    register_feed.assert_not_awaited()
