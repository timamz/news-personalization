import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.discovery import DiscoveredFeedItem
from news_service.services import coverage


@pytest.mark.asyncio
async def test_ensure_topic_coverage_deduplicates_discovered_urls(mocker) -> None:
    mocker.patch.object(coverage, "embed_text", new=AsyncMock(return_value=[0.1]))
    mocker.patch.object(coverage, "find_similar_feeds", new=AsyncMock(return_value=[]))
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

    result = await coverage.ensure_topic_coverage(AsyncMock(), ["ai"])

    assert result == [registered_feed]
    register_feed.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_topic_coverage_skips_discovered_url_already_selected(mocker) -> None:
    existing_feed = SimpleNamespace(
        id=uuid.uuid4(),
        url="https://example.com/rss.xml",
        subscriber_count=0,
    )
    mocker.patch.object(
        coverage,
        "embed_text",
        new=AsyncMock(side_effect=[[0.1], [0.2]]),
    )
    mocker.patch.object(
        coverage,
        "find_similar_feeds",
        new=AsyncMock(side_effect=[[existing_feed], []]),
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

    result = await coverage.ensure_topic_coverage(AsyncMock(), ["ai", "science"])

    assert result == [existing_feed]
    assert existing_feed.subscriber_count == 1
    register_feed.assert_not_awaited()
