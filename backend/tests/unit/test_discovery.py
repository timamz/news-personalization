from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents import discovery
from news_service.agents.discovery import (
    DiscoveredSourceItem,
    DiscoveredSourceList,
)


@pytest.mark.asyncio
async def test_discover_sources_merges_rss_and_telegram_results(mocker) -> None:
    rss_discovery = mocker.patch.object(
        discovery,
        "discover_rss_feeds",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://example.com/rss.xml",
                    topic_tags=["ai"],
                    title="Example RSS",
                    source_kind="rss",
                )
            ]
        ),
    )
    telegram_discovery = mocker.patch.object(
        discovery,
        "discover_telegram_channels",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://t.me/s/ainews",
                    topic_tags=["ai"],
                    title="Telegram @ainews",
                    source_kind="telegram_channel",
                )
            ]
        ),
    )

    result = await discovery.discover_sources(["ai"])

    assert [item.url for item in result] == [
        "https://example.com/rss.xml",
        "https://t.me/s/ainews",
    ]
    rss_discovery.assert_awaited_once_with(["ai"])
    telegram_discovery.assert_awaited_once_with(["ai"])


@pytest.mark.asyncio
async def test_discover_rss_feeds_filters_to_valid_rss_sources(mocker) -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=DiscoveredSourceList(
                        sources=[
                            DiscoveredSourceItem(
                                url=" https://example.com/rss.xml ",
                                topic_tags=["ai"],
                                title="Example RSS",
                                source_kind="rss",
                            ),
                            DiscoveredSourceItem(
                                url="https://t.me/s/not-rss",
                                topic_tags=["ai"],
                                title="Wrong kind",
                                source_kind="telegram_channel",
                            ),
                        ]
                    )
                )
            )
        ]
    )
    mocker.patch.object(
        discovery._client.beta.chat.completions,
        "parse",
        new=AsyncMock(return_value=completion),
    )
    validate_feed = mocker.patch.object(
        discovery,
        "validate_feed_url",
        new=AsyncMock(side_effect=[True]),
    )

    result = await discovery.discover_rss_feeds(["ai"])

    assert result == [
        DiscoveredSourceItem(
            url="https://example.com/rss.xml",
            topic_tags=["ai"],
            title="Example RSS",
            source_kind="rss",
        )
    ]
    validate_feed.assert_awaited_once_with("https://example.com/rss.xml")


@pytest.mark.asyncio
async def test_discover_telegram_channels_normalizes_and_validates_channels(mocker) -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=DiscoveredSourceList(
                        sources=[
                            DiscoveredSourceItem(
                                url="https://t.me/AINews",
                                topic_tags=["ai"],
                                title="AI News",
                                source_kind="telegram_channel",
                            ),
                            DiscoveredSourceItem(
                                url="https://example.com/rss.xml",
                                topic_tags=["ai"],
                                title="Wrong kind",
                                source_kind="rss",
                            ),
                        ]
                    )
                )
            )
        ]
    )
    mocker.patch.object(
        discovery._client.beta.chat.completions,
        "parse",
        new=AsyncMock(return_value=completion),
    )
    validate_channel = mocker.patch.object(
        discovery,
        "validate_telegram_channel",
        new=AsyncMock(return_value=True),
    )

    result = await discovery.discover_telegram_channels(["ai"])

    assert result == [
        DiscoveredSourceItem(
            url="https://t.me/s/ainews",
            topic_tags=["ai"],
            title="AI News",
            source_kind="telegram_channel",
        )
    ]
    validate_channel.assert_awaited_once_with("ainews")
