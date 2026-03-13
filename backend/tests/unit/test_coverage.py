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
                    title="Example Feed",
                    source_kind="rss",
                ),
                DiscoveredSourceItem(
                    url="https://example.com/rss.xml",
                    title="Example Feed Duplicate",
                    source_kind="rss",
                ),
            ]
        ),
    )
    mocker.patch.object(coverage, "discover_telegram_channels", new=AsyncMock(return_value=[]))
    mocker.patch.object(coverage, "discover_reddit_subreddits", new=AsyncMock(return_value=[]))
    mocker.patch.object(coverage, "discover_twitter_accounts", new=AsyncMock(return_value=[]))
    registered_feed = SimpleNamespace(id=uuid.uuid4(), url="https://example.com/rss.xml")
    register_feed = mocker.patch.object(
        coverage,
        "_register_feed",
        new=AsyncMock(return_value=registered_feed),
    )

    result = await coverage.ensure_prompt_coverage(session, "AI updates", [0.1])

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
                    title="Example Feed",
                    source_kind="rss",
                )
            ]
        ),
    )
    mocker.patch.object(coverage, "discover_telegram_channels", new=AsyncMock(return_value=[]))
    mocker.patch.object(coverage, "discover_reddit_subreddits", new=AsyncMock(return_value=[]))
    mocker.patch.object(coverage, "discover_twitter_accounts", new=AsyncMock(return_value=[]))
    register_feed = mocker.patch.object(coverage, "_register_feed", new=AsyncMock())

    result = await coverage.ensure_prompt_coverage(session, "AI science updates", [0.1])

    assert result == [existing_feed]
    assert existing_feed.subscriber_count == 1
    find_similar_feeds.assert_awaited_once_with(session, [0.1], limit=3)
    register_feed.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_topic_coverage_merges_rss_telegram_and_reddit_discovery(mocker) -> None:
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
                    title="Telegram @badmintonnews",
                    source_kind="telegram_channel",
                )
            ]
        ),
    )
    mocker.patch.object(
        coverage,
        "discover_reddit_subreddits",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://www.reddit.com/r/badminton/new/",
                    title="Reddit r/badminton",
                    source_kind="reddit_subreddit",
                )
            ]
        ),
    )
    mocker.patch.object(
        coverage,
        "discover_twitter_accounts",
        new=AsyncMock(
            return_value=[
                DiscoveredSourceItem(
                    url="https://x.com/openai",
                    title="X @openai",
                    source_kind="twitter_account",
                )
            ]
        ),
    )
    rss_feed = SimpleNamespace(id=uuid.uuid4(), url="https://example.com/rss.xml")
    telegram_feed = SimpleNamespace(id=uuid.uuid4(), url="https://t.me/s/badmintonnews")
    reddit_feed = SimpleNamespace(id=uuid.uuid4(), url="https://www.reddit.com/r/badminton/new/")
    twitter_feed = SimpleNamespace(id=uuid.uuid4(), url="https://x.com/openai")
    register_feed = mocker.patch.object(
        coverage,
        "_register_feed",
        new=AsyncMock(side_effect=[rss_feed, telegram_feed, reddit_feed, twitter_feed]),
    )

    result = await coverage.ensure_prompt_coverage(session, "badminton news", [0.1])

    assert result == [rss_feed, telegram_feed, reddit_feed, twitter_feed]
    assert register_feed.await_count == 4


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_discovers_all_platforms(mocker) -> None:
    mocker.patch.object(
        coverage,
        "find_similar_feeds",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        coverage,
        "discover_rss_feeds",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        coverage,
        "discover_telegram_channels",
        new=AsyncMock(return_value=[]),
    )
    discover_reddit_subreddits = mocker.patch.object(
        coverage,
        "discover_reddit_subreddits",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        coverage,
        "discover_twitter_accounts",
        new=AsyncMock(return_value=[]),
    )

    result = await coverage.ensure_prompt_coverage(
        session=AsyncMock(),
        raw_prompt="machine learning research papers",
        raw_prompt_embedding=[0.1],
    )

    assert result == []
    discover_reddit_subreddits.assert_awaited_once_with("machine learning research papers")
