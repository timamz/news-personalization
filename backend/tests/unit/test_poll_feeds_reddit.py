import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from news_service.models.rss_feed import RssFeed
from news_service.services.reddit import RedditPost
from news_service.tasks import poll_feeds


@pytest.mark.asyncio
async def test_poll_single_feed_handles_reddit_subreddit(mocker) -> None:
    feed = RssFeed(
        id=uuid.uuid4(),
        url="https://www.reddit.com/r/badminton/new/",
        title="Reddit r/badminton",
        source_description="Badminton subreddit",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )
    posts = [
        RedditPost(
            url="https://www.reddit.com/r/badminton/comments/abc123/thread/",
            title="Swiss Open discussion",
            body="What did you think about the final?",
            published_at=datetime(2026, 3, 12, 10, 47, 32, tzinfo=UTC),
        )
    ]

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=posts))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    count = await poll_feeds._poll_single_feed(session=AsyncMock(), feed=feed)

    assert count == 1
    upsert_news_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_single_feed_skips_stale_reddit_posts(mocker) -> None:
    feed = RssFeed(
        id=uuid.uuid4(),
        url="https://www.reddit.com/r/arxiv/new/",
        title="Reddit r/arxiv",
        source_description="Research subreddit",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )
    posts = [
        RedditPost(
            url="https://www.reddit.com/r/arxiv/comments/1fx4qn6/need_endorsment_on_csai/",
            title="Need Endorsment on CS.AI",
            body="Can someone endorse my paper?",
            published_at=datetime(2024, 10, 6, 0, 46, 35, tzinfo=UTC),
        )
    ]

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=posts))
    embed_texts = mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    upsert_news_item = mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    count = await poll_feeds._poll_single_feed(session=AsyncMock(), feed=feed)

    assert count == 0
    embed_texts.assert_not_awaited()
    upsert_news_item.assert_not_awaited()
