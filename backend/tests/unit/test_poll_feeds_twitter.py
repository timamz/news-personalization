import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from news_service.models.rss_feed import RssFeed
from news_service.services.twitter import TwitterPost
from news_service.tasks import poll_feeds


@pytest.mark.asyncio
async def test_poll_single_feed_handles_twitter_account(mocker) -> None:
    feed = RssFeed(
        id=uuid.uuid4(),
        url="https://x.com/openai",
        title="X @openai",
        source_description="OpenAI X account",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )
    posts = [
        TwitterPost(
            url="https://x.com/openai/status/2032045283488473242",
            title="GPT-5.4 is rolling out now in ChatGPT.",
            body="GPT-5.4 is rolling out now in ChatGPT.",
            published_at=datetime.now(UTC) - timedelta(hours=1),
        )
    ]

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=posts))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    count = await poll_feeds._poll_single_feed(session=AsyncMock(), feed=feed)

    assert count == 1
    upsert_news_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_single_feed_skips_stale_twitter_posts(mocker) -> None:
    feed = RssFeed(
        id=uuid.uuid4(),
        url="https://x.com/paperswithcode",
        title="X @paperswithcode",
        source_description="Papers with Code X account",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )
    posts = [
        TwitterPost(
            url="https://x.com/paperswithcode/status/1476544705139724288",
            title="Papers with Code: Year in Review",
            body="Year in Review",
            published_at=datetime(2021, 12, 30, 13, 24, 25, tzinfo=UTC),
        )
    ]

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=posts))
    embed_texts = mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    upsert_news_item = mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    count = await poll_feeds._poll_single_feed(session=AsyncMock(), feed=feed)

    assert count == 0
    embed_texts.assert_not_awaited()
    upsert_news_item.assert_not_awaited()
