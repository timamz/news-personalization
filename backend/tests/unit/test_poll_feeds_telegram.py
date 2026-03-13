import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from news_service.models.rss_feed import RssFeed
from news_service.services.telegram import TelegramPost
from news_service.tasks import poll_feeds


@pytest.mark.asyncio
async def test_poll_single_feed_handles_telegram_channel(mocker) -> None:
    feed = RssFeed(
        id=uuid.uuid4(),
        url="https://t.me/s/fondnauk",
        title="Telegram @fondnauk",
        source_description="Science Telegram channel",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )
    posts = [
        TelegramPost(
            url="https://t.me/fondnauk/1",
            body="First line\nSecond line",
            published_at=datetime(2026, 3, 12, 8, 0, tzinfo=UTC),
        )
    ]

    mocker.patch.object(poll_feeds, "fetch_telegram_posts", new=AsyncMock(return_value=posts))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    count = await poll_feeds._poll_single_feed(session=AsyncMock(), feed=feed)

    assert count == 1
    upsert_news_item.assert_awaited_once()
