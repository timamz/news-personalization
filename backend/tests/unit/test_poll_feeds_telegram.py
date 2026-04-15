import logging
import random
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from news_service.models.source import Source
from news_service.services.telegram import TelegramPost
from news_service.tasks import poll_adapters, poll_feeds

logging.disable(logging.CRITICAL)


def _make_telegram_source(
    *,
    channel: str | None = None,
) -> Source:
    ch = channel or f"канал_{uuid.uuid4().hex[:8]}"
    return Source(
        id=uuid.uuid4(),
        url=f"https://t.me/s/{ch}",
        title=f"Telegram @{ch}",
        source_description=f"Описание канала {ch}",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=random.randint(1, 100),
    )


def _make_fresh_telegram_post(*, channel: str = "fondnauk") -> TelegramPost:
    return TelegramPost(
        url=f"https://t.me/{channel}/{random.randint(1, 9999)}",
        body=f"Первая строка поста #{uuid.uuid4().hex[:6]}\nВторая строка",
        published_at=datetime.now(UTC) - timedelta(hours=random.randint(1, 12)),
    )


@pytest.mark.asyncio
async def test_poll_single_source_returns_one_for_fresh_telegram_post(mocker) -> None:
    src = _make_telegram_source(channel="fondnauk")
    post = _make_fresh_telegram_post(channel="fondnauk")

    mocker.patch.object(poll_adapters, "fetch_telegram_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert count == 1, "poller did not return one for a fresh telegram post"


@pytest.mark.asyncio
async def test_poll_single_source_calls_upsert_for_fresh_telegram_post(mocker) -> None:
    src = _make_telegram_source(channel="fondnauk")
    post = _make_fresh_telegram_post(channel="fondnauk")

    mocker.patch.object(poll_adapters, "fetch_telegram_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert upsert_news_item.await_count == 1, "upsert was not called for fresh telegram post"
