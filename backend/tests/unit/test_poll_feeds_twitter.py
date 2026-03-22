import logging
import random
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from news_service.models.source import Source
from news_service.services.twitter import TwitterPost
from news_service.tasks import poll_feeds

logging.disable(logging.CRITICAL)


def _make_twitter_source(
    *,
    account: str | None = None,
) -> Source:
    acct = account or f"акк_{uuid.uuid4().hex[:8]}"
    return Source(
        id=uuid.uuid4(),
        url=f"https://x.com/{acct}",
        title=f"X @{acct}",
        source_description=f"Описание аккаунта {acct}",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=random.randint(1, 100),
    )


def _make_fresh_twitter_post(*, account: str = "openai") -> TwitterPost:
    return TwitterPost(
        url=f"https://x.com/{account}/status/{random.randint(10**18, 10**19)}",
        title=f"Пост #{uuid.uuid4().hex[:6]} от @{account}",
        body=f"Текст поста #{uuid.uuid4().hex[:6]}",
        published_at=datetime.now(UTC) - timedelta(hours=random.randint(1, 12)),
    )


def _make_stale_twitter_post(*, account: str = "paperswithcode") -> TwitterPost:
    return TwitterPost(
        url=f"https://x.com/{account}/status/{random.randint(10**18, 10**19)}",
        title=f"Старый пост #{random.randint(1, 999)}",
        body="Устаревший текст поста",
        published_at=datetime(2021, 12, 30, 13, 24, 25, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_poll_single_source_returns_one_for_fresh_twitter_post(mocker) -> None:
    src = _make_twitter_source(account="openai")
    post = _make_fresh_twitter_post(account="openai")

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert count == 1, "poller did not return one for a fresh twitter post"


@pytest.mark.asyncio
async def test_poll_single_source_calls_upsert_for_fresh_twitter_post(mocker) -> None:
    src = _make_twitter_source(account="openai")
    post = _make_fresh_twitter_post(account="openai")

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert upsert_news_item.await_count == 1, "upsert was not called for fresh twitter post"


@pytest.mark.asyncio
async def test_poll_single_source_returns_zero_for_stale_twitter_post(mocker) -> None:
    src = _make_twitter_source(account="paperswithcode")
    post = _make_stale_twitter_post(account="paperswithcode")

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert count == 0, "poller did not return zero for a stale twitter post"


@pytest.mark.asyncio
async def test_poll_single_source_does_not_embed_stale_twitter_post(mocker) -> None:
    src = _make_twitter_source(account="paperswithcode")
    post = _make_stale_twitter_post(account="paperswithcode")

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=[post]))
    embed_texts = mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert embed_texts.await_count == 0, "embed_texts was called for a stale twitter post"


@pytest.mark.asyncio
async def test_poll_single_source_does_not_upsert_stale_twitter_post(mocker) -> None:
    src = _make_twitter_source(account="paperswithcode")
    post = _make_stale_twitter_post(account="paperswithcode")

    mocker.patch.object(poll_feeds, "fetch_twitter_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    upsert_news_item = mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert upsert_news_item.await_count == 0, "upsert was called for a stale twitter post"
