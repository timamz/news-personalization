import logging
import random
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from news_service.models.source import Source
from news_service.services.reddit import RedditPost
from news_service.tasks import poll_feeds

logging.disable(logging.CRITICAL)


def _make_reddit_source(
    *,
    subreddit: str | None = None,
) -> Source:
    sub = subreddit or f"sub_{uuid.uuid4().hex[:8]}"
    return Source(
        id=uuid.uuid4(),
        url=f"https://www.reddit.com/r/{sub}/new/",
        title=f"Reddit r/{sub}",
        source_description=f"Описание сабреддита {sub}",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=random.randint(1, 100),
    )


def _make_fresh_reddit_post(*, subreddit: str = "badminton") -> RedditPost:
    return RedditPost(
        url=f"https://www.reddit.com/r/{subreddit}/comments/{uuid.uuid4().hex[:6]}/тред/",
        title=f"Обсуждение турнира #{random.randint(1, 999)}",
        body="Что вы думаете о финале?",
        published_at=datetime.now(UTC) - timedelta(hours=random.randint(1, 12)),
    )


def _make_stale_reddit_post(*, subreddit: str = "arxiv") -> RedditPost:
    return RedditPost(
        url=f"https://www.reddit.com/r/{subreddit}/comments/{uuid.uuid4().hex[:6]}/тред/",
        title=f"Старый пост #{random.randint(1, 999)}",
        body="Устаревшее обсуждение",
        published_at=datetime(2024, 10, 6, 0, 46, 35, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_poll_single_source_returns_one_for_fresh_reddit_post(mocker) -> None:
    src = _make_reddit_source(subreddit="badminton")
    post = _make_fresh_reddit_post(subreddit="badminton")

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert count == 1, "poller did not return one for a fresh reddit post"


@pytest.mark.asyncio
async def test_poll_single_source_calls_upsert_for_fresh_reddit_post(mocker) -> None:
    src = _make_reddit_source(subreddit="badminton")
    post = _make_fresh_reddit_post(subreddit="badminton")

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert upsert_news_item.await_count == 1, "upsert was not called for fresh reddit post"


@pytest.mark.asyncio
async def test_poll_single_source_returns_zero_for_stale_reddit_post(mocker) -> None:
    src = _make_reddit_source(subreddit="arxiv")
    post = _make_stale_reddit_post(subreddit="arxiv")

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert count == 0, "poller did not return zero for a stale reddit post"


@pytest.mark.asyncio
async def test_poll_single_source_does_not_embed_stale_reddit_post(mocker) -> None:
    src = _make_reddit_source(subreddit="arxiv")
    post = _make_stale_reddit_post(subreddit="arxiv")

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=[post]))
    embed_texts = mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert embed_texts.await_count == 0, "embed_texts was called for a stale reddit post"


@pytest.mark.asyncio
async def test_poll_single_source_does_not_upsert_stale_reddit_post(mocker) -> None:
    src = _make_reddit_source(subreddit="arxiv")
    post = _make_stale_reddit_post(subreddit="arxiv")

    mocker.patch.object(poll_feeds, "fetch_reddit_posts", new=AsyncMock(return_value=[post]))
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock())
    upsert_news_item = mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock())

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert upsert_news_item.await_count == 0, "upsert was called for a stale reddit post"
