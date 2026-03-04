import uuid
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from news_service.models.rss_feed import RssFeed
from news_service.tasks import poll_feeds


@pytest.mark.asyncio
async def test_fetch_rss_feed_content_retries_before_succeeding() -> None:
    response = MagicMock()
    response.content = b"<rss />"
    response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=[httpx.ReadTimeout("timeout"), response])
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("news_service.tasks.poll_feeds.httpx.AsyncClient", return_value=mock_http):
        content = await poll_feeds._fetch_rss_feed_content("https://example.com/rss.xml")

    assert content == b"<rss />"
    assert mock_http.get.await_count == 2


@pytest.mark.asyncio
async def test_poll_single_feed_fetches_rss_with_helper(mocker) -> None:
    feed = RssFeed(
        id=uuid.uuid4(),
        url="https://example.com/rss.xml",
        title="Example Feed",
        topic_tags=["science"],
        topic_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "New lecture announced",
                "summary": "A new lecture was announced for next week.",
                "link": "https://example.com/posts/1",
            }
        ]
    )
    fetch_content = mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=b"<rss />"),
    )
    parse_feed = mocker.patch.object(
        poll_feeds.feedparser,
        "parse",
        return_value=parsed_feed,
    )
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    count = await poll_feeds._poll_single_feed(session=AsyncMock(), feed=feed)

    assert count == 1
    fetch_content.assert_awaited_once_with("https://example.com/rss.xml")
    parse_feed.assert_called_once_with(b"<rss />")
    upsert_news_item.assert_awaited_once()
    assert feed.last_polled_at is not None
    assert feed.last_polled_at.tzinfo == UTC
