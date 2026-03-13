import uuid
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from news_service.models.rss_feed import RssFeed
from news_service.tasks import poll_feeds


class _FakeResult:
    def __init__(self, feeds: list[RssFeed]) -> None:
        self._feeds = feeds

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[RssFeed]:
        return self._feeds


class _FakeSession:
    def __init__(self, feeds: list[RssFeed]) -> None:
        self._feeds = feeds
        self.info: dict[str, list[uuid.UUID]] = {}
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, _statement) -> _FakeResult:  # noqa: ANN001
        return _FakeResult(self._feeds)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
        return False


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
        source_description="Science news feed",
        source_description_embedding=None,
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


@pytest.mark.asyncio
async def test_poll_all_feeds_commits_after_each_feed(mocker) -> None:
    feeds = [
        RssFeed(
            id=uuid.uuid4(),
            url="https://example.com/1.xml",
            title="Feed 1",
            source_description="AI news feed",
            source_description_embedding=None,
            is_active=True,
            last_polled_at=None,
            subscriber_count=1,
        ),
        RssFeed(
            id=uuid.uuid4(),
            url="https://example.com/2.xml",
            title="Feed 2",
            source_description="ML news feed",
            source_description_embedding=None,
            is_active=True,
            last_polled_at=None,
            subscriber_count=1,
        ),
    ]
    session = _FakeSession(feeds)
    mocker.patch.object(
        poll_feeds,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        poll_feeds,
        "_poll_single_feed",
        new=AsyncMock(side_effect=[1, 2]),
    )
    send_task = mocker.patch.object(poll_feeds.celery_app, "send_task")

    result = await poll_feeds._poll_all_feeds()

    assert result == {
        "feeds_polled": 2,
        "new_items": 3,
        "event_notifications_queued": 0,
    }
    assert session.commits == 2
    assert session.rollbacks == 0
    send_task.assert_not_called()
