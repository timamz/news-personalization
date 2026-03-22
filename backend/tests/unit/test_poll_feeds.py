import logging
import uuid
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from news_service.models.source import Source
from news_service.tasks import poll_feeds

logging.disable(logging.CRITICAL)


class _FakeResult:
    def __init__(self, sources: list[Source]) -> None:
        self._sources = sources

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Source]:
        return self._sources


class _FakeSession:
    def __init__(self, sources: list[Source]) -> None:
        self._sources = sources
        self.info: dict[str, list[uuid.UUID] | set] = {}
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, _statement) -> _FakeResult:
        return _FakeResult(self._sources)

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


def _make_rss_source(
    *,
    source_id: uuid.UUID | None = None,
    url: str | None = None,
    title: str = "Лента новостей",
) -> Source:
    return Source(
        id=source_id or uuid.uuid4(),
        url=url or f"https://example.com/{uuid.uuid4().hex}.xml",
        title=title,
        source_description=f"Описание источника {uuid.uuid4().hex[:6]}",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )


@pytest.mark.asyncio
async def test_fetch_rss_feed_content_returns_bytes_after_retry() -> None:
    rss_bytes = f"<rss>{uuid.uuid4().hex}</rss>".encode()
    response = MagicMock()
    response.content = rss_bytes
    response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=[httpx.ReadTimeout("тайм-аут"), response])
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("news_service.tasks.poll_feeds.httpx.AsyncClient", return_value=mock_http):
        content = await poll_feeds._fetch_rss_feed_content(
            f"https://example.com/{uuid.uuid4().hex}.xml"
        )

    assert content == rss_bytes, "fetcher did not return expected bytes after retry"


@pytest.mark.asyncio
async def test_fetch_rss_feed_content_retries_on_read_timeout() -> None:
    response = MagicMock()
    response.content = b"<rss />"
    response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=[httpx.ReadTimeout("тайм-аут"), response])
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("news_service.tasks.poll_feeds.httpx.AsyncClient", return_value=mock_http):
        await poll_feeds._fetch_rss_feed_content(f"https://example.com/{uuid.uuid4().hex}.xml")

    assert mock_http.get.await_count == 2, "fetcher did not retry after read timeout"


@pytest.mark.asyncio
async def test_poll_single_rss_source_returns_one_for_single_entry(mocker) -> None:
    src = _make_rss_source()
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "Новая лекция по палеонтологии",
                "summary": "Лекция пройдёт в четверг вечером.",
                "link": f"https://example.com/posts/{uuid.uuid4().hex}",
            }
        ]
    )
    mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=b"<rss />"),
    )
    mocker.patch.object(poll_feeds.feedparser, "parse", return_value=parsed_feed)
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert count == 1, "poller did not return one for a single rss entry"


@pytest.mark.asyncio
async def test_poll_single_rss_source_calls_fetch_content_with_source_url(mocker) -> None:
    src = _make_rss_source()
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "Заголовок",
                "summary": "Текст новости",
                "link": f"https://example.com/{uuid.uuid4().hex}",
            }
        ]
    )
    fetch_content = mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=b"<rss />"),
    )
    mocker.patch.object(poll_feeds.feedparser, "parse", return_value=parsed_feed)
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert fetch_content.await_args[0][0] == src.url, "fetch was not called with the source url"


@pytest.mark.asyncio
async def test_poll_single_rss_source_passes_raw_bytes_to_feedparser(mocker) -> None:
    src = _make_rss_source()
    raw_bytes = f"<rss>{uuid.uuid4().hex}</rss>".encode()
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "Заголовок",
                "summary": "Текст",
                "link": f"https://example.com/{uuid.uuid4().hex}",
            }
        ]
    )
    mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=raw_bytes),
    )
    parse_feed = mocker.patch.object(poll_feeds.feedparser, "parse", return_value=parsed_feed)
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert parse_feed.call_args[0][0] == raw_bytes, "feedparser was not called with raw bytes"


@pytest.mark.asyncio
async def test_poll_single_rss_source_calls_upsert_for_each_entry(mocker) -> None:
    src = _make_rss_source()
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "Статья",
                "summary": "Текст статьи",
                "link": f"https://example.com/{uuid.uuid4().hex}",
            }
        ]
    )
    mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=b"<rss />"),
    )
    mocker.patch.object(poll_feeds.feedparser, "parse", return_value=parsed_feed)
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert_news_item = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert_news_item)

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert upsert_news_item.await_count == 1, "upsert was not called for the rss entry"


@pytest.mark.asyncio
async def test_poll_single_rss_source_sets_last_polled_at(mocker) -> None:
    src = _make_rss_source()
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "Заголовок",
                "summary": "Текст",
                "link": f"https://example.com/{uuid.uuid4().hex}",
            }
        ]
    )
    mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=b"<rss />"),
    )
    mocker.patch.object(poll_feeds.feedparser, "parse", return_value=parsed_feed)
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert src.last_polled_at is not None, "last_polled_at was not set after polling"


@pytest.mark.asyncio
async def test_poll_single_rss_source_sets_last_polled_at_with_utc_timezone(mocker) -> None:
    src = _make_rss_source()
    parsed_feed = SimpleNamespace(
        entries=[
            {
                "title": "Заголовок",
                "summary": "Текст",
                "link": f"https://example.com/{uuid.uuid4().hex}",
            }
        ]
    )
    mocker.patch.object(
        poll_feeds,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=b"<rss />"),
    )
    mocker.patch.object(poll_feeds.feedparser, "parse", return_value=parsed_feed)
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    mocker.patch.object(poll_feeds, "upsert_news_item", new=AsyncMock(return_value=object()))

    await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert src.last_polled_at.tzinfo == UTC, "last_polled_at timezone was not utc"


@pytest.mark.asyncio
async def test_poll_all_feeds_returns_correct_feeds_polled_count(mocker) -> None:
    sources = [_make_rss_source(), _make_rss_source()]
    session = _FakeSession(sources)
    mocker.patch.object(
        poll_feeds,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        poll_feeds,
        "_poll_single_source",
        new=AsyncMock(side_effect=[1, 2]),
    )
    mocker.patch.object(poll_feeds.celery_app, "send_task")

    result = await poll_feeds._poll_all_feeds()

    assert result["feeds_polled"] == 2, "feeds_polled count did not match number of sources"


@pytest.mark.asyncio
async def test_poll_all_feeds_returns_correct_new_items_total(mocker) -> None:
    sources = [_make_rss_source(), _make_rss_source()]
    session = _FakeSession(sources)
    mocker.patch.object(
        poll_feeds,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        poll_feeds,
        "_poll_single_source",
        new=AsyncMock(side_effect=[1, 2]),
    )
    mocker.patch.object(poll_feeds.celery_app, "send_task")

    result = await poll_feeds._poll_all_feeds()

    assert result["new_items"] == 3, "new_items total did not match sum of polled items"


@pytest.mark.asyncio
async def test_poll_all_feeds_commits_once_per_source(mocker) -> None:
    sources = [_make_rss_source(), _make_rss_source()]
    session = _FakeSession(sources)
    mocker.patch.object(
        poll_feeds,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        poll_feeds,
        "_poll_single_source",
        new=AsyncMock(side_effect=[1, 2]),
    )
    mocker.patch.object(poll_feeds.celery_app, "send_task")

    await poll_feeds._poll_all_feeds()

    assert session.commits == 2, "session was not committed once per source"


@pytest.mark.asyncio
async def test_poll_all_feeds_does_not_rollback_on_success(mocker) -> None:
    sources = [_make_rss_source(), _make_rss_source()]
    session = _FakeSession(sources)
    mocker.patch.object(
        poll_feeds,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        poll_feeds,
        "_poll_single_source",
        new=AsyncMock(side_effect=[1, 2]),
    )
    mocker.patch.object(poll_feeds.celery_app, "send_task")

    await poll_feeds._poll_all_feeds()

    assert session.rollbacks == 0, "session was rolled back despite no errors"


@pytest.mark.asyncio
async def test_poll_all_feeds_does_not_queue_events_when_none_exist(mocker) -> None:
    sources = [_make_rss_source(), _make_rss_source()]
    session = _FakeSession(sources)
    mocker.patch.object(
        poll_feeds,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        poll_feeds,
        "_poll_single_source",
        new=AsyncMock(side_effect=[1, 2]),
    )
    send_task = mocker.patch.object(poll_feeds.celery_app, "send_task")

    await poll_feeds._poll_all_feeds()

    assert send_task.call_count == 0, "event notifications were queued when none should exist"
