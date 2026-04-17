"""Tests for the poll_feeds task and the generic polling loop."""

import logging
import uuid
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from news_service.models.source import Source
from news_service.tasks import poll_adapters, poll_feeds

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


def _make_rss_source() -> Source:
    return Source(
        id=uuid.uuid4(),
        url=f"https://example.com/{uuid.uuid4().hex}.xml",
        title="Feed",
        source_description=f"Description {uuid.uuid4().hex[:6]}",
        source_description_embedding=None,
        is_active=True,
        last_polled_at=None,
        subscriber_count=1,
    )


@pytest.mark.asyncio
async def test_fetch_rss_feed_content_retries_after_read_timeout() -> None:
    rss_bytes = f"<rss>{uuid.uuid4().hex}</rss>".encode()
    response = MagicMock()
    response.content = rss_bytes
    response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=[httpx.ReadTimeout("timeout"), response])
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("news_service.tasks.poll_adapters.httpx.AsyncClient", return_value=mock_http):
        content = await poll_adapters._fetch_rss_feed_content(
            f"https://example.com/{uuid.uuid4().hex}.xml"
        )

    assert content == rss_bytes and mock_http.get.await_count == 2, (
        "fetcher did not retry after a read timeout and return the second-attempt bytes"
    )


@pytest.mark.asyncio
async def test_poll_single_rss_source_upserts_entries_and_stamps_last_polled_at(mocker) -> None:
    src = _make_rss_source()
    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>Lecture</title>"
        f"<link>https://example.com/posts/{uuid.uuid4().hex}</link>"
        "<summary>Body</summary></item>"
        "</channel></rss>"
    )

    mocker.patch.object(
        poll_adapters,
        "_fetch_rss_feed_content",
        new=AsyncMock(return_value=xml.encode()),
    )
    mocker.patch.object(poll_feeds, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]))
    upsert = AsyncMock(return_value=object())
    mocker.patch.object(poll_feeds, "upsert_news_item", new=upsert)

    count = await poll_feeds._poll_single_source(session=AsyncMock(), src=src)

    assert (
        count == 1
        and upsert.await_count == 1
        and src.last_polled_at is not None
        and src.last_polled_at.tzinfo == UTC
    ), "polling did not upsert the entry and stamp last_polled_at with a UTC timestamp"


@pytest.mark.asyncio
async def test_poll_all_feeds_commits_per_source_and_reports_totals(mocker) -> None:
    sources = [_make_rss_source(), _make_rss_source()]
    session = _FakeSession(sources)
    mocker.patch.object(poll_feeds, "get_task_session", return_value=_FakeSessionFactory(session))
    mocker.patch.object(poll_feeds, "_poll_single_source", new=AsyncMock(side_effect=[1, 2]))
    send_task = mocker.patch.object(poll_feeds.celery_app, "send_task")

    result = await poll_feeds._poll_all_feeds()

    assert (
        result["feeds_polled"] == 2
        and result["new_items"] == 3
        and session.commits == 2
        and session.rollbacks == 0
        and send_task.call_count == 0
    ), "poll_all_feeds did not commit per-source or report totals as expected"
