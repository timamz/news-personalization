"""Tests for the poll_feeds task and the generic polling loop."""

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from news_service.tasks import poll_adapters

logging.disable(logging.CRITICAL)


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
