from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.discovery import validate_feed_url

PATCH_TARGET = "news_service.agents.discovery.httpx.AsyncClient"


def _make_mock_client(status_code: int, text: str = "") -> AsyncMock:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text

    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.asyncio
async def test_validate_feed_url_success():
    rss_xml = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <item><title>Test</title><link>http://example.com</link></item>
      </channel>
    </rss>"""
    mock_client = _make_mock_client(200, rss_xml)

    with patch(PATCH_TARGET, return_value=mock_client):
        result = await validate_feed_url("https://example.com/feed")

    assert result is True


@pytest.mark.asyncio
async def test_validate_feed_url_404():
    mock_client = _make_mock_client(404)

    with patch(PATCH_TARGET, return_value=mock_client):
        result = await validate_feed_url("https://example.com/not-a-feed")

    assert result is False


@pytest.mark.asyncio
async def test_validate_feed_url_empty_feed():
    empty_rss = """<?xml version="1.0"?>
    <rss version="2.0"><channel></channel></rss>"""
    mock_client = _make_mock_client(200, empty_rss)

    with patch(PATCH_TARGET, return_value=mock_client):
        result = await validate_feed_url("https://example.com/empty-feed")

    assert result is False
