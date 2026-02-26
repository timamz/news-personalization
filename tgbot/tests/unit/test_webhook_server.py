from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tgbot.webhook_server import create_webhook_app, set_bot


@pytest.fixture
async def webhook_client():
    app = create_webhook_app()
    async with TestClient(TestServer(app)) as client:
        yield client


@pytest.mark.asyncio
async def test_deliver_success(webhook_client: TestClient):
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    set_bot(mock_bot)

    response = await webhook_client.post(
        "/deliver/12345",
        json={"subject": "Test Digest", "body": "Here is your news."},
    )
    assert response.status == 200
    data = await response.json()
    assert data["status"] == "delivered"

    mock_bot.send_message.assert_called_once()
    call_kwargs = mock_bot.send_message.call_args
    assert call_kwargs.kwargs["chat_id"] == 12345


@pytest.mark.asyncio
async def test_deliver_invalid_json(webhook_client: TestClient):
    set_bot(AsyncMock())

    response = await webhook_client.post(
        "/deliver/12345",
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status == 400


@pytest.mark.asyncio
async def test_deliver_no_bot(webhook_client: TestClient):
    set_bot(None)

    response = await webhook_client.post(
        "/deliver/12345",
        json={"subject": "Test", "body": "Body"},
    )
    assert response.status == 503
