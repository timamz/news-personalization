from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tgbot.webhook_server import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    create_webhook_app,
    delivery_webhook_path,
    set_bot,
)


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
        delivery_webhook_path(12345),
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
        delivery_webhook_path(12345),
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status == 400


@pytest.mark.asyncio
async def test_deliver_no_bot(webhook_client: TestClient):
    set_bot(None)

    response = await webhook_client.post(
        delivery_webhook_path(12345),
        json={"subject": "Test", "body": "Body"},
    )
    assert response.status == 503


@pytest.mark.asyncio
async def test_deliver_splits_long_message(webhook_client: TestClient):
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    set_bot(mock_bot)

    long_body = "a" * (TELEGRAM_MAX_MESSAGE_LENGTH * 2)
    response = await webhook_client.post(
        delivery_webhook_path(12345),
        json={"subject": "Long Digest", "body": long_body},
    )

    assert response.status == 200
    assert mock_bot.send_message.await_count >= 2


@pytest.mark.asyncio
async def test_deliver_rejects_legacy_unauthenticated_path(webhook_client: TestClient):
    set_bot(AsyncMock())

    response = await webhook_client.post(
        "/deliver/12345",
        json={"subject": "Test Digest", "body": "Here is your news."},
    )

    assert response.status == 403


@pytest.mark.asyncio
async def test_deliver_rejects_invalid_token(webhook_client: TestClient):
    set_bot(AsyncMock())

    response = await webhook_client.post(
        "/deliver/not-the-right-token/12345",
        json={"subject": "Test Digest", "body": "Here is your news."},
    )

    assert response.status == 403
