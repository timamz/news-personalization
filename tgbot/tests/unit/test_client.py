from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tgbot.client import BackendClient


@pytest.fixture
def client():
    return BackendClient(base_url="http://test-backend:8000")


@pytest.mark.asyncio
async def test_register_user(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": "abc-123",
        "api_key": "generated-key",
        "created_at": "2026-01-01T00:00:00Z",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        api_key = await client.register_user()

    assert api_key == "generated-key"
    mock_http.post.assert_called_once_with("http://test-backend:8000/users")


@pytest.mark.asyncio
async def test_create_subscription(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": "sub-456",
        "raw_prompt": "AI news every morning",
        "topics": ["artificial intelligence"],
        "schedule_cron": "0 8 * * *",
        "format_instructions": "brief summary",
        "delivery_webhook_url": "http://bot:8001/deliver/123",
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        sub = await client.create_subscription(
            "my-key", "AI news every morning", "http://bot:8001/deliver/123"
        )

    assert sub.id == "sub-456"
    assert sub.topics == ["artificial intelligence"]


@pytest.mark.asyncio
async def test_list_subscriptions(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "sub-1",
            "raw_prompt": "sports news",
            "topics": ["sports"],
            "schedule_cron": "0 8 * * *",
            "format_instructions": "brief summary",
            "is_active": True,
        }
    ]
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        subs = await client.list_subscriptions("my-key")

    assert len(subs) == 1
    assert subs[0].topics == ["sports"]


@pytest.mark.asyncio
async def test_send_now(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_response.json.return_value = {"task_id": "task-123", "status": "queued"}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.send_now("my-key", "sub-1")

    assert result == {"task_id": "task-123", "status": "queued"}
    mock_http.post.assert_called_once_with(
        "http://test-backend:8000/subscriptions/sub-1/send-now",
        headers={"X-API-Key": "my-key"},
    )
