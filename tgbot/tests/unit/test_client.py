from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tgbot.client import BackendClient
from tgbot.core.config import get_settings


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
async def test_get_current_user(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "user-1",
        "api_key": "generated-key",
        "timezone": "Europe/Berlin",
        "created_at": "2026-01-01T00:00:00Z",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        user = await client.get_current_user("api-key")

    assert user.timezone == "Europe/Berlin"
    mock_http.get.assert_awaited_once_with(
        "http://test-backend:8000/users/me",
        headers={"X-API-Key": "api-key"},
    )


@pytest.mark.asyncio
async def test_resolve_timezone(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "resolved",
        "candidates": [
            {
                "label": "Berlin, Germany",
                "timezone": "Europe/Berlin",
                "local_time": "2026-03-13T10:00:00+01:00",
            }
        ],
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        resolution = await client.resolve_timezone("api-key", "Berlin")

    assert resolution.status == "resolved"
    assert resolution.candidates[0].timezone == "Europe/Berlin"
    mock_http.post.assert_awaited_once_with(
        "http://test-backend:8000/users/resolve-timezone",
        headers={"X-API-Key": "api-key"},
        json={"query": "Berlin"},
    )


@pytest.mark.asyncio
async def test_update_user_timezone(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "user-1",
        "api_key": "generated-key",
        "timezone": "Europe/Berlin",
        "created_at": "2026-01-01T00:00:00Z",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.patch = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        user = await client.update_user_timezone("api-key", "Europe/Berlin")

    assert user.timezone == "Europe/Berlin"
    mock_http.patch.assert_awaited_once_with(
        "http://test-backend:8000/users/me",
        headers={"X-API-Key": "api-key"},
        json={"timezone": "Europe/Berlin"},
    )


@pytest.mark.asyncio
async def test_create_subscription(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": "sub-456",
        "raw_prompt": "AI news every morning",
        "topics": ["artificial intelligence"],
        "delivery_mode": "digest",
        "schedule_cron": "0 8 * * *",
        "format_instructions": "brief summary",
        "digest_language": "en",
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
    assert sub.delivery_mode == "digest"
    assert sub.digest_language == "en"


@pytest.mark.asyncio
async def test_create_subscription_uses_configured_timeout(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": "sub-789",
        "raw_prompt": "AI news every morning",
        "topics": ["artificial intelligence"],
        "delivery_mode": "digest",
        "schedule_cron": "0 8 * * *",
        "format_instructions": "brief summary",
        "digest_language": "en",
        "delivery_webhook_url": "http://bot:8001/deliver/123",
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http) as patched_client:
        await client.create_subscription(
            "my-key", "AI news every morning", "http://bot:8001/deliver/123"
        )

    patched_client.assert_called_once_with(
        timeout=get_settings().backend_create_subscription_timeout_seconds
    )


@pytest.mark.asyncio
async def test_create_subscription_sends_source_preferences(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": "sub-999",
        "raw_prompt": "ML news",
        "topics": ["machine learning"],
        "delivery_mode": "digest",
        "schedule_cron": "0 8 * * *",
        "format_instructions": "brief summary",
        "digest_language": "ru",
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
        await client.create_subscription(
            "my-key",
            "ML news",
            "http://bot:8001/deliver/123",
            fixed_telegram_channels=["gonzo_ml"],
            fixed_reddit_subreddits=["machinelearning"],
            fixed_twitter_accounts=["openai"],
            include_discovered_sources=True,
            schedule_cron_override="0 9 * * *",
            manual_only=False,
            delivery_mode="digest",
            digest_language="ru",
        )

    mock_http.post.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions",
        headers={"X-API-Key": "my-key"},
        json={
            "prompt": "ML news",
            "delivery_webhook_url": "http://bot:8001/deliver/123",
            "fixed_telegram_channels": ["gonzo_ml"],
            "fixed_reddit_subreddits": ["machinelearning"],
            "fixed_twitter_accounts": ["openai"],
            "include_discovered_sources": True,
            "schedule_cron_override": "0 9 * * *",
            "manual_only": False,
            "delivery_mode": "digest",
            "digest_language_override": "ru",
        },
    )


@pytest.mark.asyncio
async def test_parse_subscription_prompt(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "topics": ["machine learning"],
        "delivery_mode": "event",
        "schedule_cron": None,
        "schedule_was_explicit": False,
        "format_instructions": "brief summary",
        "digest_language": "ru",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        parsed = await client.parse_subscription_prompt("my-key", "ML новости")

    assert parsed.delivery_mode == "event"
    assert parsed.schedule_was_explicit is False
    assert parsed.schedule_cron is None
    mock_http.post.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions/parse",
        headers={"X-API-Key": "my-key"},
        json={"prompt": "ML новости"},
    )


@pytest.mark.asyncio
async def test_parse_schedule(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"schedule_cron": "0 9 * * 1-5"}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        cron = await client.parse_schedule("my-key", "каждый будний день в 9")

    assert cron == "0 9 * * 1-5"
    mock_http.post.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions/parse-schedule",
        headers={"X-API-Key": "my-key"},
        json={"schedule_text": "каждый будний день в 9"},
    )


@pytest.mark.asyncio
async def test_list_subscriptions(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "sub-1",
            "raw_prompt": "sports news",
            "topics": ["sports"],
            "delivery_mode": "event",
            "schedule_cron": "0 8 * * *",
            "format_instructions": "brief summary",
            "digest_language": "en",
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
    assert subs[0].delivery_mode == "event"
    assert subs[0].digest_language == "en"


@pytest.mark.asyncio
async def test_list_recent_events(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "news_item_ids": ["news-1", "news-2"],
        "subject": "Recent events you may have missed",
        "body": "- Demo concert\n- Another concert",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http) as patched_client:
        events = await client.list_recent_events("my-key", "sub-1")

    assert events is not None
    assert events.news_item_ids == ["news-1", "news-2"]
    assert events.subject == "Recent events you may have missed"
    patched_client.assert_called_once_with(
        timeout=get_settings().backend_slow_request_timeout_seconds
    )
    mock_http.get.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions/sub-1/recent-events",
        headers={"X-API-Key": "my-key"},
    )


@pytest.mark.asyncio
async def test_acknowledge_recent_events(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        await client.acknowledge_recent_events("my-key", "sub-1", ["news-1", "news-2"])

    mock_http.post.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions/sub-1/recent-events/acknowledge",
        headers={"X-API-Key": "my-key"},
        json={"news_item_ids": ["news-1", "news-2"]},
    )


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


@pytest.mark.asyncio
async def test_update_subscription(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "sub-1",
        "raw_prompt": "AI news every morning",
        "topics": ["artificial intelligence"],
        "delivery_mode": "digest",
        "schedule_cron": None,
        "format_instructions": "concise alerts",
        "digest_language": "ru",
        "delivery_webhook_url": "http://bot:8001/deliver/123",
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.patch = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        sub = await client.update_subscription(
            "my-key",
            "sub-1",
            schedule_cron=None,
            format_instructions="concise alerts",
            digest_language="ru",
        )

    assert sub.id == "sub-1"
    assert sub.schedule_cron is None
    assert sub.format_instructions == "concise alerts"
    assert sub.digest_language == "ru"
    mock_http.patch.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions/sub-1",
        headers={"X-API-Key": "my-key"},
        json={
            "schedule_cron": None,
            "format_instructions": "concise alerts",
            "digest_language": "ru",
        },
    )


@pytest.mark.asyncio
async def test_append_subscription_sources(client: BackendClient):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "added_telegram_channels": ["gonzo_ml"],
        "added_reddit_subreddits": ["machinelearning"],
        "added_twitter_accounts": [],
        "added_sources_count": 2,
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.append_subscription_sources(
            "api-key",
            "sub-1",
            fixed_telegram_channels=["gonzo_ml"],
            fixed_reddit_subreddits=["machinelearning"],
        )

    assert result.added_sources_count == 2
    mock_http.post.assert_awaited_once_with(
        "http://test-backend:8000/subscriptions/sub-1/sources",
        headers={"X-API-Key": "api-key"},
        json={
            "fixed_telegram_channels": ["gonzo_ml"],
            "fixed_reddit_subreddits": ["machinelearning"],
        },
    )
