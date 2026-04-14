import logging
import random
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tgbot.client import BackendClient

logging.disable(logging.CRITICAL)


def _make_http_mock(method: str, response_mock: MagicMock) -> AsyncMock:
    mock_http = AsyncMock()
    setattr(mock_http, method, AsyncMock(return_value=response_mock))
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    return mock_http


def _make_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


@pytest.mark.asyncio
async def test_register_user_posts_to_users_endpoint_and_returns_api_key() -> None:
    base = f"http://test-{uuid.uuid4().hex[:8]}:8000"
    generated_key = f"key-{uuid.uuid4().hex}"
    client = BackendClient(base_url=base)
    resp = _make_response(
        201,
        {"id": uuid.uuid4().hex, "api_key": generated_key, "created_at": "2026-01-01T00:00:00Z"},
    )
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.register_user()

    mock_http.post.assert_called_once_with(f"{base}/users")
    assert result == generated_key, "register_user did not return the expected api_key"


@pytest.mark.asyncio
async def test_get_current_user_returns_timezone() -> None:
    base = f"http://backend-{uuid.uuid4().hex[:6]}:8000"
    tz = f"Europe/City_{random.randint(1, 999)}"
    api_key = f"key-{uuid.uuid4().hex}"
    client = BackendClient(base_url=base)
    resp = _make_response(
        200, {"id": "u1", "api_key": api_key, "timezone": tz, "created_at": "2026-01-01T00:00:00Z"}
    )
    mock_http = _make_http_mock("get", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        user = await client.get_current_user(api_key)

    assert user.timezone == tz, "get_current_user did not return the expected timezone"


@pytest.mark.asyncio
async def test_resolve_timezone_returns_resolved_status_with_candidate() -> None:
    base = f"http://srv-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    tz_name = f"Europe/Zone_{random.randint(1, 99)}"
    client = BackendClient(base_url=base)
    resp = _make_response(
        200,
        {
            "status": "resolved",
            "candidates": [
                {
                    "label": "Somewhere",
                    "timezone": tz_name,
                    "local_time": "2026-03-13T10:00:00+01:00",
                }
            ],
        },
    )
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        resolution = await client.resolve_timezone(api_key, "query")

    assert resolution.status == "resolved", "resolve_timezone did not return resolved status"
    assert resolution.candidates[0].timezone == tz_name, (
        "resolve_timezone did not return the expected candidate timezone"
    )


@pytest.mark.asyncio
async def test_update_user_timezone_returns_updated_timezone() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    tz = f"America/Zone_{random.randint(1, 99)}"
    client = BackendClient(base_url=base)
    resp = _make_response(
        200, {"id": "u", "api_key": api_key, "timezone": tz, "created_at": "2026-01-01T00:00:00Z"}
    )
    mock_http = _make_http_mock("patch", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        user = await client.update_user_timezone(api_key, tz)

    assert user.timezone == tz, "update_user_timezone did not return the expected timezone"


@pytest.mark.asyncio
async def test_parse_schedule_returns_cron_expression() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    cron = f"{random.randint(0, 59)} {random.randint(0, 23)} * * 1-5"
    client = BackendClient(base_url=base)
    resp = _make_response(200, {"schedule_cron": cron})
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.parse_schedule(
            api_key,
            "\u043a\u0430\u0436\u0434\u044b\u0439 \u0431\u0443\u0434\u043d\u0438\u0439"
            " \u0434\u0435\u043d\u044c \u0432 9",
        )

    assert result == cron, "parse_schedule did not return the expected cron expression"


@pytest.mark.asyncio
async def test_list_subscriptions_returns_parsed_subscription_fields() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    sub_id = f"sub-{uuid.uuid4().hex}"
    lang = random.choice(["en", "ru"])
    client = BackendClient(base_url=base)
    resp = _make_response(200, None)
    resp.json.return_value = [
        {
            "id": sub_id,
            "raw_prompt": "\u043d\u043e\u0432\u043e\u0441\u0442\u0438",
            "user_spec": "## Topic\n\u0441\u043f\u043e\u0440\u0442",
            "delivery_mode": "event",
            "schedule_cron": "0 8 * * *",
            "format_instructions": "brief",
            "digest_language": lang,
            "is_active": True,
        }
    ]
    mock_http = _make_http_mock("get", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        subs = await client.list_subscriptions(api_key)

    assert len(subs) == 1, "list_subscriptions did not return exactly one subscription"
    assert subs[0].delivery_mode == "event", (
        "list_subscriptions did not return the expected delivery_mode"
    )
    assert subs[0].digest_language == lang, (
        "list_subscriptions did not return the expected digest_language"
    )


@pytest.mark.asyncio
async def test_acknowledge_recent_events_posts_to_correct_endpoint() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    sub_id = f"sub-{uuid.uuid4().hex}"
    news_ids = [f"news-{uuid.uuid4().hex}" for _ in range(random.randint(1, 5))]
    client = BackendClient(base_url=base)
    resp = _make_response(204)
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        await client.acknowledge_recent_events(api_key, sub_id, news_ids)

    mock_http.post.assert_awaited_once_with(
        f"{base}/subscriptions/{sub_id}/recent-events/acknowledge",
        headers={"X-API-Key": api_key},
        json={"news_item_ids": news_ids},
    )


@pytest.mark.asyncio
async def test_send_now_returns_task_id() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    sub_id = f"sub-{uuid.uuid4().hex}"
    task_id = f"task-{uuid.uuid4().hex}"
    client = BackendClient(base_url=base)
    resp = _make_response(202, {"task_id": task_id, "status": "queued"})
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.send_now(api_key, sub_id)

    assert result["task_id"] == task_id, "send_now did not return the expected task_id"


@pytest.mark.asyncio
async def test_update_subscription_returns_correct_id_and_format_instructions() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    sub_id = f"sub-{uuid.uuid4().hex}"
    fmt = f"\u0444\u043e\u0440\u043c\u0430\u0442-{uuid.uuid4().hex[:6]}"
    client = BackendClient(base_url=base)
    resp = _make_response(
        200,
        {
            "id": sub_id,
            "raw_prompt": "\u041d\u043e\u0432\u043e\u0441\u0442\u0438 \u0418\u0418",
            "user_spec": "## Topic\n\u041d\u043e\u0432\u043e\u0441\u0442\u0438 \u0418\u0418",
            "delivery_mode": "digest",
            "schedule_cron": None,
            "format_instructions": fmt,
            "digest_language": "ru",
            "delivery_webhook_url": "http://bot:8001/deliver/123",
            "is_active": True,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    mock_http = _make_http_mock("patch", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        sub = await client.update_subscription(api_key, sub_id, format_instructions=fmt)

    assert sub.id == sub_id, "update_subscription did not return the expected subscription id"
    assert sub.format_instructions == fmt, (
        "update_subscription did not return the expected format_instructions"
    )


@pytest.mark.asyncio
async def test_append_subscription_sources_returns_added_count() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    sub_id = f"sub-{uuid.uuid4().hex}"
    count = random.randint(1, 10)
    client = BackendClient(base_url=base)
    resp = _make_response(
        200,
        {
            "added_telegram_channels": ["test_ch"],
            "added_reddit_subreddits": [],
            "added_twitter_accounts": [],
            "added_sources_count": count,
        },
    )
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.append_subscription_sources(
            api_key, sub_id, fixed_telegram_channels=["test_ch"]
        )

    assert result.added_sources_count == count, (
        "append_subscription_sources did not return the expected added_sources_count"
    )


@pytest.mark.asyncio
async def test_apply_subscription_edit_config_returns_subscription() -> None:
    base = f"http://host-{uuid.uuid4().hex[:6]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    sub_id = f"sub-{uuid.uuid4().hex}"
    user_spec = f"## Topic\nРезюме-{uuid.uuid4().hex[:6]}"
    client = BackendClient(base_url=base)
    resp = _make_response(
        200,
        {
            "id": sub_id,
            "raw_prompt": "old",
            "user_spec": user_spec,
            "delivery_mode": "event",
            "schedule_cron": None,
            "format_instructions": "brief",
            "digest_language": "ru",
            "is_active": True,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.apply_subscription_edit_config(
            api_key,
            sub_id,
            config={"user_spec": user_spec},
        )

    assert result.user_spec == user_spec, (
        "apply_subscription_edit_config did not return the expected user_spec"
    )
