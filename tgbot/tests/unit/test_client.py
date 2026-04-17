"""Tests for the slim BackendClient used by the tgbot transport layer."""

import logging
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
        {
            "id": uuid.uuid4().hex,
            "api_key": generated_key,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.register_user()

    mock_http.post.assert_called_once_with(f"{base}/users")
    assert result == generated_key, "register_user did not return the expected api_key"


def test_build_create_payload_renames_digest_language_key() -> None:
    payload = BackendClient._build_create_payload(
        "topic", "http://bot/deliver/1", digest_language="fr"
    )
    assert payload["digest_language_override"] == "fr", (
        "_build_create_payload did not rename digest_language to digest_language_override"
    )
    assert "digest_language" not in payload, (
        "_build_create_payload left the pre-rename key in the payload"
    )


def test_build_create_payload_skips_none_values() -> None:
    payload = BackendClient._build_create_payload(
        "topic",
        "http://bot/deliver/1",
        schedule_cron_override=None,
        delivery_mode="digest",
    )
    assert "schedule_cron_override" not in payload, (
        "_build_create_payload included a key whose value was None"
    )
    assert payload["delivery_mode"] == "digest", (
        "_build_create_payload dropped a non-None keyword argument"
    )
