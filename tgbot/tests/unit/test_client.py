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


@pytest.mark.asyncio
async def test_acknowledge_onboarding_posts_with_api_key_header() -> None:
    base = f"http://test-{uuid.uuid4().hex[:8]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    client = BackendClient(base_url=base)
    resp = _make_response(204)
    mock_http = _make_http_mock("post", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        await client.acknowledge_onboarding(api_key)

    mock_http.post.assert_called_once_with(
        f"{base}/users/me/acknowledge-onboarding",
        headers={"X-API-Key": api_key},
    )


@pytest.mark.asyncio
async def test_api_key_is_valid_returns_true_on_200() -> None:
    base = f"http://test-{uuid.uuid4().hex[:8]}:8000"
    api_key = f"key-{uuid.uuid4().hex}"
    client = BackendClient(base_url=base)
    resp = _make_response(200, {"id": uuid.uuid4().hex})
    mock_http = _make_http_mock("get", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.api_key_is_valid(api_key)

    mock_http.get.assert_called_once_with(
        f"{base}/users/me",
        headers={"X-API-Key": api_key},
    )
    assert result is True, "api_key_is_valid returned False for a 200 response"


@pytest.mark.asyncio
async def test_api_key_is_valid_returns_false_on_401() -> None:
    base = f"http://test-{uuid.uuid4().hex[:8]}:8000"
    api_key = f"stale-{uuid.uuid4().hex}"
    client = BackendClient(base_url=base)
    resp = _make_response(401)
    mock_http = _make_http_mock("get", resp)

    with patch("tgbot.client.httpx.AsyncClient", return_value=mock_http):
        result = await client.api_key_is_valid(api_key)

    assert result is False, "api_key_is_valid did not treat 401 as an invalid key"
