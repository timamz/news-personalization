"""Unit tests for the admin-alert throttling and dispatch."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.core.provider_errors import ProviderLimitError
from news_service.services.admin_alerts import notify_provider_limit

logging.disable(logging.CRITICAL)

MODULE = "news_service.services.admin_alerts"


def _redis_with_storage(initial: dict[str, str] | None = None) -> MagicMock:
    storage = dict(initial or {})

    async def mock_set(key, value, ex=None, nx=False):
        del ex
        if nx and key in storage:
            return None
        storage[key] = value
        return True

    fake = MagicMock()
    fake.set = AsyncMock(side_effect=mock_set)
    fake._storage = storage
    return fake


@pytest.mark.asyncio
async def test_alert_is_skipped_when_webhook_url_is_unset(mocker) -> None:
    settings = mocker.patch(f"{MODULE}.get_settings").return_value
    settings.admin_alert_webhook_url = None
    settings.admin_alert_throttle_seconds = 1800
    settings.http_timeout_seconds = 30.0
    redis_fake = _redis_with_storage()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)
    http_factory = mocker.patch(f"{MODULE}.httpx.AsyncClient")

    await notify_provider_limit(ProviderLimitError(provider="p", kind="balance", message="x"))

    assert not http_factory.called, "unset webhook url must not produce HTTP traffic"


@pytest.mark.asyncio
async def test_first_alert_for_provider_kind_is_dispatched(mocker) -> None:
    settings = mocker.patch(f"{MODULE}.get_settings").return_value
    settings.admin_alert_webhook_url = "http://tgbot.test/deliver/tok/123"
    settings.admin_alert_throttle_seconds = 1800
    settings.http_timeout_seconds = 30.0
    redis_fake = _redis_with_storage()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    post_mock = AsyncMock(return_value=response)
    client = MagicMock()
    client.post = post_mock
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(f"{MODULE}.httpx.AsyncClient", return_value=client_cm)

    await notify_provider_limit(
        ProviderLimitError(
            provider="openai/text-embedding-3-small",
            kind="balance",
            message="недостаточно средств",
        )
    )

    assert post_mock.await_count == 1, (
        f"first alert was not POSTed: await_count={post_mock.await_count}"
    )


@pytest.mark.asyncio
async def test_duplicate_alert_within_throttle_window_is_suppressed(mocker) -> None:
    settings = mocker.patch(f"{MODULE}.get_settings").return_value
    settings.admin_alert_webhook_url = "http://tgbot.test/deliver/tok/123"
    settings.admin_alert_throttle_seconds = 1800
    settings.http_timeout_seconds = 30.0
    redis_fake = _redis_with_storage(
        {"alert:provider_limit:openai/text-embedding-3-small:balance": "1"}
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)
    http_factory = mocker.patch(f"{MODULE}.httpx.AsyncClient")

    await notify_provider_limit(
        ProviderLimitError(
            provider="openai/text-embedding-3-small",
            kind="balance",
            message="второй раз",
        )
    )

    assert not http_factory.called, (
        "duplicate alert within throttle window must not produce HTTP traffic"
    )


@pytest.mark.asyncio
async def test_different_kinds_for_same_provider_are_not_throttled_together(mocker) -> None:
    settings = mocker.patch(f"{MODULE}.get_settings").return_value
    settings.admin_alert_webhook_url = "http://tgbot.test/deliver/tok/123"
    settings.admin_alert_throttle_seconds = 1800
    settings.http_timeout_seconds = 30.0
    redis_fake = _redis_with_storage(
        {"alert:provider_limit:openai/text-embedding-3-small:balance": "1"}
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    post_mock = AsyncMock(return_value=response)
    client = MagicMock()
    client.post = post_mock
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(f"{MODULE}.httpx.AsyncClient", return_value=client_cm)

    await notify_provider_limit(
        ProviderLimitError(
            provider="openai/text-embedding-3-small",
            kind="auth",
            message="rotated key",
        )
    )

    assert post_mock.await_count == 1, (
        "throttle key must be per (provider, kind); auth alert was incorrectly suppressed "
        "by a prior balance alert"
    )


@pytest.mark.asyncio
async def test_alert_body_includes_provider_and_kind(mocker) -> None:
    settings = mocker.patch(f"{MODULE}.get_settings").return_value
    settings.admin_alert_webhook_url = "http://tgbot.test/deliver/tok/123"
    settings.admin_alert_throttle_seconds = 1800
    settings.http_timeout_seconds = 30.0
    redis_fake = _redis_with_storage()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    post_mock = AsyncMock(return_value=response)
    client = MagicMock()
    client.post = post_mock
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(f"{MODULE}.httpx.AsyncClient", return_value=client_cm)

    await notify_provider_limit(
        ProviderLimitError(provider="yandex_search", kind="auth", message="role missing")
    )

    payload = post_mock.await_args.kwargs["json"]
    body = payload["body"]
    assert "yandex_search" in body and "auth" in body and "role missing" in body, (
        f"alert body lost provider/kind/message: {body!r}"
    )
