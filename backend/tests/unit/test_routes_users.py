"""Tests for the user profile routes."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.api.routes_users import update_user_profile
from news_service.schemas.user import UserUpdate

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_update_user_profile_persists_delivery_webhook_without_touching_timezone() -> None:
    user = SimpleNamespace(
        timezone="Europe/Moscow",
        delivery_webhook_url=None,
    )
    session = AsyncMock()
    payload = UserUpdate(delivery_webhook_url="http://tgbot.test/deliver/abc/123")

    result = await update_user_profile(payload, user=user, session=session)

    assert result.delivery_webhook_url == payload.delivery_webhook_url, (
        "update_user_profile did not persist the delivery webhook URL"
    )
    assert result.timezone == "Europe/Moscow", (
        "update_user_profile should not change timezone when none was provided"
    )
    session.commit.assert_awaited_once()
