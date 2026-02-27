from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot import user_registry


@pytest.mark.asyncio
async def test_ensure_api_key_returns_existing_key(monkeypatch):
    backend = SimpleNamespace(register_user=AsyncMock())
    get_api_key = AsyncMock(return_value="existing-key")
    save_api_key = AsyncMock()

    monkeypatch.setattr(user_registry, "get_api_key", get_api_key)
    monkeypatch.setattr(user_registry, "save_api_key", save_api_key)

    result = await user_registry.ensure_api_key(telegram_id=100, backend=backend)

    assert result == "existing-key"
    backend.register_user.assert_not_awaited()
    save_api_key.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_api_key_registers_and_saves_key(monkeypatch):
    backend = SimpleNamespace(register_user=AsyncMock(return_value="new-key"))
    get_api_key = AsyncMock(return_value=None)
    save_api_key = AsyncMock()

    monkeypatch.setattr(user_registry, "get_api_key", get_api_key)
    monkeypatch.setattr(user_registry, "save_api_key", save_api_key)

    result = await user_registry.ensure_api_key(telegram_id=101, backend=backend)

    assert result == "new-key"
    backend.register_user.assert_awaited_once()
    save_api_key.assert_awaited_once_with(101, "new-key")
