import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot import user_registry


def _reset_validation_cache(monkeypatch) -> None:
    monkeypatch.setattr(user_registry, "_VALIDATED_KEYS", set())


@pytest.mark.asyncio
async def test_ensure_api_key_returns_existing_key_after_backend_validates_it(monkeypatch):
    _reset_validation_cache(monkeypatch)
    existing_key = f"existing-{uuid.uuid4().hex}"
    backend = SimpleNamespace(
        register_user=AsyncMock(),
        api_key_is_valid=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(user_registry, "get_api_key", AsyncMock(return_value=existing_key))
    monkeypatch.setattr(user_registry, "save_api_key", AsyncMock())

    result = await user_registry.ensure_api_key(telegram_id=100, backend=backend)

    assert result == existing_key, (
        f"ensure_api_key returned {result!r} but the cached key was {existing_key!r}"
    )
    backend.register_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_api_key_skips_validation_after_first_successful_check(monkeypatch):
    _reset_validation_cache(monkeypatch)
    existing_key = f"existing-{uuid.uuid4().hex}"
    validator = AsyncMock(return_value=True)
    backend = SimpleNamespace(register_user=AsyncMock(), api_key_is_valid=validator)
    monkeypatch.setattr(user_registry, "get_api_key", AsyncMock(return_value=existing_key))
    monkeypatch.setattr(user_registry, "save_api_key", AsyncMock())

    await user_registry.ensure_api_key(telegram_id=100, backend=backend)
    await user_registry.ensure_api_key(telegram_id=100, backend=backend)

    assert validator.await_count == 1, (
        f"api_key_is_valid was called {validator.await_count} times, expected a single validation"
    )


@pytest.mark.asyncio
async def test_ensure_api_key_reregisters_when_backend_rejects_cached_key(monkeypatch):
    _reset_validation_cache(monkeypatch)
    stale_key = f"stale-{uuid.uuid4().hex}"
    fresh_key = f"fresh-{uuid.uuid4().hex}"
    backend = SimpleNamespace(
        register_user=AsyncMock(return_value=fresh_key),
        api_key_is_valid=AsyncMock(return_value=False),
    )
    save_api_key = AsyncMock()
    monkeypatch.setattr(user_registry, "get_api_key", AsyncMock(return_value=stale_key))
    monkeypatch.setattr(user_registry, "save_api_key", save_api_key)

    result = await user_registry.ensure_api_key(telegram_id=777, backend=backend)

    assert result == fresh_key, (
        f"ensure_api_key returned {result!r} but expected the freshly-registered key"
    )
    save_api_key.assert_awaited_once_with(777, fresh_key)


@pytest.mark.asyncio
async def test_ensure_api_key_registers_when_no_key_is_cached(monkeypatch):
    _reset_validation_cache(monkeypatch)
    fresh_key = f"fresh-{uuid.uuid4().hex}"
    backend = SimpleNamespace(
        register_user=AsyncMock(return_value=fresh_key),
        api_key_is_valid=AsyncMock(),
    )
    save_api_key = AsyncMock()
    monkeypatch.setattr(user_registry, "get_api_key", AsyncMock(return_value=None))
    monkeypatch.setattr(user_registry, "save_api_key", save_api_key)

    result = await user_registry.ensure_api_key(telegram_id=101, backend=backend)

    assert result == fresh_key, (
        f"ensure_api_key returned {result!r} but expected the freshly-registered key"
    )
    backend.api_key_is_valid.assert_not_awaited()
