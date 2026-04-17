"""Tests for the per-telegram-user local sqlite storage."""

import os
import tempfile

import pytest

from tgbot.storage import (
    clear_conversation_id,
    get_api_key,
    get_conversation_id,
    init_db,
    save_api_key,
    save_conversation_id,
)


def _temp_db() -> str:
    handle, path = tempfile.mkstemp(suffix=".sqlite", dir=tempfile.gettempdir())
    os.close(handle)
    return path


@pytest.mark.asyncio
async def test_save_api_key_returns_same_value_on_fetch() -> None:
    db_path = _temp_db()
    await init_db(db_path)
    telegram_id = 424242
    api_key = "tgbot-test-key-abc"
    await save_api_key(telegram_id, api_key, db_path=db_path)
    assert await get_api_key(telegram_id, db_path=db_path) == api_key, (
        "get_api_key did not return the value that was just saved"
    )


@pytest.mark.asyncio
async def test_get_api_key_returns_none_for_unknown_user() -> None:
    db_path = _temp_db()
    await init_db(db_path)
    assert await get_api_key(99999999, db_path=db_path) is None, (
        "get_api_key did not return None for an unregistered telegram_id"
    )


@pytest.mark.asyncio
async def test_save_api_key_overwrites_previous_key() -> None:
    db_path = _temp_db()
    await init_db(db_path)
    telegram_id = 111222
    await save_api_key(telegram_id, "old-key", db_path=db_path)
    await save_api_key(telegram_id, "new-key", db_path=db_path)
    assert await get_api_key(telegram_id, db_path=db_path) == "new-key", (
        "second save_api_key did not overwrite the earlier value"
    )


@pytest.mark.asyncio
async def test_save_conversation_id_persists_between_calls() -> None:
    db_path = _temp_db()
    await init_db(db_path)
    telegram_id = 333444
    await save_api_key(telegram_id, "key", db_path=db_path)
    conversation_id = "conv-xyz-42"
    await save_conversation_id(telegram_id, conversation_id, db_path=db_path)
    assert await get_conversation_id(telegram_id, db_path=db_path) == conversation_id, (
        "get_conversation_id did not return the persisted value"
    )


@pytest.mark.asyncio
async def test_clear_conversation_id_removes_previously_saved_value() -> None:
    db_path = _temp_db()
    await init_db(db_path)
    telegram_id = 555666
    await save_api_key(telegram_id, "key", db_path=db_path)
    await save_conversation_id(telegram_id, "c1", db_path=db_path)
    await clear_conversation_id(telegram_id, db_path=db_path)
    assert await get_conversation_id(telegram_id, db_path=db_path) is None, (
        "clear_conversation_id did not remove the stored conversation id"
    )


@pytest.mark.asyncio
async def test_get_conversation_id_returns_none_for_unknown_user() -> None:
    db_path = _temp_db()
    await init_db(db_path)
    assert await get_conversation_id(12345, db_path=db_path) is None, (
        "get_conversation_id did not return None for an unregistered telegram_id"
    )
