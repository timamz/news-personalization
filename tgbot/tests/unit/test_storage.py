from pathlib import Path

import aiosqlite
import pytest

from tgbot.language import LanguagePreference
from tgbot.storage import (
    get_api_key,
    get_language_preference,
    get_ui_language,
    init_db,
    save_api_key,
    save_language_preference,
    save_ui_language,
)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.mark.asyncio
async def test_init_db_creates_table(db_path: str):
    await init_db(db_path)
    key = await get_api_key(12345, db_path)
    assert key is None
    assert await get_ui_language(12345, db_path) is None
    assert await get_language_preference(12345, db_path) is None


@pytest.mark.asyncio
async def test_save_and_retrieve_api_key(db_path: str):
    await init_db(db_path)
    await save_api_key(12345, "test-key-abc", db_path)
    key = await get_api_key(12345, db_path)
    assert key == "test-key-abc"


@pytest.mark.asyncio
async def test_overwrite_api_key(db_path: str):
    await init_db(db_path)
    await save_api_key(12345, "old-key", db_path)
    await save_api_key(12345, "new-key", db_path)
    key = await get_api_key(12345, db_path)
    assert key == "new-key"


@pytest.mark.asyncio
async def test_different_users(db_path: str):
    await init_db(db_path)
    await save_api_key(111, "key-a", db_path)
    await save_api_key(222, "key-b", db_path)
    assert await get_api_key(111, db_path) == "key-a"
    assert await get_api_key(222, db_path) == "key-b"


@pytest.mark.asyncio
async def test_save_and_retrieve_language_preference(db_path: str):
    await init_db(db_path)
    await save_language_preference(
        12345,
        "test-key-abc",
        LanguagePreference(mode="fixed", code="ru"),
        db_path,
    )

    preference = await get_language_preference(12345, db_path)

    assert preference == LanguagePreference(mode="fixed", code="ru")


@pytest.mark.asyncio
async def test_save_and_retrieve_ui_language(db_path: str):
    await init_db(db_path)
    await save_ui_language(12345, "test-key-abc", "ru", db_path)

    assert await get_ui_language(12345, db_path) == "ru"


@pytest.mark.asyncio
async def test_save_api_key_preserves_language_preference(db_path: str):
    await init_db(db_path)
    await save_language_preference(
        12345,
        "old-key",
        LanguagePreference(mode="ask", code=None),
        db_path,
    )

    await save_api_key(12345, "new-key", db_path)

    assert await get_api_key(12345, db_path) == "new-key"
    assert await get_language_preference(12345, db_path) == LanguagePreference(
        mode="ask",
        code=None,
    )


@pytest.mark.asyncio
async def test_init_db_migrates_existing_users_table(db_path: str):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE users (telegram_id INTEGER PRIMARY KEY, api_key TEXT NOT NULL)"
        )
        await db.commit()

    await init_db(db_path)
    await save_language_preference(
        12345,
        "test-key-abc",
        LanguagePreference(mode="fixed", code="en"),
        db_path,
    )

    assert await get_language_preference(12345, db_path) == LanguagePreference(
        mode="fixed",
        code="en",
    )
    await save_ui_language(12345, "test-key-abc", "ru", db_path)
    assert await get_ui_language(12345, db_path) == "ru"
