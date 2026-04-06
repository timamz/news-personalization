import logging
import random
import uuid
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

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_get_api_key_returns_none_for_unknown_user(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    await init_db(db_path)

    result = await get_api_key(random.randint(10000, 99999), db_path)

    assert result is None, "get_api_key did not return None for an unknown user"


@pytest.mark.asyncio
async def test_get_ui_language_returns_none_for_unknown_user(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    await init_db(db_path)

    result = await get_ui_language(random.randint(10000, 99999), db_path)

    assert result is None, "get_ui_language did not return None for an unknown user"


@pytest.mark.asyncio
async def test_get_language_preference_returns_none_for_unknown_user(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    await init_db(db_path)

    result = await get_language_preference(random.randint(10000, 99999), db_path)

    assert result is None, "get_language_preference did not return None for an unknown user"


@pytest.mark.asyncio
async def test_save_and_retrieve_api_key(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid = random.randint(10000, 99999)
    key = f"\u043a\u043b\u044e\u0447-{uuid.uuid4().hex}"
    await init_db(db_path)
    await save_api_key(tid, key, db_path)

    result = await get_api_key(tid, db_path)

    assert result == key, "save_api_key/get_api_key roundtrip did not preserve the key"


@pytest.mark.asyncio
async def test_overwrite_api_key_keeps_latest(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid = random.randint(10000, 99999)
    old_key = f"old-{uuid.uuid4().hex}"
    new_key = f"new-{uuid.uuid4().hex}"
    await init_db(db_path)
    await save_api_key(tid, old_key, db_path)
    await save_api_key(tid, new_key, db_path)

    result = await get_api_key(tid, db_path)

    assert result == new_key, "overwriting api_key did not keep the latest value"


@pytest.mark.asyncio
async def test_different_users_have_separate_keys(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid_a = random.randint(10000, 49999)
    tid_b = random.randint(50000, 99999)
    key_a = f"\u043a\u043b\u044e\u0447-A-{uuid.uuid4().hex}"
    key_b = f"\u043a\u043b\u044e\u0447-B-{uuid.uuid4().hex}"
    await init_db(db_path)
    await save_api_key(tid_a, key_a, db_path)
    await save_api_key(tid_b, key_b, db_path)

    result_a = await get_api_key(tid_a, db_path)
    result_b = await get_api_key(tid_b, db_path)

    assert result_a == key_a, "different users did not get separate api keys (user A)"
    assert result_b == key_b, "different users did not get separate api keys (user B)"


@pytest.mark.asyncio
async def test_save_and_retrieve_language_preference(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid = random.randint(10000, 99999)
    api_key = f"\u043a\u043b\u044e\u0447-{uuid.uuid4().hex}"
    await init_db(db_path)
    await save_language_preference(
        tid, api_key, LanguagePreference(mode="fixed", code="ru"), db_path
    )

    result = await get_language_preference(tid, db_path)

    assert result == LanguagePreference(mode="fixed", code="ru"), (
        "save_language_preference/get_language_preference roundtrip did not preserve the preference"
    )


@pytest.mark.asyncio
async def test_save_and_retrieve_ui_language(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid = random.randint(10000, 99999)
    api_key = f"key-{uuid.uuid4().hex}"
    lang = random.choice(["en", "ru"])
    await init_db(db_path)
    await save_ui_language(tid, api_key, lang, db_path)

    result = await get_ui_language(tid, db_path)

    assert result == lang, (
        "save_ui_language/get_ui_language roundtrip did not preserve the language"
    )


@pytest.mark.asyncio
async def test_save_api_key_preserves_language_preference(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid = random.randint(10000, 99999)
    old_key = f"old-{uuid.uuid4().hex}"
    new_key = f"new-{uuid.uuid4().hex}"
    await init_db(db_path)
    await save_language_preference(tid, old_key, LanguagePreference(mode="ask", code=None), db_path)
    await save_api_key(tid, new_key, db_path)

    result = await get_language_preference(tid, db_path)

    assert result == LanguagePreference(mode="ask", code=None), (
        "save_api_key overwrote the existing language preference"
    )


@pytest.mark.asyncio
async def test_save_api_key_preserves_language_preference_and_updates_key(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    tid = random.randint(10000, 99999)
    old_key = f"old-{uuid.uuid4().hex}"
    new_key = f"new-{uuid.uuid4().hex}"
    await init_db(db_path)
    await save_language_preference(tid, old_key, LanguagePreference(mode="ask", code=None), db_path)
    await save_api_key(tid, new_key, db_path)

    result = await get_api_key(tid, db_path)

    assert result == new_key, "save_api_key did not update the key after preserving preference"


@pytest.mark.asyncio
async def test_init_db_migrates_existing_users_table_adds_language_preference(
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE users (telegram_id INTEGER PRIMARY KEY, api_key TEXT NOT NULL)"
        )
        await db.commit()

    await init_db(db_path)
    tid = random.randint(10000, 99999)
    api_key = f"\u043a\u043b\u044e\u0447-{uuid.uuid4().hex}"
    await save_language_preference(
        tid, api_key, LanguagePreference(mode="fixed", code="en"), db_path
    )

    result = await get_language_preference(tid, db_path)

    assert result == LanguagePreference(mode="fixed", code="en"), (
        "init_db migration did not add language_preference columns"
    )


@pytest.mark.asyncio
async def test_init_db_migrates_existing_users_table_adds_ui_language(tmp_path: Path) -> None:
    db_path = str(tmp_path / f"test-{uuid.uuid4().hex}.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE users (telegram_id INTEGER PRIMARY KEY, api_key TEXT NOT NULL)"
        )
        await db.commit()

    await init_db(db_path)
    tid = random.randint(10000, 99999)
    api_key = f"key-{uuid.uuid4().hex}"
    await save_ui_language(tid, api_key, "ru", db_path)

    result = await get_ui_language(tid, db_path)

    assert result == "ru", "init_db migration did not add ui_language column"
