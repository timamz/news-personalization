from pathlib import Path

import pytest

from tgbot.storage import get_api_key, init_db, save_api_key


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.mark.asyncio
async def test_init_db_creates_table(db_path: str):
    await init_db(db_path)
    key = await get_api_key(12345, db_path)
    assert key is None


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
