"""Per-telegram-user local storage: api_key + current backend conversation id.

The tgbot used to track UI language and subscription language preference here,
but the backend agent now owns everything user-facing. All that remains is the
API key (so we don't register the same telegram_id twice) and the id of the
currently-open backend conversation (so consecutive messages continue the same
chat instead of starting a fresh one each time).
"""

import os
from pathlib import Path

import aiosqlite

DB_PATH = os.getenv("BOT_STORAGE_PATH", str(Path.home() / "bot_storage.db"))


async def init_db(db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "  telegram_id INTEGER PRIMARY KEY,"
            "  api_key TEXT NOT NULL,"
            "  conversation_id TEXT"
            ")"
        )
        columns = await _table_columns(db)
        if "conversation_id" not in columns:
            await db.execute("ALTER TABLE users ADD COLUMN conversation_id TEXT")
        await db.commit()


async def get_api_key(telegram_id: int, db_path: str = DB_PATH) -> str | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT api_key FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None


async def save_api_key(telegram_id: int, api_key: str, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO users (telegram_id, api_key) VALUES (?, ?) "
            "ON CONFLICT(telegram_id) DO UPDATE SET api_key = excluded.api_key",
            (telegram_id, api_key),
        )
        await db.commit()


async def get_conversation_id(telegram_id: int, db_path: str = DB_PATH) -> str | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT conversation_id FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return row[0]


async def save_conversation_id(
    telegram_id: int,
    conversation_id: str,
    db_path: str = DB_PATH,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET conversation_id = ? WHERE telegram_id = ?",
            (conversation_id, telegram_id),
        )
        await db.commit()


async def clear_conversation_id(telegram_id: int, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET conversation_id = NULL WHERE telegram_id = ?",
            (telegram_id,),
        )
        await db.commit()


async def _table_columns(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute("PRAGMA table_info(users)")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}
