"""Per-telegram-user local storage: just the backend API key.

The backend owns the conversation (one persistent thread per user,
server-side), so the tgbot no longer tracks any conversation id. All
that lives here is the mapping from telegram_id to the API key issued
on first /start.
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
            "  api_key TEXT NOT NULL"
            ")"
        )
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
