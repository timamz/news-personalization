from pathlib import Path

import aiosqlite

DB_PATH = str(Path.home() / "bot_storage.db")


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
            "SELECT api_key FROM users WHERE telegram_id = ?", (telegram_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None


async def save_api_key(telegram_id: int, api_key: str, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR REPLACE INTO users (telegram_id, api_key) VALUES (?, ?)",
            (telegram_id, api_key),
        )
        await db.commit()
