import os
from pathlib import Path

import aiosqlite

from tgbot.language import LanguagePreference, UILanguage, normalize_language_code

DB_PATH = os.getenv("BOT_STORAGE_PATH", str(Path.home() / "bot_storage.db"))


async def init_db(db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "  telegram_id INTEGER PRIMARY KEY,"
            "  api_key TEXT NOT NULL,"
            "  ui_language TEXT,"
            "  language_mode TEXT,"
            "  language_code TEXT"
            ")"
        )
        columns = await _table_columns(db)
        if "ui_language" not in columns:
            await db.execute("ALTER TABLE users ADD COLUMN ui_language TEXT")
        if "language_mode" not in columns:
            await db.execute("ALTER TABLE users ADD COLUMN language_mode TEXT")
        if "language_code" not in columns:
            await db.execute("ALTER TABLE users ADD COLUMN language_code TEXT")
        await db.commit()


async def get_api_key(telegram_id: int, db_path: str = DB_PATH) -> str | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT api_key FROM users WHERE telegram_id = ?", (telegram_id,))
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


async def get_language_preference(
    telegram_id: int,
    db_path: str = DB_PATH,
) -> LanguagePreference | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT language_mode, language_code FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None

        language_mode = row[0]
        language_code = normalize_language_code(row[1])
        if language_mode == "ask":
            return LanguagePreference(mode="ask", code=None)
        if language_mode == "fixed" and language_code is not None:
            return LanguagePreference(mode="fixed", code=language_code)
        return None


async def get_ui_language(telegram_id: int, db_path: str = DB_PATH) -> UILanguage | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT ui_language FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        return normalize_language_code(row[0]) if row else None


async def save_ui_language(
    telegram_id: int,
    api_key: str,
    ui_language: UILanguage,
    db_path: str = DB_PATH,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO users (telegram_id, api_key, ui_language) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(telegram_id) DO UPDATE SET "
            "api_key = excluded.api_key, "
            "ui_language = excluded.ui_language",
            (telegram_id, api_key, ui_language),
        )
        await db.commit()


async def save_language_preference(
    telegram_id: int,
    api_key: str,
    preference: LanguagePreference,
    db_path: str = DB_PATH,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO users (telegram_id, api_key, language_mode, language_code) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(telegram_id) DO UPDATE SET "
            "api_key = excluded.api_key, "
            "language_mode = excluded.language_mode, "
            "language_code = excluded.language_code",
            (telegram_id, api_key, preference.mode, preference.code),
        )
        await db.commit()


async def _table_columns(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute("PRAGMA table_info(users)")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}
