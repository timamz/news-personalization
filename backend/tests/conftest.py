import os
from pathlib import Path

from sqlalchemy.engine import make_url

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")


def _read_database_url_from_env_file() -> str | None:
    env_file = Path(".env")
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("DATABASE_URL="):
            return line.split("=", maxsplit=1)[1].strip()
    return None


base_database_url = os.environ.get("DATABASE_URL") or _read_database_url_from_env_file()
if base_database_url:
    parsed_url = make_url(base_database_url)
    database_name = parsed_url.database
    if database_name and not database_name.endswith("_test"):
        parsed_url = parsed_url.set(database=f"{database_name}_test")
    os.environ.setdefault("DATABASE_URL", parsed_url.render_as_string(hide_password=False))
else:
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://news:news@localhost:5432/news_test")
