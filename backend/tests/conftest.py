import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.engine import make_url

from news_service.agents.event import RecentEventsPreviewDecision

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


@pytest.fixture(autouse=True)
def mock_event_preview_renderer(mocker) -> None:
    async def _preview_renderer(
        *,
        raw_prompt: str,
        target_language: str,
        event_matching_mode: str,
        lookback_days: int,
        candidate_events: list[str],
        recent_notifications: list[str],
    ) -> RecentEventsPreviewDecision:
        del target_language, event_matching_mode, recent_notifications
        selected_ids: list[str] = []
        selected_entries: list[str] = []
        normalized_prompt = raw_prompt.casefold()
        for entry in candidate_events:
            normalized_entry = entry.casefold()
            if (
                ("дробыш" in normalized_prompt or "drobyshev" in normalized_prompt)
                and "дробыш" not in normalized_entry
                and "drobyshev" not in normalized_entry
            ):
                continue
            for line in entry.splitlines():
                if line.startswith("ID: "):
                    selected_ids.append(line.removeprefix("ID: ").strip())
                    selected_entries.append(entry)
                    break
        return RecentEventsPreviewDecision(
            selected_item_ids=selected_ids,
            subject="Recent events you may have missed",
            body=f"Lookback: {lookback_days} days\n\n" + "\n\n".join(selected_entries),
        )

    mocker.patch(
        "news_service.services.event_notifications.render_recent_events_preview",
        new=AsyncMock(side_effect=_preview_renderer),
    )
