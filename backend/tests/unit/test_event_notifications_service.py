from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from news_service.services import event_notifications


def _make_item(
    *,
    item_id: str = "news-1",
    headline: str = "Станислав Дробышевский выступит с лекцией",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        source_id="feed-1",
        source="Telegram @fondnauk",
        headline=headline,
        body="Лекция Станислава Владимировича Дробышевского пройдет в Москве.",
        published_at=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 4, 12, 1, tzinfo=UTC),
        url=f"https://example.com/{item_id}",
    )


@pytest.mark.asyncio
async def test_notification_history_entry_from_item_uses_headline_and_body() -> None:
    item = _make_item()
    entry = event_notifications.notification_history_entry_from_item(
        item, sent_at=datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    )

    assert entry.title == item.headline
    assert entry.summary == item.body
    assert entry.source == item.source
