import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from news_service.services import event_notifications

logging.disable(logging.CRITICAL)


def _make_item(
    *,
    item_id: str | None = None,
    headline: str = "Станислав Дробышевский выступит с лекцией",
    body: str = "Лекция Станислава Владимировича Дробышевского пройдет в Москве.",
    source: str = "Telegram @fondnauk",
) -> SimpleNamespace:
    resolved_id = item_id or str(uuid.uuid4())
    return SimpleNamespace(
        id=resolved_id,
        source_id=str(uuid.uuid4()),
        source=source,
        headline=headline,
        body=body,
        published_at=datetime(2026, 3, random.randint(1, 28), 12, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 3, random.randint(1, 28), 12, 1, tzinfo=UTC),
        url=f"https://example.com/{resolved_id}",
    )


@pytest.mark.asyncio
async def test_notification_history_entry_from_item_maps_all_fields() -> None:
    source_name = f"Telegram @канал_{uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline="Новая лекция по антропологии",
        body="Профессор расскажет о находках в Денисовой пещере.",
        source=source_name,
    )
    sent_at = datetime(2026, 3, random.randint(1, 28), 14, 0, tzinfo=UTC)

    entry = event_notifications.notification_history_entry_from_item(item, sent_at=sent_at)

    assert entry.title == item.headline, "entry title did not match item headline"
    assert entry.summary == item.body, "entry summary did not match item body"
    assert entry.source == item.source, "entry source did not match item source"
