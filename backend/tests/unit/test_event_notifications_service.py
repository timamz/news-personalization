from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.services import event_notifications


class _FakeResult:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[object]:
        return self._values


class _FakeSession:
    def __init__(self, results: list[list[object]]) -> None:
        self._results = [_FakeResult(values) for values in results]

    async def execute(self, _statement) -> _FakeResult:  # noqa: ANN001
        return self._results.pop(0)


def _make_item(
    *,
    item_id: str = "news-1",
    headline: str = "Станислав Дробышевский выступит с лекцией",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        feed_id="feed-1",
        source="Telegram @fondnauk",
        headline=headline,
        body="Лекция Станислава Владимировича Дробышевского пройдет в Москве.",
        published_at=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 4, 12, 1, tzinfo=UTC),
        url=f"https://example.com/{item_id}",
    )


def _make_subscription() -> SimpleNamespace:
    return SimpleNamespace(
        id="sub-1",
        user_id="user-1",
        raw_prompt="Only lectures by Stanislav Drobyshevsky himself",
        canonical_prompt="Only lectures by Stanislav Drobyshevsky himself",
        digest_language="en",
    )


@pytest.mark.asyncio
async def test_notification_history_entry_from_item_uses_headline_and_body() -> None:
    item = _make_item()
    entry = event_notifications.notification_history_entry_from_item(
        item, sent_at=datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    )

    assert entry.title == item.headline
    assert entry.summary == item.body[:200]
    assert entry.source == item.source


@pytest.mark.asyncio
async def test_list_recent_subscription_events_filters_duplicates() -> None:
    items = [
        _make_item(item_id="news-1", headline="Лекция Дробышевского"),
        _make_item(item_id="news-2", headline="Лекция Дробышевского"),
        _make_item(item_id="news-3", headline="Другая лекция Дробышевского"),
    ]
    subscription = _make_subscription()
    session = _FakeSession(
        [
            items,
            [],
        ]
    )

    result = await event_notifications.list_recent_subscription_events(session, subscription)

    result_ids = [item.id for item in result]
    assert "news-1" in result_ids
    assert "news-3" in result_ids
    assert "news-2" not in result_ids


@pytest.mark.asyncio
async def test_list_recent_subscription_events_loads_history_for_current_subscription(
    mocker,
) -> None:
    item = _make_item()
    subscription = _make_subscription()
    session = _FakeSession([[item]])

    history_loader = mocker.patch.object(
        event_notifications,
        "load_recent_notification_history",
        new=AsyncMock(return_value=[]),
    )

    result = await event_notifications.list_recent_subscription_events(session, subscription)

    assert [matched.id for matched in result] == ["news-1"]
    history_loader.assert_awaited_once_with(session, subscription.id)


@pytest.mark.asyncio
async def test_build_recent_events_preview_returns_single_preview() -> None:
    item_one = _make_item(item_id="news-1", headline="Первая лекция")
    item_two = _make_item(item_id="news-2", headline="Вторая лекция")
    item_two.url = "https://example.com/news-2"

    preview = await event_notifications.build_recent_events_preview(
        "ru",
        [item_one, item_two],
        lookback_days=7,
    )

    assert preview.news_item_ids == ["news-1", "news-2"]
    assert preview.subject == "Что вы могли пропустить"
    assert "Вторая лекция" in preview.body
    assert "Telegram @fondnauk" not in preview.body
    assert "https://example.com/news-2" in preview.body
