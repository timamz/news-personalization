from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.event import EventMatchDecision, NotificationDuplicateDecision
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
    event_title: str = "Новая лекция",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        feed_id="feed-1",
        source="Telegram @fondnauk",
        headline="Станислав Дробышевский выступит с лекцией",
        body="Лекция Станислава Владимировича Дробышевского пройдет в Москве.",
        event_title=event_title,
        event_summary="Лекция Станислава Дробышевского.",
        event_starts_at=datetime(2026, 3, 4, 16, 0, tzinfo=UTC),
        published_at=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 4, 12, 1, tzinfo=UTC),
        url=f"https://example.com/{item_id}",
    )


def _make_subscription(*, strict: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        id="sub-1",
        user_id="user-1",
        raw_prompt="Only lectures by Stanislav Drobyshevsky himself",
        event_matching_mode="strict_with_prefilter" if strict else "basic",
    )


@pytest.mark.asyncio
async def test_subscription_matches_event_skips_judge_for_basic_mode(mocker) -> None:
    item = _make_item()
    subscription = _make_subscription(strict=False)

    judge = mocker.patch.object(event_notifications, "judge_event_match", new=AsyncMock())

    result = await event_notifications.subscription_matches_event(subscription, item)

    assert result is True
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_matches_event_uses_prompt_judge(mocker) -> None:
    item = _make_item()
    subscription = _make_subscription()

    judge = mocker.patch.object(
        event_notifications,
        "judge_event_match",
        new=AsyncMock(
            return_value=EventMatchDecision(
                matches=True,
                reason="The post clearly announces the requested lecturer.",
            )
        ),
    )

    result = await event_notifications.subscription_matches_event(subscription, item)

    assert result is True
    judge.assert_awaited_once()


@pytest.mark.asyncio
async def test_notification_was_already_shown_skips_when_history_is_empty(mocker) -> None:
    item = _make_item()
    judge = mocker.patch.object(
        event_notifications,
        "judge_notification_duplicate",
        new=AsyncMock(),
    )

    result = await event_notifications.notification_was_already_shown(item, [])

    assert result is False
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_recent_subscription_events_uses_shared_duplicate_judge(mocker) -> None:
    items = [
        _make_item(item_id="news-1", event_title="Лекция Дробышевского"),
        _make_item(item_id="news-2", event_title="Напоминание о лекции Дробышевского"),
        _make_item(item_id="news-3", event_title="Другая лекция Дробышевского"),
    ]
    items[2].event_starts_at = datetime(2026, 3, 5, 16, 0, tzinfo=UTC)
    subscription = _make_subscription(strict=False)
    history_item = _make_item(item_id="history-1", event_title="Старое событие")
    session = _FakeSession(
        [
            items,
            [(datetime(2026, 3, 1, 12, 0, tzinfo=UTC), history_item)],
        ]
    )

    duplicate_judge = mocker.patch.object(
        event_notifications,
        "judge_notification_duplicate",
        new=AsyncMock(
            side_effect=[
                NotificationDuplicateDecision(
                    already_notified=False,
                    reason="The first lecture is new.",
                ),
                NotificationDuplicateDecision(
                    already_notified=True,
                    reason="This is the same lecture reminder.",
                ),
                NotificationDuplicateDecision(
                    already_notified=False,
                    reason="This is a different lecture.",
                ),
            ]
        ),
    )

    result = await event_notifications.list_recent_subscription_events(session, subscription)

    assert [item.id for item in result] == ["news-1", "news-3"]
    assert duplicate_judge.await_count == 3


@pytest.mark.asyncio
async def test_list_recent_subscription_events_filters_strict_matches_before_duplicate_check(
    mocker,
) -> None:
    items = [
        _make_item(item_id="news-1"),
        _make_item(item_id="news-2", event_title="Другая лекция"),
    ]
    subscription = _make_subscription(strict=True)
    session = _FakeSession(
        [
            items,
            [],
        ]
    )

    match_judge = mocker.patch.object(
        event_notifications,
        "judge_event_match",
        new=AsyncMock(
            side_effect=[
                EventMatchDecision(matches=True, reason="match"),
                EventMatchDecision(matches=False, reason="no match"),
            ]
        ),
    )
    duplicate_judge = mocker.patch.object(
        event_notifications,
        "judge_notification_duplicate",
        new=AsyncMock(
            return_value=NotificationDuplicateDecision(
                already_notified=False,
                reason="No duplicates.",
            )
        ),
    )

    result = await event_notifications.list_recent_subscription_events(session, subscription)

    assert [item.id for item in result] == ["news-1"]
    assert match_judge.await_count == 2
    assert duplicate_judge.await_count == 0


@pytest.mark.asyncio
async def test_list_recent_subscription_events_loads_history_for_current_subscription(
    mocker,
) -> None:
    item = _make_item()
    subscription = _make_subscription(strict=False)
    session = _FakeSession([[item]])

    history_loader = mocker.patch.object(
        event_notifications,
        "load_recent_notification_history",
        new=AsyncMock(return_value=[]),
    )
    duplicate_judge = mocker.patch.object(
        event_notifications,
        "judge_notification_duplicate",
        new=AsyncMock(
            return_value=NotificationDuplicateDecision(
                already_notified=False,
                reason="This is new for the subscription.",
            )
        ),
    )

    result = await event_notifications.list_recent_subscription_events(session, subscription)

    assert [matched.id for matched in result] == ["news-1"]
    history_loader.assert_awaited_once_with(session, subscription.id)
    duplicate_judge.assert_not_awaited()
