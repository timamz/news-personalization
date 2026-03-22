import logging
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from news_service.models.subscription import Subscription
from news_service.tasks import schedule_digests

logging.disable(logging.CRITICAL)


class _FakeResult:
    def __init__(self, rows: list[tuple[Subscription, str | None]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[Subscription, str | None]]:
        return self._rows


class _FakeSession:
    def __init__(self, rows: list[tuple[Subscription, str | None]]) -> None:
        self._rows = rows
        self.committed = False

    async def execute(self, _statement) -> _FakeResult:
        return _FakeResult(self._rows)

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
        return False


def _make_subscription(
    *,
    delivery_mode: str = "digest",
    schedule_cron: str | None,
    created_at: datetime,
    last_digest_scheduled_at: datetime | None = None,
) -> Subscription:
    return Subscription(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        raw_prompt=f"Новости ИИ {uuid.uuid4().hex[:4]}",
        prompt_summary=f"Дайджест {uuid.uuid4().hex[:4]}",
        delivery_mode=delivery_mode,
        schedule_cron=schedule_cron,
        format_instructions="краткое описание",
        delivery_webhook_url=f"http://example.com/hook/{uuid.uuid4().hex[:6]}",
        is_active=True,
        created_at=created_at,
        last_digest_scheduled_at=last_digest_scheduled_at,
    )


@pytest.mark.asyncio
async def test_schedule_due_digests_queues_due_subscription(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    due = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=datetime(2026, 2, 25, 8, 0, tzinfo=UTC),
    )
    session = _FakeSession([(due, "UTC")])
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    await schedule_digests._schedule_due_digests(now=now)

    send_task.assert_called_once_with(
        "news_service.tasks.deliver_digest.deliver_digest",
        args=[str(due.id)],
    )
    assert send_task.call_count == 1, "scheduler did not queue exactly one due subscription"


@pytest.mark.asyncio
async def test_schedule_due_digests_does_not_queue_not_due_subscription(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    not_due = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=now,
    )
    session = _FakeSession([(not_due, "UTC")])
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    await schedule_digests._schedule_due_digests(now=now)

    assert send_task.call_count == 0, "scheduler queued a not-due subscription"


@pytest.mark.asyncio
async def test_schedule_due_digests_skips_invalid_cron(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    invalid = _make_subscription(
        schedule_cron="invalid cron",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
    )
    session = _FakeSession([(invalid, "UTC")])
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    await schedule_digests._schedule_due_digests(now=now)

    assert send_task.call_count == 0, "scheduler queued a subscription with invalid cron"


@pytest.mark.asyncio
async def test_schedule_due_digests_skips_event_subscription(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    event_sub = _make_subscription(
        delivery_mode="event",
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
    )
    session = _FakeSession([(event_sub, "UTC")])
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    await schedule_digests._schedule_due_digests(now=now)

    assert send_task.call_count == 0, "scheduler queued an event-mode subscription"


@pytest.mark.asyncio
async def test_schedule_due_digests_returns_correct_counts(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    due = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=datetime(2026, 2, 25, 8, 0, tzinfo=UTC),
    )
    not_due = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=now,
    )
    invalid = _make_subscription(
        schedule_cron="invalid cron",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
    )
    manual_only = _make_subscription(
        schedule_cron=None,
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
    )
    event_sub = _make_subscription(
        delivery_mode="event",
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
    )

    session = _FakeSession(
        [
            (due, "UTC"),
            (not_due, "UTC"),
            (invalid, "UTC"),
            (manual_only, "UTC"),
            (event_sub, "UTC"),
        ]
    )
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(schedule_digests.celery_app, "send_task", MagicMock())

    result = await schedule_digests._schedule_due_digests(now=now)

    assert result == {"checked": 5, "queued": 1, "invalid_cron": 1}, (
        "scheduler did not return correct counts"
    )


@pytest.mark.asyncio
async def test_schedule_due_digests_commits_session(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    due = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=datetime(2026, 2, 25, 8, 0, tzinfo=UTC),
    )
    session = _FakeSession([(due, "UTC")])
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(schedule_digests.celery_app, "send_task", MagicMock())

    await schedule_digests._schedule_due_digests(now=now)

    assert session.committed is True, "scheduler did not commit the session"


def test_truncate_to_minute_converts_to_utc() -> None:
    local_time = datetime(2026, 2, 26, 10, 30, 45)
    result = schedule_digests._truncate_to_minute(local_time)
    assert result == datetime(2026, 2, 26, 10, 30, tzinfo=UTC), (
        "truncate_to_minute did not strip seconds and set UTC"
    )
