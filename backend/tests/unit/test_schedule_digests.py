import logging
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
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
    topic = f"Topic {uuid.uuid4().hex[:4]}"
    return Subscription(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_spec=f"{topic}. Short digest.",
        delivery_mode=delivery_mode,
        schedule_cron=schedule_cron,
        delivery_webhook_url=f"http://example.com/hook/{uuid.uuid4().hex[:6]}",
        is_active=True,
        created_at=created_at,
        last_digest_scheduled_at=last_digest_scheduled_at,
    )


@pytest.mark.asyncio
async def test_schedule_due_digests_queues_due_digest_and_commits(mocker) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    due = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=datetime(2026, 2, 25, 8, 0, tzinfo=UTC),
    )
    session = _FakeSession([(due, "UTC")])
    mocker.patch.object(
        schedule_digests, "get_task_session", return_value=_FakeSessionFactory(session)
    )
    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    await schedule_digests._schedule_due_digests(now=now)

    send_task.assert_called_once_with(
        "news_service.tasks.deliver_digest.deliver_digest",
        args=[str(due.id)],
    )
    assert session.committed is True, "scheduler did not commit after queuing"


@pytest.mark.asyncio
async def test_schedule_due_digests_returns_correct_counts_across_all_subscription_types(
    mocker,
) -> None:
    now = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    rows = [
        (
            _make_subscription(
                schedule_cron="0 8 * * *",
                created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
                last_digest_scheduled_at=datetime(2026, 2, 25, 8, 0, tzinfo=UTC),
            ),
            "UTC",
        ),
        (
            _make_subscription(
                schedule_cron="0 8 * * *",
                created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
                last_digest_scheduled_at=now,
            ),
            "UTC",
        ),
        (
            _make_subscription(
                schedule_cron="invalid cron",
                created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
            ),
            "UTC",
        ),
        (
            _make_subscription(
                schedule_cron=None,
                created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
            ),
            "UTC",
        ),
        (
            _make_subscription(
                delivery_mode="event",
                schedule_cron="0 8 * * *",
                created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
            ),
            "UTC",
        ),
    ]
    mocker.patch.object(
        schedule_digests,
        "get_task_session",
        return_value=_FakeSessionFactory(_FakeSession(rows)),
    )
    mocker.patch.object(schedule_digests.celery_app, "send_task", MagicMock())

    result = await schedule_digests._schedule_due_digests(now=now)

    assert result == {"checked": 5, "queued": 1, "invalid_cron": 1}, (
        "scheduler did not classify each subscription correctly"
    )


def test_truncate_to_minute_converts_to_utc() -> None:
    result = schedule_digests._truncate_to_minute(datetime(2026, 2, 26, 10, 30, 45))
    assert result == datetime(2026, 2, 26, 10, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_schedule_due_digests_fires_lead_time_before_user_specified_cron(mocker) -> None:
    lead_minutes = 10
    cron_match = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    fire_at = cron_match - timedelta(minutes=lead_minutes)
    one_minute_earlier = fire_at - timedelta(minutes=1)
    sub = _make_subscription(
        schedule_cron="0 8 * * *",
        created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=datetime(2026, 2, 25, 8, 0, tzinfo=UTC),
    )
    mocker.patch.object(
        schedule_digests,
        "get_settings",
        return_value=SimpleNamespace(digest_lead_time_minutes=lead_minutes),
    )
    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    early_session = _FakeSession([(sub, "UTC")])
    mocker.patch.object(
        schedule_digests, "get_task_session", return_value=_FakeSessionFactory(early_session)
    )
    await schedule_digests._schedule_due_digests(now=one_minute_earlier)
    assert send_task.call_count == 0, "digest queued before the lead-time window opened"

    fire_session = _FakeSession([(sub, "UTC")])
    mocker.patch.object(
        schedule_digests, "get_task_session", return_value=_FakeSessionFactory(fire_session)
    )
    await schedule_digests._schedule_due_digests(now=fire_at)
    assert send_task.call_count == 1, "digest was not queued at lead-time before the cron match"
    assert sub.last_digest_scheduled_at == cron_match, (
        "last_digest_scheduled_at must record the cron-matched wall time, not the launch instant"
    )
