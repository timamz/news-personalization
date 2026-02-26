import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from news_service.models.subscription import Subscription
from news_service.tasks import schedule_digests


class _FakeResult:
    def __init__(self, subscriptions: list[Subscription]) -> None:
        self._subscriptions = subscriptions

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Subscription]:
        return self._subscriptions


class _FakeSession:
    def __init__(self, subscriptions: list[Subscription]) -> None:
        self._subscriptions = subscriptions
        self.committed = False

    async def execute(self, _statement) -> _FakeResult:  # noqa: ANN001
        return _FakeResult(self._subscriptions)

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
    schedule_cron: str,
    created_at: datetime,
    last_digest_scheduled_at: datetime | None = None,
) -> Subscription:
    return Subscription(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        raw_prompt="AI news",
        topics=["ai"],
        schedule_cron=schedule_cron,
        format_instructions="brief summary",
        delivery_webhook_url="http://example.com/hook",
        is_active=True,
        created_at=created_at,
        last_digest_scheduled_at=last_digest_scheduled_at,
    )


@pytest.mark.asyncio
async def test_schedule_due_digests_queues_only_due_subscriptions(mocker):
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

    session = _FakeSession([due, not_due, invalid])
    mocker.patch.object(
        schedule_digests,
        "async_session_factory",
        return_value=_FakeSessionFactory(session),
    )

    send_task = MagicMock()
    mocker.patch.object(schedule_digests.celery_app, "send_task", send_task)

    result = await schedule_digests._schedule_due_digests(now=now)

    send_task.assert_called_once_with(
        "news_service.tasks.deliver_digest.deliver_digest",
        args=[str(due.id)],
    )
    assert due.last_digest_scheduled_at == now
    assert not_due.last_digest_scheduled_at == now
    assert invalid.last_digest_scheduled_at is None
    assert result == {"checked": 3, "queued": 1, "invalid_cron": 1}
    assert session.committed is True


def test_truncate_to_minute_converts_to_utc():
    local_time = datetime(2026, 2, 26, 10, 30, 45)
    assert schedule_digests._truncate_to_minute(local_time) == datetime(
        2026, 2, 26, 10, 30, tzinfo=UTC
    )
