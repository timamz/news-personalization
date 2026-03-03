import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from news_service.models.subscription import Subscription
from news_service.tasks import deliver_digest


class _FakeResult:
    def __init__(self, subscription: Subscription | None) -> None:
        self._subscription = subscription

    def scalar_one_or_none(self) -> Subscription | None:
        return self._subscription


class _FakeSession:
    def __init__(self, subscription: Subscription | None) -> None:
        self._subscription = subscription
        self.committed = False

    async def execute(self, _statement) -> _FakeResult:  # noqa: ANN001
        return _FakeResult(self._subscription)

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
        return False


def _make_subscription(*, delivery_mode: str = "digest") -> Subscription:
    return Subscription(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        raw_prompt="AI updates",
        topics=["artificial intelligence"],
        delivery_mode=delivery_mode,
        schedule_cron="0 8 * * *",
        format_instructions="brief summary",
        delivery_webhook_url="http://frontend.example.test/deliver/1",
        is_active=True,
        created_at=datetime(2026, 2, 26, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=None,
    )


@pytest.mark.asyncio
async def test_deliver_digest_notifies_when_empty_and_requested(mocker) -> None:
    subscription = _make_subscription()
    session = _FakeSession(subscription)
    mocker.patch.object(
        deliver_digest,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))

    channel = AsyncMock()
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=True)

    assert result == {"status": "notified", "reason": "no_new_items"}
    channel.send.assert_awaited_once_with(
        "No new updates right now",
        "No new articles since your last digest. Try again a bit later.",
    )
    assert session.committed is False


@pytest.mark.asyncio
async def test_deliver_digest_skips_when_empty_and_notify_disabled(mocker) -> None:
    subscription = _make_subscription()
    session = _FakeSession(subscription)
    mocker.patch.object(
        deliver_digest,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    get_channel = mocker.patch.object(deliver_digest, "get_delivery_channel")

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {"status": "skipped", "reason": "no_new_items"}
    get_channel.assert_not_called()
    assert session.committed is False


@pytest.mark.asyncio
async def test_deliver_digest_commits_after_successful_delivery(mocker) -> None:
    subscription = _make_subscription()
    session = _FakeSession(subscription)
    mocker.patch.object(
        deliver_digest,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        deliver_digest,
        "generate_digest",
        new=AsyncMock(return_value="digest body"),
    )

    channel = AsyncMock()
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {"status": "delivered", "subscription_id": str(subscription.id)}
    channel.send.assert_awaited_once()
    assert session.committed is True


@pytest.mark.asyncio
async def test_deliver_digest_does_not_commit_on_delivery_failure(mocker) -> None:
    subscription = _make_subscription()
    session = _FakeSession(subscription)
    mocker.patch.object(
        deliver_digest,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        deliver_digest,
        "generate_digest",
        new=AsyncMock(return_value="digest body"),
    )

    channel = AsyncMock()
    channel.send = AsyncMock(side_effect=RuntimeError("delivery failed"))
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    with pytest.raises(RuntimeError, match="delivery failed"):
        await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert session.committed is False


@pytest.mark.asyncio
async def test_deliver_digest_skips_event_subscription(mocker) -> None:
    subscription = _make_subscription(delivery_mode="event")
    session = _FakeSession(subscription)
    mocker.patch.object(
        deliver_digest,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    generate_digest = mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock())

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {"status": "skipped", "reason": "wrong_delivery_mode"}
    generate_digest.assert_not_awaited()
    assert session.committed is False
