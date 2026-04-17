import logging
import random
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from news_service.models.subscription import Subscription
from news_service.tasks import deliver_digest

logging.disable(logging.CRITICAL)


class _FakeResult:
    def __init__(self, subscription: Subscription | None) -> None:
        self._subscription = subscription

    def scalar_one_or_none(self) -> Subscription | None:
        return self._subscription


class _FakeSession:
    def __init__(self, subscription: Subscription | None) -> None:
        self._subscription = subscription
        self.committed = False

    async def execute(self, _statement) -> _FakeResult:
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


def _make_subscription(delivery_mode: str, webhook_url: str) -> Subscription:
    topic = f"Новости об ИИ {uuid.uuid4().hex[:6]}"
    return Subscription(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_spec=f"## Topic\n{topic}\n\n## Preferences\nкраткая сводка",
        delivery_mode=delivery_mode,
        schedule_cron=f"{random.randint(0, 59)} {random.randint(0, 23)} * * *",
        delivery_webhook_url=webhook_url,
        is_active=True,
        created_at=datetime(2026, 2, 26, 8, 0, tzinfo=UTC),
        last_digest_scheduled_at=None,
    )


def _patch_session(mocker, subscription: Subscription) -> _FakeSession:
    session = _FakeSession(subscription)
    mocker.patch.object(
        deliver_digest, "get_task_session", return_value=_FakeSessionFactory(session)
    )
    return session


@pytest.mark.asyncio
async def test_deliver_digest_returns_notified_status_when_empty_and_notify_requested(
    mocker,
) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    channel = AsyncMock()
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=True)

    assert result == {
        "status": "notified",
        "reason": "no_new_items",
    }, "deliver_digest did not return notified status when empty and notify requested"


@pytest.mark.asyncio
async def test_deliver_digest_sends_empty_notification_via_channel_when_empty(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    channel = AsyncMock()
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    await deliver_digest._deliver_digest(subscription.id, notify_if_empty=True)

    assert channel.send.await_count == 1, (
        "deliver_digest did not send empty notification via channel"
    )


@pytest.mark.asyncio
async def test_deliver_digest_returns_skipped_when_empty_and_notify_disabled(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    mocker.patch.object(deliver_digest, "get_delivery_channel")

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {
        "status": "skipped",
        "reason": "no_new_items",
    }, "deliver_digest did not return skipped when empty and notify disabled"


@pytest.mark.asyncio
async def test_deliver_digest_does_not_call_channel_when_empty_and_notify_disabled(
    mocker,
) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    get_channel = mocker.patch.object(deliver_digest, "get_delivery_channel")

    await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert not get_channel.called, (
        "deliver_digest called get_delivery_channel when empty and notify disabled"
    )


@pytest.mark.asyncio
async def test_deliver_digest_returns_delivered_status_on_success(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    _patch_session(mocker, subscription)
    digest_body = f"Дайджест новостей {uuid.uuid4().hex[:8]}"
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=digest_body))
    channel = AsyncMock()
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {
        "status": "delivered",
        "subscription_id": str(subscription.id),
    }, "deliver_digest did not return delivered status on successful delivery"


@pytest.mark.asyncio
async def test_deliver_digest_commits_session_after_successful_delivery(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    session = _patch_session(mocker, subscription)
    mocker.patch.object(
        deliver_digest,
        "generate_digest",
        new=AsyncMock(return_value=f"Дайджест {uuid.uuid4().hex[:6]}"),
    )
    channel = AsyncMock()
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert session.committed is True, (
        "deliver_digest did not commit session after successful delivery"
    )


@pytest.mark.asyncio
async def test_deliver_digest_does_not_commit_on_delivery_failure(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    session = _patch_session(mocker, subscription)
    mocker.patch.object(
        deliver_digest,
        "generate_digest",
        new=AsyncMock(return_value=f"Дайджест {uuid.uuid4().hex[:6]}"),
    )
    channel = AsyncMock()
    channel.send = AsyncMock(side_effect=RuntimeError())
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=channel)

    with pytest.raises(RuntimeError):
        await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert session.committed is False, "deliver_digest committed session despite delivery failure"


@pytest.mark.asyncio
async def test_deliver_digest_skips_event_subscription(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("event", webhook)
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock())

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {
        "status": "skipped",
        "reason": "wrong_delivery_mode",
    }, "deliver_digest did not skip event subscription"


@pytest.mark.asyncio
async def test_deliver_digest_does_not_call_generate_digest_for_event_subscription(
    mocker,
) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("event", webhook)
    _patch_session(mocker, subscription)
    generate_digest = mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock())

    await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert generate_digest.await_count == 0, (
        "deliver_digest called generate_digest for event subscription"
    )


@pytest.mark.asyncio
async def test_deliver_digest_does_not_commit_when_empty(mocker) -> None:
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription("digest", webhook)
    session = _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    mocker.patch.object(deliver_digest, "get_delivery_channel", return_value=AsyncMock())

    await deliver_digest._deliver_digest(subscription.id, notify_if_empty=True)

    assert session.committed is False, "deliver_digest committed session when digest was empty"
