import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
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


def _make_subscription(delivery_mode: str) -> Subscription:
    return Subscription(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_spec=f"Topic {uuid.uuid4().hex[:4]}.",
        delivery_mode=delivery_mode,
        schedule_cron=f"{random.randint(0, 59)} {random.randint(0, 23)} * * *",
        delivery_webhook_url=f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver",
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
async def test_deliver_digest_delivers_and_commits_on_success(mocker) -> None:
    subscription = _make_subscription("digest")
    session = _patch_session(mocker, subscription)
    mocker.patch.object(
        deliver_digest, "generate_digest", new=AsyncMock(return_value=f"D {uuid.uuid4().hex[:6]}")
    )
    deliver_mock = mocker.patch.object(deliver_digest, "deliver", new=AsyncMock())

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {"status": "delivered", "subscription_id": str(subscription.id)}
    assert session.committed is True and deliver_mock.await_count == 1, (
        "successful digest delivery did not commit or did not call the webhook"
    )


@pytest.mark.asyncio
async def test_deliver_digest_does_not_commit_on_webhook_failure(mocker) -> None:
    subscription = _make_subscription("digest")
    session = _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value="digest"))
    mocker.patch.object(deliver_digest, "deliver", new=AsyncMock(side_effect=RuntimeError()))

    with pytest.raises(RuntimeError):
        await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert session.committed is False


@pytest.mark.asyncio
async def test_deliver_digest_sends_notification_when_empty_and_notify_requested(mocker) -> None:
    subscription = _make_subscription("digest")
    session = _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    deliver_mock = mocker.patch.object(deliver_digest, "deliver", new=AsyncMock())

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=True)

    assert result == {"status": "notified", "reason": "no_new_items"}
    assert deliver_mock.await_count == 1 and session.committed is False, (
        "empty-notify path did not dispatch exactly one webhook or wrongly committed"
    )


@pytest.mark.asyncio
async def test_deliver_digest_skips_silently_when_empty_and_notify_disabled(mocker) -> None:
    subscription = _make_subscription("digest")
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=None))
    deliver_mock = mocker.patch.object(deliver_digest, "deliver", new=AsyncMock())

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {"status": "skipped", "reason": "no_new_items"}
    assert deliver_mock.await_count == 0


@pytest.mark.asyncio
async def test_deliver_digest_skips_event_mode_subscription_without_invoking_generator(
    mocker,
) -> None:
    subscription = _make_subscription("event")
    _patch_session(mocker, subscription)
    generate_mock = mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock())

    result = await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    assert result == {"status": "skipped", "reason": "wrong_delivery_mode"}
    assert generate_mock.await_count == 0, (
        "event-mode subscription should short-circuit before invoking the digest generator"
    )


@pytest.mark.asyncio
async def test_deliver_digest_falls_back_to_the_users_default_webhook_url(mocker) -> None:
    subscription = _make_subscription("digest")
    webhook_url = f"http://tgbot-{uuid.uuid4().hex[:8]}.test/deliver"
    digest_text = f"D {uuid.uuid4().hex[:6]}"
    subscription.delivery_webhook_url = None
    subscription.__dict__["user"] = SimpleNamespace(delivery_webhook_url=webhook_url)
    _patch_session(mocker, subscription)
    mocker.patch.object(deliver_digest, "generate_digest", new=AsyncMock(return_value=digest_text))
    deliver_mock = mocker.patch.object(deliver_digest, "deliver", new=AsyncMock())

    await deliver_digest._deliver_digest(subscription.id, notify_if_empty=False)

    deliver_mock.assert_awaited_once_with(webhook_url, "", digest_text)
