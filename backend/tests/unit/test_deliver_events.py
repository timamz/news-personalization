import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.event import EventAssessmentResult
from news_service.tasks import deliver_events

logging.disable(logging.CRITICAL)


class _FakeResult:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[object]:
        return self._values


class _FakeSession:
    def __init__(
        self,
        *,
        item: object | None,
        subscriptions: list[object] | None = None,
        sent_subscription_ids: list[uuid.UUID] | None = None,
    ) -> None:
        self._item = item
        self._results = [
            _FakeResult(subscriptions or []),
            _FakeResult(sent_subscription_ids or []),
        ]
        self.added: list[object] = []
        self.commits = 0

    async def get(self, _model, _item_id) -> object | None:
        return self._item

    async def execute(self, _statement) -> _FakeResult:
        return self._results.pop(0)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
        return False


def _make_item(headline: str, body: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        headline=headline,
        body=body,
        published_at=datetime(
            2026, random.randint(1, 12), random.randint(1, 28), 12, 0, tzinfo=UTC
        ),
        source=f"Источник-{uuid.uuid4().hex[:6]}",
        url=url,
    )


def _make_subscription(prompt: str, webhook_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        raw_prompt=prompt,
        canonical_prompt=prompt,
        delivery_webhook_url=webhook_url,
        digest_language="ru",
    )


def _patch_session(mocker, session: _FakeSession) -> None:
    mocker.patch.object(
        deliver_events, "get_task_session", return_value=_FakeSessionFactory(session)
    )


@pytest.mark.asyncio
async def test_deliver_event_notifications_returns_delivered_status(mocker) -> None:
    headline = f"Артист анонсировал тур {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline,
        f"Описание тура {uuid.uuid4().hex[:6]}",
        f"https://example-{uuid.uuid4().hex[:8]}.test/events/1",
    )
    prompt = f"Уведомить о лекциях Дробышевского {uuid.uuid4().hex[:6]}"
    webhook = f"http://frontend-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(item=item, subscriptions=[subscription])
    _patch_session(mocker, session)
    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=True,
                notification_body=f"Уведомление {uuid.uuid4().hex[:6]}",
                reason=f"Причина {uuid.uuid4().hex[:6]}",
            )
        ),
    )

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result["status"] == "delivered", (
        "deliver_event_notifications did not return delivered status for relevant event"
    )


@pytest.mark.asyncio
async def test_deliver_event_notifications_delivers_exactly_one(mocker) -> None:
    headline = f"Событие {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline, f"Тело {uuid.uuid4().hex[:6]}", f"https://e-{uuid.uuid4().hex[:8]}.test/1"
    )
    prompt = f"Подписка {uuid.uuid4().hex[:6]}"
    webhook = f"http://fe-{uuid.uuid4().hex[:8]}.test/d"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(item=item, subscriptions=[subscription])
    _patch_session(mocker, session)
    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=True,
                notification_body=f"Текст {uuid.uuid4().hex[:6]}",
                reason=f"Причина {uuid.uuid4().hex[:6]}",
            )
        ),
    )

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result["delivered"] == 1, (
        "deliver_event_notifications did not report exactly one delivery"
    )


@pytest.mark.asyncio
async def test_deliver_event_notifications_commits_session_after_delivery(mocker) -> None:
    headline = f"Коммит-событие {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline, f"Тело {uuid.uuid4().hex[:6]}", f"https://e-{uuid.uuid4().hex[:8]}.test/1"
    )
    prompt = f"Подписка {uuid.uuid4().hex[:6]}"
    webhook = f"http://fe-{uuid.uuid4().hex[:8]}.test/d"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(item=item, subscriptions=[subscription])
    _patch_session(mocker, session)
    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=True,
                notification_body=f"Текст {uuid.uuid4().hex[:6]}",
                reason=f"Причина {uuid.uuid4().hex[:6]}",
            )
        ),
    )

    await deliver_events._deliver_event_notifications(item.id)

    assert session.commits == 1, "deliver_event_notifications did not commit session after delivery"


@pytest.mark.asyncio
async def test_deliver_event_notifications_adds_sent_record(mocker) -> None:
    headline = f"Запись-событие {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline, f"Тело {uuid.uuid4().hex[:6]}", f"https://e-{uuid.uuid4().hex[:8]}.test/1"
    )
    prompt = f"Подписка {uuid.uuid4().hex[:6]}"
    webhook = f"http://fe-{uuid.uuid4().hex[:8]}.test/d"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(item=item, subscriptions=[subscription])
    _patch_session(mocker, session)
    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=True,
                notification_body=f"Текст {uuid.uuid4().hex[:6]}",
                reason=f"Причина {uuid.uuid4().hex[:6]}",
            )
        ),
    )

    await deliver_events._deliver_event_notifications(item.id)

    assert len(session.added) == 1, "deliver_event_notifications did not add sent record to session"


@pytest.mark.asyncio
async def test_deliver_event_notifications_returns_skipped_when_item_not_found(mocker) -> None:
    session = _FakeSession(item=None)
    _patch_session(mocker, session)

    result = await deliver_events._deliver_event_notifications(uuid.uuid4())

    assert result == {
        "status": "skipped",
        "reason": "news_item_not_found",
    }, "deliver_event_notifications did not return skipped when item not found"


@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_when_assessment_not_relevant(mocker) -> None:
    headline = f"Нерелевантное событие {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline, f"Тело {uuid.uuid4().hex[:6]}", f"https://e-{uuid.uuid4().hex[:8]}.test/1"
    )
    prompt = f"Подписка {uuid.uuid4().hex[:6]}"
    webhook = f"http://fe-{uuid.uuid4().hex[:8]}.test/d"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(item=item, subscriptions=[subscription])
    _patch_session(mocker, session)
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=False,
                notification_body="",
                reason=f"Причина {uuid.uuid4().hex[:6]}",
            )
        ),
    )

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result["status"] == "skipped", (
        "deliver_event_notifications did not skip when assessment is not relevant"
    )


@pytest.mark.asyncio
async def test_deliver_event_notifications_does_not_call_channel_when_not_relevant(
    mocker,
) -> None:
    headline = f"Нерелевантное {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline, f"Тело {uuid.uuid4().hex[:6]}", f"https://e-{uuid.uuid4().hex[:8]}.test/1"
    )
    prompt = f"Подписка {uuid.uuid4().hex[:6]}"
    webhook = f"http://fe-{uuid.uuid4().hex[:8]}.test/d"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(item=item, subscriptions=[subscription])
    _patch_session(mocker, session)
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=False,
                notification_body="",
                reason=f"Не релевантно {uuid.uuid4().hex[:6]}",
            )
        ),
    )
    get_channel = mocker.patch.object(deliver_events, "get_delivery_channel")

    await deliver_events._deliver_event_notifications(item.id)

    assert not get_channel.called, (
        "deliver_event_notifications called channel when event not relevant"
    )



@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_already_sent_subscription(mocker) -> None:
    headline = f"Уже отправлено {uuid.uuid4().hex[:6]}"
    item = _make_item(
        headline, f"Тело {uuid.uuid4().hex[:6]}", f"https://e-{uuid.uuid4().hex[:8]}.test/1"
    )
    prompt = f"Подписка {uuid.uuid4().hex[:6]}"
    webhook = f"http://fe-{uuid.uuid4().hex[:8]}.test/d"
    subscription = _make_subscription(prompt, webhook)
    session = _FakeSession(
        item=item,
        subscriptions=[subscription],
        sent_subscription_ids=[subscription.id],
    )
    _patch_session(mocker, session)
    mocker.patch.object(deliver_events, "get_delivery_channel")

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {
        "status": "skipped",
        "reason": "already_sent",
    }, "deliver_event_notifications did not skip already sent subscription"
