import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.event import EventAssessmentResult
from news_service.tasks import deliver_events


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

    async def get(self, _model, _item_id) -> object | None:  # noqa: ANN001
        return self._item

    async def execute(self, _statement) -> _FakeResult:  # noqa: ANN001
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


def _make_item() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        feed_id=uuid.uuid4(),
        headline="Artist announces tour",
        body="The artist confirmed a new world tour for this summer.",
        published_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        source="Music Feed",
        url="https://example.com/events/1",
    )


def _make_subscription() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        raw_prompt="Notify me when Drobyshevsky lectures are announced",
        canonical_prompt="Notify me when Drobyshevsky lectures are announced",
        event_matching_mode="basic",
        delivery_webhook_url="http://frontend.example.test/deliver/1",
        digest_language="en",
    )


@pytest.mark.asyncio
async def test_deliver_event_notifications_sends_and_marks_sent(mocker) -> None:
    item = _make_item()
    subscription = _make_subscription()
    session = _FakeSession(item=item, subscriptions=[subscription])
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    mocker.patch.object(
        deliver_events,
        "load_recent_notification_history",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=True,
                title="World tour announced",
                summary="The artist confirmed a new world tour.",
                when="Summer 2026",
                notification_body="World tour announced\nSummer 2026\nThe artist confirmed.",
                reason="Matches the subscription about Drobyshevsky lectures.",
            )
        ),
    )

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {
        "status": "delivered",
        "delivered": 1,
        "failed": 0,
        "news_item_id": str(item.id),
    }
    channel.send.assert_awaited_once()
    assert session.commits == 1
    assert len(session.added) == 1


@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_not_found(mocker) -> None:
    session = _FakeSession(item=None)
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    result = await deliver_events._deliver_event_notifications(uuid.uuid4())

    assert result == {"status": "skipped", "reason": "news_item_not_found"}


@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_when_assessment_not_relevant(mocker) -> None:
    item = _make_item()
    subscription = _make_subscription()
    session = _FakeSession(item=item, subscriptions=[subscription])
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        deliver_events,
        "load_recent_notification_history",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(
            return_value=EventAssessmentResult(
                is_relevant_event=False,
                title=None,
                summary=None,
                when=None,
                notification_body="",
                reason="The event does not match the subscription.",
            )
        ),
    )
    get_channel = mocker.patch.object(deliver_events, "get_delivery_channel")

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {"status": "skipped", "reason": "already_sent"}
    get_channel.assert_not_called()


@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_headline_duplicate(mocker) -> None:
    item = _make_item()
    subscription = _make_subscription()
    session = _FakeSession(item=item, subscriptions=[subscription])
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )

    history_entry = SimpleNamespace(
        sent_at=datetime(2026, 3, 1, 10, 0, tzinfo=UTC),
        source="Music Feed",
        title="Artist announces tour",
        summary="The artist confirmed a new world tour.",
    )
    mocker.patch.object(
        deliver_events,
        "load_recent_notification_history",
        new=AsyncMock(return_value=[history_entry]),
    )
    assess_mock = mocker.patch.object(
        deliver_events,
        "assess_and_compose_event_notification",
        new=AsyncMock(),
    )
    get_channel = mocker.patch.object(deliver_events, "get_delivery_channel")

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {"status": "skipped", "reason": "already_sent"}
    assess_mock.assert_not_awaited()
    get_channel.assert_not_called()


@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_already_sent_subscription(mocker) -> None:
    item = _make_item()
    subscription = _make_subscription()
    session = _FakeSession(
        item=item,
        subscriptions=[subscription],
        sent_subscription_ids=[subscription.id],
    )
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    get_channel = mocker.patch.object(deliver_events, "get_delivery_channel")

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {"status": "skipped", "reason": "already_sent"}
    get_channel.assert_not_called()
