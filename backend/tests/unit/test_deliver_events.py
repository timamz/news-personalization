import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


def _make_item(*, event_title: str | None = "World tour announced") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        feed_id=uuid.uuid4(),
        headline="Artist announces tour",
        body="The artist confirmed a new world tour for this summer.",
        event_title=event_title,
        event_summary="The artist confirmed a new world tour for this summer.",
        event_starts_at=datetime(2026, 6, 1, 18, 0, tzinfo=UTC),
        published_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        source="Music Feed",
        url="https://example.com/events/1",
    )


def _make_subscription() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="Notify me when Drobyshevsky lectures are announced",
        event_matching_mode="basic",
        event_constraints=[],
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
async def test_deliver_event_notifications_skips_non_event_item(mocker) -> None:
    item = _make_item(event_title=None)
    session = _FakeSession(item=item)
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    get_channel = mocker.patch.object(deliver_events, "get_delivery_channel")

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {"status": "skipped", "reason": "not_event"}
    get_channel.assert_not_called()


@pytest.mark.asyncio
async def test_deliver_event_notifications_skips_strict_subscription_when_prefilter_fails(
    mocker,
) -> None:
    item = _make_item()
    subscription_data = _make_subscription().__dict__.copy()
    subscription_data["event_matching_mode"] = "strict_with_prefilter"
    subscription_data["event_constraints"] = [
        {
            "key": "speaker_must_be_drobyshevsky",
            "description": "Primary speaker identity",
            "value_type": "string",
            "match_mode": "exact",
            "required_string": "станислав владимирович дробышевский",
            "prefilter_terms": ["станислав владимирович дробышевский"],
        }
    ]
    subscription = SimpleNamespace(**subscription_data)
    item.headline = "Artist announces tour"
    item.body = "No relevant name is present here."
    item.event_title = "World tour announced"
    item.event_summary = "This is unrelated to the requested lecturer."
    session = _FakeSession(item=item, subscriptions=[subscription])
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    parse_values = mocker.patch.object(
        deliver_events,
        "parse_event_constraint_values",
        new=AsyncMock(),
    )
    get_channel = mocker.patch.object(deliver_events, "get_delivery_channel")

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {"status": "skipped", "reason": "already_sent"}
    parse_values.assert_not_awaited()
    get_channel.assert_not_called()


@pytest.mark.asyncio
async def test_deliver_event_notifications_applies_strict_constraint_match(mocker) -> None:
    item = _make_item(event_title="Новая лекция Дробышевского")
    item.headline = "Станислав Дробышевский выступит с лекцией"
    item.body = "Лекция Станислава Владимировича Дробышевского пройдет в Москве."
    item.event_summary = "Анонс лекции Станислава Дробышевского."
    subscription_data = _make_subscription().__dict__.copy()
    subscription_data["event_matching_mode"] = "strict_with_prefilter"
    subscription_data["event_constraints"] = [
        {
            "key": "speaker_must_be_drobyshevsky",
            "description": "Primary speaker identity",
            "value_type": "string",
            "match_mode": "exact",
            "required_string": "станислав владимирович дробышевский",
            "prefilter_terms": ["станислав владимирович дробышевский", "дробышевский"],
        },
        {
            "key": "is_other_person_speaking_under_brand",
            "description": "Whether another person is speaking under the same brand",
            "value_type": "boolean",
            "match_mode": "equals",
            "required_boolean": False,
            "prefilter_terms": ["лекция"],
        },
    ]
    subscription = SimpleNamespace(**subscription_data)
    session = _FakeSession(item=item, subscriptions=[subscription])
    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=_FakeSessionFactory(session),
    )
    mocker.patch.object(
        deliver_events,
        "parse_event_constraint_values",
        new=AsyncMock(
            return_value={
                "speaker_must_be_drobyshevsky": "станислав владимирович дробышевский",
                "is_other_person_speaking_under_brand": False,
            }
        ),
    )

    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)

    result = await deliver_events._deliver_event_notifications(item.id)

    assert result == {
        "status": "delivered",
        "delivered": 1,
        "failed": 0,
        "news_item_id": str(item.id),
    }
    channel.send.assert_awaited_once()
