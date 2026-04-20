"""Tests for the weekly event verifier Celery task."""

import logging
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.event.verifier import MissedEvent
from news_service.tasks import reflect_events

logging.disable(logging.CRITICAL)


def _subscription(last_reflected_at: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_spec=f"спец {uuid.uuid4().hex[:4]}",
        digest_language="ru",
        delivery_webhook_url=f"http://fe-{uuid.uuid4().hex[:6]}.test/hook",
        last_reflected_at=last_reflected_at,
    )


def _wrap_session(session: MagicMock) -> MagicMock:
    return MagicMock(
        __aenter__=AsyncMock(return_value=session),
        __aexit__=AsyncMock(return_value=False),
    )


def _install_session_factory(mocker, session: MagicMock) -> None:
    mocker.patch.object(
        reflect_events,
        "get_task_session",
        side_effect=lambda: _wrap_session(session),
    )


@pytest.mark.asyncio
async def test_task_skips_when_no_due_subscriptions(mocker) -> None:
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
    )
    _install_session_factory(mocker, session)

    result = await reflect_events._reflect_event_subscriptions()

    assert result == {"status": "skipped", "reason": "no_due_subscriptions"}, (
        "task did not short-circuit when there are no due event subscriptions"
    )


@pytest.mark.asyncio
async def test_task_dispatches_discovery_reasons_the_agent_emitted(mocker) -> None:
    sub = _subscription()
    reason_a = f"событие пропущено A {uuid.uuid4().hex[:4]}"
    reason_b = f"событие пропущено B {uuid.uuid4().hex[:4]}"

    outer_session = MagicMock()
    outer_session.execute = AsyncMock(
        return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [sub]))
    )
    inner_session = MagicMock()
    inner_session.execute = AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: sub))
    inner_session.commit = AsyncMock()
    mocker.patch.object(
        reflect_events,
        "get_task_session",
        side_effect=[_wrap_session(outer_session), _wrap_session(inner_session)],
    )
    mocker.patch.object(
        reflect_events,
        "load_recent_notification_history",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch.object(reflect_events, "_load_source_contexts", new=AsyncMock(return_value=[]))
    mocker.patch.object(
        reflect_events,
        "run_event_verifier",
        new=AsyncMock(
            return_value={
                "missed_events": [],
                "discovery_reasons": [reason_a, reason_b],
                "status_messages": [],
                "search_budget_used": 1,
                "observations": "done",
            }
        ),
    )
    send_task_spy = mocker.patch.object(reflect_events.celery_app, "send_task")

    await reflect_events._reflect_event_subscriptions()

    dispatched = [call.args[0] for call in send_task_spy.call_args_list]
    reasons = [call.kwargs.get("args", call.args[1:])[1] for call in send_task_spy.call_args_list]
    assert (
        send_task_spy.call_count == 2
        and all(
            t == "news_service.tasks.discover_sources.discover_sources_for_subscription"
            for t in dispatched
        )
        and set(reasons) == {reason_a, reason_b}
    ), "task did not dispatch one discovery call per agent-emitted reason"


@pytest.mark.asyncio
async def test_task_updates_last_reflected_at_after_successful_run(mocker) -> None:
    sub = _subscription()
    outer_session = MagicMock()
    outer_session.execute = AsyncMock(
        return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [sub]))
    )
    inner_session = MagicMock()
    inner_session.execute = AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: sub))
    inner_session.commit = AsyncMock()
    mocker.patch.object(
        reflect_events,
        "get_task_session",
        side_effect=[_wrap_session(outer_session), _wrap_session(inner_session)],
    )
    mocker.patch.object(
        reflect_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(reflect_events, "_load_source_contexts", new=AsyncMock(return_value=[]))
    mocker.patch.object(
        reflect_events,
        "run_event_verifier",
        new=AsyncMock(
            return_value={
                "missed_events": [],
                "discovery_reasons": [],
                "status_messages": [],
                "search_budget_used": 0,
                "observations": "done",
            }
        ),
    )
    mocker.patch.object(reflect_events.celery_app, "send_task")

    before = datetime.now(UTC)
    await reflect_events._reflect_event_subscriptions()

    assert sub.last_reflected_at is not None and sub.last_reflected_at >= before, (
        "task did not stamp last_reflected_at after a successful verifier run"
    )


@pytest.mark.asyncio
async def test_task_swallows_per_subscription_failure_without_aborting_others(mocker) -> None:
    sub_failing = _subscription()
    sub_good = _subscription()

    outer_session = MagicMock()
    outer_session.execute = AsyncMock(
        return_value=SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [sub_failing, sub_good])
        )
    )

    def _inner_for(sub):
        inner = MagicMock()
        inner.execute = AsyncMock(return_value=SimpleNamespace(scalar_one=lambda s=sub: s))
        inner.commit = AsyncMock()
        return inner

    mocker.patch.object(
        reflect_events,
        "get_task_session",
        side_effect=[
            _wrap_session(outer_session),
            _wrap_session(_inner_for(sub_failing)),
            _wrap_session(_inner_for(sub_good)),
        ],
    )
    mocker.patch.object(
        reflect_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(reflect_events, "_load_source_contexts", new=AsyncMock(return_value=[]))

    async def _verifier_side_effect(**kwargs):
        if kwargs["subscription"].id == sub_failing.id:
            raise RuntimeError(f"boom {uuid.uuid4().hex[:4]}")
        return {
            "missed_events": [],
            "discovery_reasons": [],
            "status_messages": [],
            "search_budget_used": 0,
            "observations": "done",
        }

    mocker.patch.object(
        reflect_events, "run_event_verifier", new=AsyncMock(side_effect=_verifier_side_effect)
    )
    mocker.patch.object(reflect_events.celery_app, "send_task")

    result = await reflect_events._reflect_event_subscriptions()

    assert result["processed"] == 1 and result["failed"] == 1, (
        "task did not isolate a per-subscription failure from its siblings"
    )


@pytest.mark.asyncio
async def test_deliver_and_record_miss_creates_synthetic_news_item_and_sent_item(mocker) -> None:
    subscription = _subscription()
    miss = MissedEvent(
        title=f"Catch-up {uuid.uuid4().hex[:5]}",
        summary="Brief summary",
        source_url=f"https://official-{uuid.uuid4().hex[:8]}.test/announcement",
        happened_at="2026-04-18",
    )

    sentinel = SimpleNamespace(
        id=uuid.uuid4(),
        url=reflect_events.VERIFIER_SENTINEL_SOURCE_URL,
        title=reflect_events.VERIFIER_SENTINEL_SOURCE_TITLE,
    )

    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    mocker.patch.object(
        reflect_events,
        "_get_or_create_verifier_source",
        new=AsyncMock(return_value=sentinel),
    )
    session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(scalar_one_or_none=lambda: None),
            SimpleNamespace(scalar_one_or_none=lambda: None),
        ]
    )
    deliver_spy = mocker.patch.object(reflect_events, "deliver", new=AsyncMock())

    await reflect_events._deliver_and_record_miss(
        session=session,
        subscription=subscription,
        miss=miss,
    )

    added_types = [type(call.args[0]).__name__ for call in session.add.call_args_list]
    assert (
        deliver_spy.call_count == 1 and "NewsItem" in added_types and "SentItem" in added_types
    ), "catch-up delivery did not create both a synthetic NewsItem and a SentItem"
