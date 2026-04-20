"""Tests for batch event notification delivery."""

import logging
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.event.batch_assessor import BatchAssessmentResult, ItemAssessment
from news_service.tasks import deliver_events

logging.disable(logging.CRITICAL)


def _make_item(headline: str, body: str, url: str, source_id: uuid.UUID) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        source_id=source_id,
        headline=headline,
        body=body,
        published_at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        url=url,
    )


def _make_subscription(
    prompt: str, webhook_url: str, source_ids: list[uuid.UUID]
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_spec=prompt,
        delivery_webhook_url=webhook_url,
        digest_language="ru",
        _source_ids=source_ids,
    )


def _mock_batch_result(
    items: list[SimpleNamespace], relevant_ids: set[str]
) -> BatchAssessmentResult:
    assessments = []
    for item in items:
        is_relevant = str(item.id) in relevant_ids
        assessments.append(
            ItemAssessment(
                item_id=str(item.id),
                is_relevant=is_relevant,
                notification_body=f"Уведомление о {item.headline}" if is_relevant else "",
                reason="Соответствует" if is_relevant else "Не подходит",
            )
        )
    return BatchAssessmentResult(assessments=assessments)


@pytest.mark.asyncio
async def test_batch_delivers_relevant_event(mocker) -> None:
    source_id = uuid.uuid4()
    item = _make_item(
        f"Событие {uuid.uuid4().hex[:6]}",
        f"Описание {uuid.uuid4().hex[:6]}",
        f"https://e-{uuid.uuid4().hex[:8]}.test/1",
        source_id,
    )
    subscription = _make_subscription(
        f"Подписка {uuid.uuid4().hex[:6]}",
        f"http://fe-{uuid.uuid4().hex[:8]}.test/d",
        [source_id],
    )

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [item])),
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [subscription])),
            SimpleNamespace(all=lambda: [(source_id,)]),
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])),
        ]
    )
    fake_session.add = lambda x: None
    fake_session.commit = AsyncMock()

    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=fake_session), __aexit__=AsyncMock(return_value=False)
        ),
    )
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )

    batch_result = _mock_batch_result([item], {str(item.id)})
    mocker.patch.object(
        deliver_events, "assess_batch_events", new=AsyncMock(return_value=batch_result)
    )

    mocker.patch.object(deliver_events, "deliver", new=AsyncMock())

    result = await deliver_events._deliver_event_notifications_batch([item.id])

    assert result["status"] == "delivered", (
        "batch delivery did not return delivered status for relevant event"
    )
    assert result["delivered"] >= 1, "batch delivery did not report at least one delivery"


@pytest.mark.asyncio
async def test_batch_doesnt_send_irrelevant_items_into_the_judge_loop(mocker) -> None:
    source_id = uuid.uuid4()
    relevant_item = _make_item(
        f"Событие {uuid.uuid4().hex[:6]}",
        f"Описание {uuid.uuid4().hex[:6]}",
        f"https://e-{uuid.uuid4().hex[:8]}.test/1",
        source_id,
    )
    irrelevant_item = _make_item(
        f"Событие {uuid.uuid4().hex[:6]}",
        f"Описание {uuid.uuid4().hex[:6]}",
        f"https://e-{uuid.uuid4().hex[:8]}.test/2",
        source_id,
    )
    subscription = _make_subscription(
        f"Подписка {uuid.uuid4().hex[:6]}",
        f"http://fe-{uuid.uuid4().hex[:8]}.test/d",
        [source_id],
    )

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [relevant_item, irrelevant_item])
            ),
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [subscription])),
            SimpleNamespace(all=lambda: [(source_id,)]),
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])),
        ]
    )
    fake_session.add = lambda x: None
    fake_session.commit = AsyncMock()

    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=fake_session), __aexit__=AsyncMock(return_value=False)
        ),
    )
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        deliver_events,
        "assess_batch_events",
        new=AsyncMock(
            return_value=_mock_batch_result(
                [relevant_item, irrelevant_item],
                {str(relevant_item.id)},
            )
        ),
    )
    judge_spy = mocker.patch.object(
        deliver_events,
        "_judge_and_revise",
        new=AsyncMock(
            return_value=(
                BatchAssessmentResult(
                    assessments=[
                        ItemAssessment(
                            item_id=str(relevant_item.id),
                            is_relevant=True,
                            notification_body=f"Уведомление о {relevant_item.headline}",
                            reason="Соответствует",
                        )
                    ]
                ),
                set(),
            )
        ),
    )
    deliver_mock = mocker.patch.object(deliver_events, "deliver", new=AsyncMock())

    await deliver_events._deliver_event_notifications_batch([relevant_item.id, irrelevant_item.id])

    judge_input = judge_spy.await_args.kwargs["assessment"]
    assert (
        [a.item_id for a in judge_input.assessments] == [str(relevant_item.id)]
        and deliver_mock.await_count == 1
    ), "event delivery did not filter the batch down to relevant items before judging"


@pytest.mark.asyncio
async def test_batch_falls_back_to_the_users_default_webhook_url(mocker) -> None:
    source_id = uuid.uuid4()
    item = _make_item(
        f"Событие {uuid.uuid4().hex[:6]}",
        f"Описание {uuid.uuid4().hex[:6]}",
        f"https://e-{uuid.uuid4().hex[:8]}.test/1",
        source_id,
    )
    webhook_url = f"http://tgbot-{uuid.uuid4().hex[:8]}.test/deliver"
    subscription = _make_subscription(
        f"Подписка {uuid.uuid4().hex[:6]}",
        webhook_url,
        [source_id],
    )
    subscription.delivery_webhook_url = None
    subscription.user = SimpleNamespace(delivery_webhook_url=webhook_url)

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [item])),
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [subscription])),
            SimpleNamespace(all=lambda: [(source_id,)]),
            SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])),
        ]
    )
    fake_session.add = lambda x: None
    fake_session.commit = AsyncMock()

    mocker.patch.object(
        deliver_events,
        "get_task_session",
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=fake_session), __aexit__=AsyncMock(return_value=False)
        ),
    )
    mocker.patch.object(
        deliver_events, "load_recent_notification_history", new=AsyncMock(return_value=[])
    )

    batch_result = _mock_batch_result([item], {str(item.id)})
    mocker.patch.object(
        deliver_events, "assess_batch_events", new=AsyncMock(return_value=batch_result)
    )

    deliver_mock = mocker.patch.object(deliver_events, "deliver", new=AsyncMock())

    await deliver_events._deliver_event_notifications_batch([item.id])

    deliver_mock.assert_awaited_once_with(webhook_url, "", f"Уведомление о {item.headline}")


@pytest.mark.asyncio
async def test_batch_skips_when_no_items(mocker) -> None:
    result = await deliver_events._deliver_event_notifications_batch([])
    assert result["status"] == "skipped", "batch delivery did not skip empty batch"


@pytest.mark.asyncio
async def test_batch_assessment_result_model_accepts_mixed_results() -> None:
    result = BatchAssessmentResult(
        assessments=[
            ItemAssessment(
                item_id=str(uuid.uuid4()),
                is_relevant=True,
                notification_body=f"Текст {uuid.uuid4().hex[:6]}",
                reason="Совпадает",
            ),
            ItemAssessment(
                item_id=str(uuid.uuid4()),
                is_relevant=False,
                notification_body="",
                reason="Не подходит",
            ),
        ]
    )
    relevant = [a for a in result.assessments if a.is_relevant]
    assert len(relevant) == 1, "BatchAssessmentResult did not preserve mixed assessment results"


@pytest.mark.asyncio
async def test_item_assessment_model_requires_reason() -> None:
    with pytest.raises(ValueError):
        ItemAssessment(item_id=str(uuid.uuid4()), is_relevant=True, reason="ab")
