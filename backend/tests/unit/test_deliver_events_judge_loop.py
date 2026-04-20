"""Tests for the judge-critic loop wrapped around the batch assessor in deliver_events."""

import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.agents.event.batch_assessor import BatchAssessmentResult, ItemAssessment
from news_service.agents.event.judge import BatchJudgeResult, ItemVerdict
from news_service.tasks import deliver_events
from news_service.tasks.deliver_events import _judge_and_revise

logging.disable(logging.CRITICAL)


def _assessment(*pairs: tuple[str, bool, str]) -> BatchAssessmentResult:
    return BatchAssessmentResult(
        assessments=[
            ItemAssessment(
                item_id=iid,
                is_relevant=relevant,
                notification_body=body,
                reason=f"причина {uuid.uuid4().hex[:4]}",
            )
            for iid, relevant, body in pairs
        ]
    )


def _item(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "headline": f"Заголовок {uuid.uuid4().hex[:5]}",
        "body": f"Тело {uuid.uuid4().hex[:5]}",
        "url": f"https://example.test/{uuid.uuid4().hex[:8]}",
        "published_at": "unknown",
    }


@pytest.mark.asyncio
async def test_judge_loop_is_skipped_when_no_items_are_relevant(mocker) -> None:
    judge_spy = mocker.patch.object(
        deliver_events,
        "judge_batch_events",
        new=AsyncMock(side_effect=AssertionError("judge must not be called")),
    )
    sub_id = uuid.uuid4()
    assessment = _assessment(
        (str(uuid.uuid4()), False, ""),
        (str(uuid.uuid4()), False, ""),
    )

    final, dropped = await _judge_and_revise(
        assessment=assessment,
        items_for_llm=[_item(a.item_id) for a in assessment.assessments],
        user_spec=f"спец {uuid.uuid4().hex[:4]}",
        target_language="ru",
        history_strings=[],
        subscription_id=sub_id,
    )

    assert judge_spy.call_count == 0 and final is assessment and dropped == set(), (
        "judge loop did not short-circuit on an all-irrelevant batch"
    )


@pytest.mark.asyncio
async def test_judge_loop_reruns_assessor_only_for_revise_items_and_merges(mocker) -> None:
    good = str(uuid.uuid4())
    bad = str(uuid.uuid4())
    initial = _assessment((good, True, "Хороший body"), (bad, True, "**Жирный** body"))

    mocker.patch.object(
        deliver_events,
        "judge_batch_events",
        new=AsyncMock(
            side_effect=[
                BatchJudgeResult(
                    per_item=[
                        ItemVerdict(item_id=good, verdict="PASS", feedback=""),
                        ItemVerdict(item_id=bad, verdict="REVISE", feedback="markdown bold"),
                    ],
                    overall="REVISE",
                ),
                BatchJudgeResult(
                    per_item=[
                        ItemVerdict(item_id=good, verdict="PASS", feedback=""),
                        ItemVerdict(item_id=bad, verdict="PASS", feedback=""),
                    ],
                    overall="PASS",
                ),
            ]
        ),
    )

    revised_body = f"Переписано plain text {uuid.uuid4().hex[:4]}"
    assessor_spy = mocker.patch.object(
        deliver_events,
        "assess_batch_events",
        new=AsyncMock(
            return_value=_assessment((bad, True, revised_body)),
        ),
    )

    final, dropped = await _judge_and_revise(
        assessment=initial,
        items_for_llm=[_item(good), _item(bad)],
        user_spec="спец",
        target_language="ru",
        history_strings=[],
        subscription_id=uuid.uuid4(),
    )

    bad_result = next(a for a in final.assessments if a.item_id == bad)
    good_result = next(a for a in final.assessments if a.item_id == good)
    revise_call_kwargs = assessor_spy.call_args.kwargs
    assert (
        bad_result.notification_body == revised_body
        and good_result.notification_body == "Хороший body"
        and dropped == set()
        and [i["item_id"] for i in revise_call_kwargs["items"]] == [bad]
    ), "judge loop did not merge revised REVISE item while leaving PASS item untouched"


@pytest.mark.asyncio
async def test_judge_loop_drops_items_still_revise_after_max_revisions(mocker, monkeypatch) -> None:
    stubborn = str(uuid.uuid4())
    initial = _assessment((stubborn, True, "проблемный body"))
    monkeypatch.setattr(deliver_events.settings, "event_judge_max_revisions", 1)

    mocker.patch.object(
        deliver_events,
        "judge_batch_events",
        new=AsyncMock(
            return_value=BatchJudgeResult(
                per_item=[ItemVerdict(item_id=stubborn, verdict="REVISE", feedback="still bad")],
                overall="REVISE",
            )
        ),
    )
    mocker.patch.object(
        deliver_events,
        "assess_batch_events",
        new=AsyncMock(return_value=_assessment((stubborn, True, "всё ещё плохо"))),
    )

    _, dropped = await _judge_and_revise(
        assessment=initial,
        items_for_llm=[_item(stubborn)],
        user_spec="спец",
        target_language="ru",
        history_strings=[],
        subscription_id=uuid.uuid4(),
    )

    assert dropped == {stubborn}, (
        "judge loop did not drop item that stayed REVISE past max_revisions"
    )


@pytest.mark.asyncio
async def test_judge_loop_falls_through_when_judge_raises(mocker) -> None:
    item_id = str(uuid.uuid4())
    original = _assessment((item_id, True, "плохой body"))

    mocker.patch.object(
        deliver_events,
        "judge_batch_events",
        new=AsyncMock(side_effect=RuntimeError(f"upstream down {uuid.uuid4().hex[:4]}")),
    )
    assessor_spy = mocker.patch.object(
        deliver_events,
        "assess_batch_events",
        new=AsyncMock(side_effect=AssertionError("assessor must not be called after judge fails")),
    )

    final, dropped = await _judge_and_revise(
        assessment=original,
        items_for_llm=[_item(item_id)],
        user_spec="спец",
        target_language="ru",
        history_strings=[],
        subscription_id=uuid.uuid4(),
    )

    assert final is original and dropped == set() and assessor_spy.call_count == 0, (
        "judge loop did not fall through with unreviewed output when judge raised"
    )
