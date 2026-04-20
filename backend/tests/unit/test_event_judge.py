"""Tests for the batch event judge -- per-item PASS/REVISE critic."""

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.event.batch_assessor import BatchAssessmentResult, ItemAssessment
from news_service.agents.event.judge import (
    BatchJudgeResult,
    ItemVerdict,
    judge_batch_events,
)

logging.disable(logging.CRITICAL)

_JUDGE_PATH = "news_service.agents.event.judge.chat_completion"


def _fake_completion(parsed: object) -> MagicMock:
    msg = MagicMock()
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _assessment(
    item_id: str, is_relevant: bool, body: str = "", reason: str = ""
) -> ItemAssessment:
    return ItemAssessment(
        item_id=item_id,
        is_relevant=is_relevant,
        notification_body=body,
        reason=reason or f"причина {uuid.uuid4().hex[:4]}",
    )


@pytest.mark.asyncio
async def test_judge_returns_pass_overall_when_all_items_pass(mocker) -> None:
    item_id = str(uuid.uuid4())
    parsed = BatchJudgeResult(
        per_item=[ItemVerdict(item_id=item_id, verdict="PASS", feedback="")],
        overall="PASS",
    )
    mocker.patch(_JUDGE_PATH, new=AsyncMock(return_value=_fake_completion(parsed)))

    result = await judge_batch_events(
        assessment=BatchAssessmentResult(
            assessments=[_assessment(item_id, True, body="Короткое сообщение", reason="подходит")]
        ),
        user_spec=f"Спецификация {uuid.uuid4().hex[:5]}",
        recent_notification_history=[],
        max_history_chars=100_000,
    )

    assert result.overall == "PASS", "judge did not return overall PASS when all items pass"


@pytest.mark.asyncio
async def test_judge_returns_revise_when_any_item_fails(mocker) -> None:
    good_id = str(uuid.uuid4())
    bad_id = str(uuid.uuid4())
    parsed = BatchJudgeResult(
        per_item=[
            ItemVerdict(item_id=good_id, verdict="PASS", feedback=""),
            ItemVerdict(
                item_id=bad_id,
                verdict="REVISE",
                feedback=f"Body использует markdown bold {uuid.uuid4().hex[:4]}",
            ),
        ],
        overall="REVISE",
    )
    mocker.patch(_JUDGE_PATH, new=AsyncMock(return_value=_fake_completion(parsed)))

    result = await judge_batch_events(
        assessment=BatchAssessmentResult(
            assessments=[
                _assessment(good_id, True, body="Plain text body"),
                _assessment(bad_id, True, body="**Bold title** и дальше текст"),
            ]
        ),
        user_spec="Подписка",
        recent_notification_history=[],
        max_history_chars=100_000,
    )

    revise = [v for v in result.per_item if v.verdict == "REVISE"]
    assert result.overall == "REVISE" and len(revise) == 1 and revise[0].item_id == bad_id, (
        "judge did not surface the single-item REVISE correctly"
    )


@pytest.mark.asyncio
async def test_judge_rejects_empty_assessment_batch() -> None:
    with pytest.raises(ValueError):
        await judge_batch_events(
            assessment=BatchAssessmentResult(assessments=[]),
            user_spec="нечто",
            recent_notification_history=[],
            max_history_chars=100,
        )


@pytest.mark.asyncio
async def test_judge_rejects_verdicts_for_unknown_item_ids(mocker) -> None:
    real_id = str(uuid.uuid4())
    ghost_id = str(uuid.uuid4())
    parsed = BatchJudgeResult(
        per_item=[ItemVerdict(item_id=ghost_id, verdict="PASS", feedback="")],
        overall="PASS",
    )
    mocker.patch(_JUDGE_PATH, new=AsyncMock(return_value=_fake_completion(parsed)))

    with pytest.raises(ValueError):
        await judge_batch_events(
            assessment=BatchAssessmentResult(assessments=[_assessment(real_id, False)]),
            user_spec=f"спец {uuid.uuid4().hex[:3]}",
            recent_notification_history=[],
            max_history_chars=100_000,
        )


@pytest.mark.asyncio
async def test_judge_truncates_oversized_history_block(mocker) -> None:
    item_id = str(uuid.uuid4())
    parsed = BatchJudgeResult(
        per_item=[ItemVerdict(item_id=item_id, verdict="PASS", feedback="")],
        overall="PASS",
    )
    captured: list[dict] = []

    async def _capturing_chat_completion(**kwargs):
        captured.append(kwargs)
        return _fake_completion(parsed)

    mocker.patch(_JUDGE_PATH, new=_capturing_chat_completion)

    big_entry = "Очень длинная запись " * 5000
    await judge_batch_events(
        assessment=BatchAssessmentResult(assessments=[_assessment(item_id, False)]),
        user_spec="спец",
        recent_notification_history=[big_entry, big_entry],
        max_history_chars=1000,
    )

    user_msg = captured[0]["messages"][1]["content"]
    assert "(truncated)" in user_msg, "judge did not truncate history exceeding max_history_chars"
