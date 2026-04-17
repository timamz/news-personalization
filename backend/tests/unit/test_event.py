"""Tests for the batch event assessor agent."""

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.event.batch_assessor import (
    BatchAssessmentResult,
    ItemAssessment,
    assess_batch_events,
)

logging.disable(logging.CRITICAL)

_BATCH_PATH = "news_service.agents.event.batch_assessor.chat_completion"


def _fake_completion(parsed: object) -> MagicMock:
    msg = MagicMock()
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_batch_assess_returns_relevant_items(mocker) -> None:
    item_id = str(uuid.uuid4())
    parsed = BatchAssessmentResult(
        assessments=[
            ItemAssessment(
                item_id=item_id,
                is_relevant=True,
                notification_body=f"Уведомление {uuid.uuid4().hex[:6]}",
                reason="Соответствует подписке",
            )
        ]
    )
    mocker.patch(_BATCH_PATH, new=AsyncMock(return_value=_fake_completion(parsed)))

    result = await assess_batch_events(
        items=[
            {
                "item_id": item_id,
                "headline": f"Событие {uuid.uuid4().hex[:6]}",
                "body": "Описание события",
                "url": f"https://example.com/{uuid.uuid4().hex}",
            }
        ],
        user_spec="Уведомлять о новых лекциях",
        target_language="ru",
        recent_notification_history=[],
        max_history_chars=100_000,
    )

    assert result.assessments[0].is_relevant is True, (
        "batch assessment did not mark relevant item as relevant"
    )


@pytest.mark.asyncio
async def test_batch_assess_returns_not_relevant_items(mocker) -> None:
    item_id = str(uuid.uuid4())
    parsed = BatchAssessmentResult(
        assessments=[
            ItemAssessment(
                item_id=item_id,
                is_relevant=False,
                notification_body="",
                reason="Не совпадает с подпиской",
            )
        ]
    )
    mocker.patch(_BATCH_PATH, new=AsyncMock(return_value=_fake_completion(parsed)))

    result = await assess_batch_events(
        items=[
            {
                "item_id": item_id,
                "headline": f"Нерелевантное {uuid.uuid4().hex[:6]}",
                "body": "Текст",
                "url": f"https://example.com/{uuid.uuid4().hex}",
            }
        ],
        user_spec="Уведомлять о новых лекциях",
        target_language="ru",
        recent_notification_history=[],
        max_history_chars=100_000,
    )

    assert result.assessments[0].is_relevant is False, (
        "batch assessment did not mark irrelevant item"
    )


@pytest.mark.asyncio
async def test_batch_assess_handles_multiple_items(mocker) -> None:
    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())
    parsed = BatchAssessmentResult(
        assessments=[
            ItemAssessment(
                item_id=id_a, is_relevant=True, notification_body="A", reason="Совпадает"
            ),
            ItemAssessment(
                item_id=id_b, is_relevant=False, notification_body="", reason="Не совпадает"
            ),
        ]
    )
    mocker.patch(_BATCH_PATH, new=AsyncMock(return_value=_fake_completion(parsed)))

    result = await assess_batch_events(
        items=[
            {"item_id": id_a, "headline": "A", "body": "A", "url": "http://a.test"},
            {"item_id": id_b, "headline": "B", "body": "B", "url": "http://b.test"},
        ],
        user_spec="Тема",
        target_language="en",
        recent_notification_history=[],
        max_history_chars=100_000,
    )

    assert len(result.assessments) == 2, "batch assessment did not return all items"
    relevant = [a for a in result.assessments if a.is_relevant]
    assert len(relevant) == 1, "batch assessment did not return exactly one relevant item"
