import logging
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.event import (
    EventAssessmentResult,
    RecentEventsPreviewDecision,
)

logging.disable(logging.CRITICAL)


def _fake_completion(parsed: object) -> MagicMock:
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def _fake_client(parsed: object) -> AsyncMock:
    client = AsyncMock()
    client.beta.chat.completions.parse = AsyncMock(return_value=_fake_completion(parsed))
    return client


@pytest.mark.asyncio
async def test_assess_relevant_event_returns_is_relevant_true() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=True,
        notification_body=f"Новый эпизод #{uuid.uuid4().hex[:6]}",
        reason="Пост совпадает с запросом подписки",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Финал сериала Разделение",
            body="Новый эпизод выходит в пятницу.",
            url=f"https://example.com/{uuid.uuid4().hex}",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Уведомлять о новых эпизодах Разделения",
            target_language="ru",
            recent_notification_history=[],
            max_history_chars=100_000,
        )

    assert result.is_relevant_event is True, "assessment did not mark relevant event as relevant"


@pytest.mark.asyncio
async def test_assess_relevant_event_returns_nonempty_notification_body() -> None:
    body_text = f"Финал сезона подтверждён #{uuid.uuid4().hex[:6]}"
    parsed = EventAssessmentResult(
        is_relevant_event=True,
        notification_body=body_text,
        reason="Событие соответствует подписке",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Severance finale announced",
            body="The new episode arrives next Friday.",
            url=f"https://example.com/{uuid.uuid4().hex}",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[],
            max_history_chars=100_000,
        )

    assert result.notification_body != "", "relevant event returned empty notification body"


@pytest.mark.asyncio
async def test_assess_irrelevant_event_returns_is_relevant_false() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=False,
        notification_body="",
        reason="Общая новость, не совпадает с запросом",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Квартальная отчётность",
            body="Компания показала рост выручки за квартал.",
            url=f"https://example.com/{uuid.uuid4().hex}",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[],
            max_history_chars=100_000,
        )

    assert result.is_relevant_event is False, "irrelevant event was marked as relevant"


@pytest.mark.asyncio
async def test_assess_irrelevant_event_returns_empty_notification_body() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=False,
        notification_body="",
        reason="Статья не о сериале Разделение",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Quarterly earnings",
            body="The company reported higher revenue this quarter.",
            url=f"https://example.com/{uuid.uuid4().hex}",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[],
            max_history_chars=100_000,
        )

    assert result.notification_body == "", "irrelevant event returned non-empty notification body"


@pytest.mark.asyncio
async def test_assess_duplicate_event_returns_is_relevant_false() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=False,
        notification_body="",
        reason="Пользователь уже был уведомлён об этом",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Напоминание о финале Разделения",
            body="Напоминаем: финал в пятницу.",
            url=f"https://example.com/{uuid.uuid4().hex}",
            published_at=datetime(2026, 3, 14, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[
                "Title: Severance season finale\nSummary: Apple confirmed the finale release date."
            ],
            max_history_chars=100_000,
        )

    assert result.is_relevant_event is False, "duplicate event was not detected as irrelevant"


@pytest.mark.asyncio
async def test_render_recent_events_preview_returns_selected_item_ids() -> None:
    event_id = f"event-{uuid.uuid4().hex[:8]}"
    parsed = RecentEventsPreviewDecision(
        selected_item_ids=[event_id],
        subject="Что вы могли пропустить",
        body=f"- Лекция Дробышевского\nhttps://example.com/{event_id}",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import render_recent_events_preview

        result = await render_recent_events_preview(
            raw_prompt="Только лекции Дробышевского",
            target_language="ru",
            lookback_days=7,
            candidate_events=[
                f"ID: {event_id}\nTitle: Лекция Станислава Дробышевского\n"
                f"URL: https://example.com/{event_id}"
            ],
            recent_notifications=[],
        )

    assert result.selected_item_ids == [event_id], "preview did not return expected item ids"


@pytest.mark.asyncio
async def test_render_recent_events_preview_returns_expected_subject() -> None:
    subject_text = f"Пропущенные события #{uuid.uuid4().hex[:6]}"
    parsed = RecentEventsPreviewDecision(
        selected_item_ids=["event-1"],
        subject=subject_text,
        body="- Событие\nhttps://example.com/event-1",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import render_recent_events_preview

        result = await render_recent_events_preview(
            raw_prompt="Лекции по антропологии",
            target_language="ru",
            lookback_days=7,
            candidate_events=["ID: event-1\nTitle: Лекция\nURL: https://example.com/event-1"],
            recent_notifications=[],
        )

    assert result.subject == subject_text, "preview subject did not match expected value"


@pytest.mark.asyncio
async def test_render_recent_events_preview_body_contains_event_url() -> None:
    event_url = f"https://example.com/событие-{uuid.uuid4().hex[:8]}"
    parsed = RecentEventsPreviewDecision(
        selected_item_ids=["event-1"],
        subject="Что вы могли пропустить",
        body=f"- Лекция Дробышевского\n{event_url}",
    )

    with patch("news_service.agents.event._client", _fake_client(parsed)):
        from news_service.agents.event import render_recent_events_preview

        result = await render_recent_events_preview(
            raw_prompt="Только лекции Дробышевского",
            target_language="ru",
            lookback_days=7,
            candidate_events=[f"ID: event-1\nTitle: Лекция Дробышевского\nURL: {event_url}"],
            recent_notifications=[],
        )

    assert event_url in result.body, "preview body did not contain the event url"
