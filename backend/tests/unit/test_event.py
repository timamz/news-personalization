from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.event import (
    EventAssessmentResult,
    RecentEventsPreviewDecision,
)


@pytest.mark.asyncio
async def test_assess_and_compose_event_notification_returns_relevant() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=True,
        notification_body="Severance season finale\n2026-03-20\nApple confirmed the finale.",
        reason="The article announces an upcoming TV event matching the subscription.",
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.event._client", mock_client):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Severance finale announced",
            body="The new episode arrives next Friday.",
            url="https://example.com/severance-finale",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[],
            max_history_chars=100_000,
        )

    assert result.is_relevant_event is True
    assert result.notification_body != ""
    assert "announced" not in result.reason or len(result.reason) >= 3


@pytest.mark.asyncio
async def test_assess_and_compose_event_notification_returns_not_relevant() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=False,
        notification_body="",
        reason="The article is general news, not an upcoming event.",
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.event._client", mock_client):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Quarterly earnings",
            body="The company reported higher revenue this quarter.",
            url="https://example.com/quarterly-earnings",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[],
            max_history_chars=100_000,
        )

    assert result.is_relevant_event is False
    assert result.notification_body == ""


@pytest.mark.asyncio
async def test_assess_and_compose_event_notification_detects_duplicate() -> None:
    parsed = EventAssessmentResult(
        is_relevant_event=False,
        notification_body="",
        reason="The user was already notified about this event.",
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.event._client", mock_client):
        from news_service.agents.event import assess_and_compose_event_notification

        result = await assess_and_compose_event_notification(
            headline="Severance finale reminder",
            body="Reminder: the finale airs next Friday.",
            url="https://example.com/severance-reminder",
            published_at=datetime(2026, 3, 14, 10, 0, tzinfo=UTC),
            raw_prompt="Notify me when new Severance episodes are announced",
            target_language="en",
            recent_notification_history=[
                "Title: Severance season finale\nSummary: Apple confirmed the finale release date."
            ],
            max_history_chars=100_000,
        )

    assert result.is_relevant_event is False


@pytest.mark.asyncio
async def test_render_recent_events_preview_returns_subject_and_body() -> None:
    parsed = RecentEventsPreviewDecision(
        selected_item_ids=["event-1"],
        subject="Что вы могли пропустить",
        body="- Лекция Станислава Дробышевского\nhttps://example.com/event-1",
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.event._client", mock_client):
        from news_service.agents.event import render_recent_events_preview

        result = await render_recent_events_preview(
            raw_prompt="Только лекции Дробышевского",
            target_language="ru",
            lookback_days=7,
            candidate_events=[
                "ID: event-1\nTitle: Лекция Станислава Дробышевского\n"
                "URL: https://example.com/event-1"
            ],
            recent_notifications=[],
        )

    assert result.selected_item_ids == ["event-1"]
    assert result.subject == "Что вы могли пропустить"
    assert "https://example.com/event-1" in result.body
