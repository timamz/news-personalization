from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.event import (
    EventMatchDecision,
    LocalizedEventText,
    NotificationDuplicateDecision,
    RecentEventsPreviewDecision,
    UpcomingEventCandidate,
)


@pytest.mark.asyncio
async def test_extract_upcoming_event_returns_candidate() -> None:
    parsed = UpcomingEventCandidate(
        is_upcoming_event=True,
        title="Severance season finale",
        summary="Apple confirmed the finale release date.",
        starts_at=datetime(2026, 3, 20, 0, 0, tzinfo=UTC),
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
        from news_service.agents.event import extract_upcoming_event

        result = await extract_upcoming_event(
            "Severance finale announced",
            "The new episode arrives next Friday.",
            datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
        )

    assert result is not None
    assert result.title == "Severance season finale"
    assert result.starts_at == datetime(2026, 3, 20, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_extract_upcoming_event_returns_none_for_regular_news() -> None:
    parsed = UpcomingEventCandidate(
        is_upcoming_event=False,
        title=None,
        summary=None,
        starts_at=None,
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
        from news_service.agents.event import extract_upcoming_event

        result = await extract_upcoming_event(
            "Quarterly earnings",
            "The company reported higher revenue this quarter.",
            datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
        )

    assert result is None


@pytest.mark.asyncio
async def test_judge_event_match_returns_decision() -> None:
    parsed = EventMatchDecision(
        matches=True,
        reason="The post explicitly announces a lecture by the requested person.",
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
        from news_service.agents.event import judge_event_match

        result = await judge_event_match(
            headline="Новая лекция",
            body="Приглашаем на новую лекцию.",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Только лекции Дробышевского",
            event_title="Лекция Станислава Дробышевского",
        )

    assert result.matches is True
    assert "requested person" in result.reason


@pytest.mark.asyncio
async def test_judge_notification_duplicate_returns_decision() -> None:
    parsed = NotificationDuplicateDecision(
        already_notified=True,
        reason="The history already contains the same lecture announcement.",
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
        from news_service.agents.event import judge_notification_duplicate

        result = await judge_notification_duplicate(
            headline="Новая лекция",
            body="Приглашаем на новую лекцию.",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            recent_notifications=[
                "Shown at: 2026-03-12T10:00:00+00:00\nEvent: Лекция Станислава Дробышевского"
            ],
            event_title="Лекция Станислава Дробышевского",
        )

    assert result.already_notified is True
    assert "same lecture announcement" in result.reason


@pytest.mark.asyncio
async def test_localize_event_text_returns_localized_fields() -> None:
    parsed = LocalizedEventText(
        title="Предстоящая лекция Станислава Дробышевского",
        summary="Лекция Станислава Дробышевского пройдет на следующей неделе.",
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
        from news_service.agents.event import localize_event_text

        result = await localize_event_text(
            headline="Stanislav Drobyshevsky lecture announced",
            body="A new lecture was announced for next week.",
            event_title="Stanislav Drobyshevsky lecture",
            event_summary="A new lecture was announced for next week.",
            event_starts_at=datetime(2026, 3, 20, 19, 0, tzinfo=UTC),
            target_language="ru",
        )

    assert result.title == "Предстоящая лекция Станислава Дробышевского"
    assert "следующей неделе" in result.summary


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
            event_matching_mode="strict_with_prefilter",
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
