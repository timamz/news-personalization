from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.event import UpcomingEventCandidate


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
