from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.agents.event import ParsedEventConstraintValues, UpcomingEventCandidate
from news_service.schemas.subscription import EventConstraint


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
async def test_parse_event_constraint_values_returns_typed_mapping() -> None:
    parsed = ParsedEventConstraintValues(
        values=[
            {
                "key": "speaker_must_be_drobyshevsky",
                "string_value": "станислав владимирович дробышевский",
            },
            {
                "key": "is_other_person_speaking_under_brand",
                "boolean_value": False,
            },
        ]
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    constraints = [
        EventConstraint(
            key="speaker_must_be_drobyshevsky",
            description="Primary speaker identity",
            value_type="string",
            match_mode="exact",
            required_string="станислав владимирович дробышевский",
        ),
        EventConstraint(
            key="is_other_person_speaking_under_brand",
            description="Whether another person is speaking under the Drobyshevsky brand",
            value_type="boolean",
            match_mode="equals",
            required_boolean=False,
        ),
    ]

    with patch("news_service.agents.event._client", mock_client):
        from news_service.agents.event import parse_event_constraint_values

        result = await parse_event_constraint_values(
            headline="Новая лекция",
            body="Анонс новой лекции.",
            published_at=datetime(2026, 3, 13, 10, 0, tzinfo=UTC),
            raw_prompt="Только лекции Дробышевского",
            constraints=constraints,
        )

    assert result == {
        "speaker_must_be_drobyshevsky": "станислав владимирович дробышевский",
        "is_other_person_speaking_under_brand": False,
    }
