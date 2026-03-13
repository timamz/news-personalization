from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.subscription import SubscriptionConfig


@pytest.fixture
def mock_config():
    return SubscriptionConfig(
        prompt_summary="AI news every third day",
        delivery_mode="digest",
        event_matching_mode="basic",
        schedule_cron="0 8 */3 * *",
        schedule_was_explicit=True,
        format_instructions="brief summary",
        digest_language="en",
    )


@pytest.mark.asyncio
async def test_parse_subscription_returns_config(mock_config):
    mock_message = MagicMock()
    mock_message.parsed = mock_config

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.parser._client", mock_client):
        from news_service.agents.parser import parse_subscription

        result = await parse_subscription("I want AI news every third day in the morning")

    assert result.delivery_mode == "digest"
    assert result.event_matching_mode == "basic"
    assert result.prompt_summary == "AI news every third day"
    assert result.schedule_cron == "0 8 */3 * *"
    assert result.schedule_was_explicit is True
    assert result.format_instructions == "brief summary"
    assert result.digest_language == "en"


@pytest.mark.asyncio
async def test_parse_subscription_raises_on_empty():
    mock_message = MagicMock()
    mock_message.parsed = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with (
        patch("news_service.agents.parser._client", mock_client),
        pytest.raises(ValueError, match="empty parsed response"),
    ):
        from news_service.agents.parser import parse_subscription

        await parse_subscription("something")


@pytest.mark.asyncio
async def test_parse_schedule_preference_returns_cron():
    parsed_schedule = MagicMock()
    parsed_schedule.schedule_cron = "0 9 * * 1-5"

    mock_message = MagicMock()
    mock_message.parsed = parsed_schedule

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.parser._client", mock_client):
        from news_service.agents.parser import parse_schedule_preference

        result = await parse_schedule_preference("every weekday at 9")

    assert result == "0 9 * * 1-5"
