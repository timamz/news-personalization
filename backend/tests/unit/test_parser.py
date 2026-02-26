from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.subscription import SubscriptionConfig


@pytest.fixture
def mock_config():
    return SubscriptionConfig(
        topics=["artificial intelligence", "machine learning"],
        schedule_cron="0 8 */3 * *",
        format_instructions="brief summary",
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

    assert result.topics == ["artificial intelligence", "machine learning"]
    assert result.schedule_cron == "0 8 */3 * *"
    assert result.format_instructions == "brief summary"


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
