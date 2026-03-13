from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.subscription import SubscriptionEditProposalResponse


@pytest.mark.asyncio
async def test_propose_subscription_edit_returns_structured_response() -> None:
    parsed = SubscriptionEditProposalResponse(
        canonical_prompt="Notify me when new episodes of Frieren and Apothecary Diaries air.",
        prompt_summary="Anime episode notifications",
        format_instructions="brief summary",
        change_summary="Added Frieren to the tracked anime list.",
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.agents.subscription_edit._client", mock_client):
        from news_service.agents.subscription_edit import propose_subscription_edit

        result = await propose_subscription_edit(
            canonical_prompt="Notify me when new episodes of Apothecary Diaries air.",
            format_instructions="brief summary",
            change_request="Also add Frieren.",
        )

    assert result == parsed


@pytest.mark.asyncio
async def test_propose_subscription_edit_raises_on_empty_response() -> None:
    mock_message = MagicMock()
    mock_message.parsed = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with (
        patch("news_service.agents.subscription_edit._client", mock_client),
        pytest.raises(ValueError, match="empty response"),
    ):
        from news_service.agents.subscription_edit import propose_subscription_edit

        await propose_subscription_edit(
            canonical_prompt="Notify me when new episodes of Apothecary Diaries air.",
            format_instructions="brief summary",
            change_request="Also add Frieren.",
        )
