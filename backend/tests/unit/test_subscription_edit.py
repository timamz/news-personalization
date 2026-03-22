import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.subscription import SubscriptionEditProposalResponse

logging.disable(logging.CRITICAL)


def _make_parsed_response(
    canonical_prompt: str,
    prompt_summary: str,
    change_summary: str,
) -> SubscriptionEditProposalResponse:
    return SubscriptionEditProposalResponse(
        canonical_prompt=canonical_prompt,
        prompt_summary=prompt_summary,
        format_instructions="краткое описание",
        change_summary=change_summary,
    )


@pytest.mark.asyncio
async def test_propose_subscription_edit_returns_structured_response() -> None:
    tag = uuid.uuid4().hex[:6]
    parsed = _make_parsed_response(
        canonical_prompt=f"Уведомляйте о новых сериях Фрирен и Аптекарских дневников. tag={tag}",
        prompt_summary=f"Уведомления об аниме-эпизодах {tag}",
        change_summary=f"Добавлена Фрирен в список отслеживаемых аниме. tag={tag}",
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
            canonical_prompt=f"Уведомляйте о новых сериях Аптекарских дневников. tag={tag}",
            format_instructions="краткое описание",
            change_request=f"Также добавьте Фрирен. tag={tag}",
        )

    assert result == parsed, "propose_subscription_edit did not return expected structured response"


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
        pytest.raises(ValueError),
    ):
        from news_service.agents.subscription_edit import propose_subscription_edit

        await propose_subscription_edit(
            canonical_prompt=f"Уведомляйте о новых сериях. {uuid.uuid4().hex[:4]}",
            format_instructions="краткое описание",
            change_request=f"Добавьте Фрирен. {uuid.uuid4().hex[:4]}",
        )
