from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_describe_source_returns_normalized_description() -> None:
    parsed = MagicMock()
    parsed.description = "  Research  news   from arXiv and related ML papers.  "

    mock_message = MagicMock()
    mock_message.parsed = parsed

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with patch("news_service.services.source_descriptions._client", mock_client):
        from news_service.services.source_descriptions import describe_source

        result = await describe_source(
            source_kind="rss",
            title="arXiv cs.LG",
            url="https://export.arxiv.org/rss/cs.LG",
        )

    assert result == "Research news from arXiv and related ML papers."


@pytest.mark.asyncio
async def test_describe_source_raises_on_empty_result() -> None:
    mock_message = MagicMock()
    mock_message.parsed = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    with (
        patch("news_service.services.source_descriptions._client", mock_client),
        pytest.raises(ValueError, match="empty source description"),
    ):
        from news_service.services.source_descriptions import describe_source

        await describe_source(
            source_kind="rss",
            title="arXiv cs.LG",
            url="https://export.arxiv.org/rss/cs.LG",
        )
