import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_describe_source_returns_normalized_description() -> None:
    tag = uuid.uuid4().hex[:6]
    raw_description = f"  Научные  новости   из arXiv и статьи по ML. tag={tag}  "
    expected = f"Научные новости из arXiv и статьи по ML. tag={tag}"

    parsed = MagicMock()
    parsed.description = raw_description

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
            title=f"arXiv cs.LG {uuid.uuid4().hex[:4]}",
            url=f"https://export.arxiv.org/rss/{uuid.uuid4().hex[:4]}",
        )

    assert result == expected, "describe_source did not normalize whitespace in description"


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
        pytest.raises(ValueError),
    ):
        from news_service.services.source_descriptions import describe_source

        await describe_source(
            source_kind="rss",
            title=f"arXiv cs.LG {uuid.uuid4().hex[:4]}",
            url=f"https://export.arxiv.org/rss/{uuid.uuid4().hex[:4]}",
        )
