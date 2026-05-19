import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.services.source_descriptions import describe_source

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_describe_source_returns_normalized_description(mocker) -> None:
    tag = uuid.uuid4().hex[:6]
    raw_description = f"  Научные  новости   из arXiv и статьи по ML. tag={tag}  "
    expected = f"Научные новости из arXiv и статьи по ML. tag={tag}"

    msg = MagicMock()
    msg.content = raw_description
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]

    mocker.patch(
        "news_service.services.source_descriptions.chat_completion",
        new=AsyncMock(return_value=completion),
    )

    result = await describe_source(
        source_kind="rss",
        title=f"arXiv cs.LG {uuid.uuid4().hex[:4]}",
        url=f"https://export.arxiv.org/rss/{uuid.uuid4().hex[:4]}",
    )

    assert result == expected, "describe_source did not normalize whitespace in description"
