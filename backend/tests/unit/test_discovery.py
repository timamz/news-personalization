from unittest.mock import MagicMock

import pytest

from news_service.agents import discovery
from news_service.agents.discovery import search_web


@pytest.mark.asyncio
async def test_search_web_returns_output_text(mocker) -> None:
    mock_response = MagicMock()
    mock_response.output_text = "Here are some RSS feeds about AI: https://example.com/rss.xml"

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    mocker.patch.object(discovery, "_sync_client", mock_client)

    result = await search_web("best RSS feeds about AI")

    assert result == "Here are some RSS feeds about AI: https://example.com/rss.xml"
    mock_client.responses.create.assert_called_once_with(
        model=discovery.settings.llm_model,
        tools=[{"type": "web_search"}],
        input="best RSS feeds about AI",
    )


@pytest.mark.asyncio
async def test_search_web_initializes_client_when_none(mocker) -> None:
    mock_response = MagicMock()
    mock_response.output_text = "Search results"

    mock_client_instance = MagicMock()
    mock_client_instance.responses.create.return_value = mock_response

    mocker.patch.object(discovery, "_sync_client", None)
    mock_openai_cls = mocker.patch(
        "news_service.agents.discovery.OpenAI",
        return_value=mock_client_instance,
    )

    result = await search_web("test query")

    assert result == "Search results"
    mock_openai_cls.assert_called_once_with(api_key=discovery.settings.openai_api_key)

    # Reset the global so other tests are not affected.
    discovery._sync_client = None
