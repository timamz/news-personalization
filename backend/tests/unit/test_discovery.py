import logging
import uuid
from unittest.mock import MagicMock

import pytest

from news_service.agents import discovery
from news_service.agents.discovery import search_web

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_search_web_returns_output_text(mocker) -> None:
    expected_text = f"Результаты поиска RSS-каналов об ИИ {uuid.uuid4().hex[:8]}"
    mock_response = MagicMock()
    mock_response.output_text = expected_text
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mocker.patch.object(discovery, "_sync_client", mock_client)

    query = f"лучшие RSS-каналы об ИИ {uuid.uuid4().hex[:6]}"
    result = await search_web(query)

    assert result == expected_text, "search_web did not return the expected output text"


@pytest.mark.asyncio
async def test_search_web_passes_query_to_client(mocker) -> None:
    mock_response = MagicMock()
    mock_response.output_text = f"Ответ {uuid.uuid4().hex[:6]}"
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mocker.patch.object(discovery, "_sync_client", mock_client)

    query = f"поиск источников {uuid.uuid4().hex[:6]}"
    await search_web(query)

    assert mock_client.responses.create.call_args.kwargs["input"] == query, (
        "search_web did not pass the query to the OpenAI client"
    )


@pytest.mark.asyncio
async def test_search_web_requests_web_search_tool(mocker) -> None:
    mock_response = MagicMock()
    mock_response.output_text = f"Результат {uuid.uuid4().hex[:6]}"
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mocker.patch.object(discovery, "_sync_client", mock_client)

    await search_web(f"запрос {uuid.uuid4().hex[:6]}")

    assert mock_client.responses.create.call_args.kwargs["tools"] == [{"type": "web_search"}], (
        "search_web did not request the web_search tool"
    )


@pytest.mark.asyncio
async def test_search_web_initializes_client_when_none(mocker) -> None:
    mock_response = MagicMock()
    mock_response.output_text = f"Результаты {uuid.uuid4().hex[:6]}"
    mock_client_instance = MagicMock()
    mock_client_instance.responses.create.return_value = mock_response
    mocker.patch.object(discovery, "_sync_client", None)
    mock_openai_cls = mocker.patch(
        "news_service.agents.discovery.OpenAI",
        return_value=mock_client_instance,
    )

    await search_web(f"тестовый запрос {uuid.uuid4().hex[:6]}")

    assert mock_openai_cls.called, (
        "search_web did not initialize OpenAI client when _sync_client was None"
    )
    discovery._sync_client = None
