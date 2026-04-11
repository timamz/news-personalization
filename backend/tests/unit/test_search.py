import logging
import uuid

import pytest

from news_service.services import search as search_module
from news_service.services.search import search_web

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_search_web_with_searxng_returns_formatted_results(mocker) -> None:
    title = f"Результат-{uuid.uuid4().hex[:8]}"
    url = f"https://{uuid.uuid4().hex[:8]}.example.com"
    snippet = f"Описание источника {uuid.uuid4().hex[:8]}"

    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            web_search_provider="searxng",
            searxng_url="http://test-searxng:8080",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )

    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": title, "url": url, "content": snippet},
        ],
    }
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=mock_client)

    result = await search_web(f"запрос {uuid.uuid4().hex[:6]}")

    assert title in result, "search_web did not include result title"
    assert url in result, "search_web did not include result URL"
    assert snippet in result, "search_web did not include result snippet"


@pytest.mark.asyncio
async def test_search_web_with_searxng_passes_query_to_searxng(mocker) -> None:
    query = f"поиск источников {uuid.uuid4().hex[:6]}"

    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            web_search_provider="searxng",
            searxng_url="http://test-searxng:8080",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )

    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=mock_client)

    await search_web(query)

    call_kwargs = mock_client.get.call_args
    assert call_kwargs.kwargs["params"]["q"] == query, "search_web did not pass query to SearXNG"


@pytest.mark.asyncio
async def test_search_web_with_searxng_requests_json_format(mocker) -> None:
    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            web_search_provider="searxng",
            searxng_url="http://test-searxng:8080",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )

    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=mock_client)

    await search_web(f"запрос {uuid.uuid4().hex[:6]}")

    call_kwargs = mock_client.get.call_args
    assert call_kwargs.kwargs["params"]["format"] == "json", (
        "search_web did not request JSON format from SearXNG"
    )


@pytest.mark.asyncio
async def test_search_web_with_searxng_returns_no_results_message(mocker) -> None:
    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            web_search_provider="searxng",
            searxng_url="http://test-searxng:8080",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )

    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=mock_client)

    result = await search_web(f"запрос {uuid.uuid4().hex[:6]}")

    assert "No search results found" in result, "search_web did not return empty results message"


@pytest.mark.asyncio
async def test_search_web_with_searxng_limits_to_max_results(mocker) -> None:
    results = [
        {"title": f"Title-{i}", "url": f"https://r{i}.example.com", "content": f"Snippet-{i}"}
        for i in range(20)
    ]

    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            web_search_provider="searxng",
            searxng_url="http://test-searxng:8080",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )

    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"results": results}
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=mock_client)

    result = await search_web(f"запрос {uuid.uuid4().hex[:6]}")

    assert "r10.example.com" not in result, "search_web did not limit results to max count"
    assert "r9.example.com" in result, "search_web excluded result within max count"


@pytest.mark.asyncio
async def test_search_web_raises_on_unknown_provider(mocker) -> None:
    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            web_search_provider="unknown_provider",
        ),
    )

    with pytest.raises(ValueError, match="Unknown web_search_provider"):
        await search_web(f"запрос {uuid.uuid4().hex[:6]}")
