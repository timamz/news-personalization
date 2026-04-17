import logging
import uuid

import pytest

from news_service.services import search as search_module
from news_service.services.search import search_web

logging.disable(logging.CRITICAL)


def _patch_settings(mocker) -> None:
    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            searxng_url="http://test-searxng:8080",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )


def _mock_searxng_response(mocker, results: list[dict]) -> object:
    response = mocker.MagicMock()
    response.json.return_value = {"results": results}
    response.raise_for_status = mocker.MagicMock()

    client = mocker.AsyncMock()
    client.get.return_value = response
    client.__aenter__ = mocker.AsyncMock(return_value=client)
    client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=client)
    return client


@pytest.mark.asyncio
async def test_search_web_formats_results_and_passes_query_as_json_request(mocker) -> None:
    title = f"Result-{uuid.uuid4().hex[:8]}"
    url = f"https://{uuid.uuid4().hex[:8]}.example.com"
    snippet = f"Description {uuid.uuid4().hex[:8]}"
    query = f"search {uuid.uuid4().hex[:6]}"

    _patch_settings(mocker)
    client = _mock_searxng_response(mocker, [{"title": title, "url": url, "content": snippet}])

    result = await search_web(query)

    params = client.get.call_args.kwargs["params"]
    assert params["q"] == query and params["format"] == "json"
    assert title in result and url in result and snippet in result, (
        "search_web did not surface title, URL, and snippet in the formatted response"
    )


@pytest.mark.asyncio
async def test_search_web_returns_empty_marker_when_no_results(mocker) -> None:
    _patch_settings(mocker)
    _mock_searxng_response(mocker, [])

    result = await search_web(f"query {uuid.uuid4().hex[:6]}")
    assert "No search results found" in result


@pytest.mark.asyncio
async def test_search_web_limits_response_to_max_results(mocker) -> None:
    results = [
        {"title": f"Title-{i}", "url": f"https://r{i}.example.com", "content": f"Snippet-{i}"}
        for i in range(20)
    ]
    _patch_settings(mocker)
    _mock_searxng_response(mocker, results)

    result = await search_web(f"query {uuid.uuid4().hex[:6]}")
    assert "r9.example.com" in result and "r10.example.com" not in result, (
        "search_web did not cap the rendered result set at the configured maximum"
    )
