import base64
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
            yandex_search_api_key=f"key-{uuid.uuid4().hex[:8]}",
            yandex_search_type="COM",
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )


def _encode_yandex_response(docs: list[dict]) -> str:
    groups = "".join(
        f"""
        <group>
          <doc>
            <url>{d["url"]}</url>
            <title>{d["title"]}</title>
            <passages><passage>{d["passage"]}</passage></passages>
          </doc>
        </group>
        """
        for d in docs
    )
    xml = f"""<?xml version="1.0"?>
    <yandexsearch version="1.0">
      <response>
        <results>
          <grouping>{groups}</grouping>
        </results>
      </response>
    </yandexsearch>"""
    return base64.b64encode(xml.encode("utf-8")).decode("ascii")


def _mock_yandex_response(mocker, raw_data: str | None, *, status_code: int = 200) -> object:
    response = mocker.MagicMock()
    response.status_code = status_code
    response.json.return_value = {} if raw_data is None else {"rawData": raw_data}

    client = mocker.AsyncMock()
    client.post.return_value = response
    client.__aenter__ = mocker.AsyncMock(return_value=client)
    client.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("news_service.services.search.httpx.AsyncClient", return_value=client)
    return client


@pytest.mark.asyncio
async def test_search_web_formats_title_url_and_passage_from_yandex_xml(mocker) -> None:
    title = f"Результат-{uuid.uuid4().hex[:8]}"
    url = f"https://{uuid.uuid4().hex[:8]}.example.com"
    passage = f"Описание {uuid.uuid4().hex[:8]}"
    query = f"search {uuid.uuid4().hex[:6]}"

    _patch_settings(mocker)
    _mock_yandex_response(
        mocker, _encode_yandex_response([{"url": url, "title": title, "passage": passage}])
    )

    result = await search_web(query)

    assert title in result and url in result and passage in result, (
        "search_web did not surface the Yandex title, URL, and passage in the formatted response"
    )


@pytest.mark.asyncio
async def test_search_web_sends_api_key_and_configured_search_type(mocker) -> None:
    api_key = f"key-{uuid.uuid4().hex}"
    search_type = "RU"
    query = f"query {uuid.uuid4().hex[:6]}"

    mocker.patch.object(
        search_module,
        "settings",
        mocker.MagicMock(
            yandex_search_api_key=api_key,
            yandex_search_type=search_type,
            http_timeout_seconds=10.0,
            proxy_url=None,
        ),
    )
    client = _mock_yandex_response(mocker, _encode_yandex_response([]))

    await search_web(query)

    call = client.post.call_args
    assert call.kwargs["headers"]["Authorization"] == f"Api-Key {api_key}", (
        "search_web did not forward the Yandex API key in the Authorization header"
    )
    assert call.kwargs["json"]["query"]["searchType"] == f"SEARCH_TYPE_{search_type}", (
        "search_web did not propagate the configured search type into the request payload"
    )
    assert call.kwargs["json"]["query"]["queryText"] == query, (
        "search_web did not pass the raw query text through to Yandex"
    )


@pytest.mark.asyncio
async def test_search_web_returns_empty_marker_when_yandex_payload_lacks_raw_data(mocker) -> None:
    _patch_settings(mocker)
    _mock_yandex_response(mocker, None)

    result = await search_web(f"query {uuid.uuid4().hex[:6]}")

    assert "No search results found" in result, (
        "search_web did not emit the empty-results marker when Yandex returned no rawData"
    )


@pytest.mark.asyncio
async def test_search_web_caps_rendered_groups_at_ten(mocker) -> None:
    docs = [
        {
            "url": f"https://r{i}.example.com",
            "title": f"Title-{i}",
            "passage": f"Passage-{i}",
        }
        for i in range(20)
    ]
    _patch_settings(mocker)
    _mock_yandex_response(mocker, _encode_yandex_response(docs))

    result = await search_web(f"query {uuid.uuid4().hex[:6]}")

    assert "r9.example.com" in result and "r10.example.com" not in result, (
        "search_web did not cap the rendered Yandex result set at the configured maximum"
    )


@pytest.mark.asyncio
async def test_search_web_strips_hlword_highlighting_from_title_and_passage(mocker) -> None:
    raw_data = base64.b64encode(
        b"""<?xml version="1.0"?>
        <yandexsearch version="1.0"><response><results><grouping><group><doc>
          <url>https://example.com/page</url>
          <title>Hello <hlword>world</hlword> today</title>
          <passages><passage>A <hlword>quick</hlword> fox jumps</passage></passages>
        </doc></group></grouping></results></response></yandexsearch>"""
    ).decode("ascii")

    _patch_settings(mocker)
    _mock_yandex_response(mocker, raw_data)

    result = await search_web(f"query {uuid.uuid4().hex[:6]}")

    assert "Hello world today" in result and "A quick fox jumps" in result, (
        "search_web did not strip <hlword> highlighting tags from title and passage text"
    )
    assert "<hlword>" not in result, (
        "search_web leaked raw <hlword> markup into the formatted output"
    )
