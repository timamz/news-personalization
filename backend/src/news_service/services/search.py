"""Web search via the Yandex Search API (REST).

Calls the Yandex Cloud Search API endpoint with an ``Api-Key`` credential and
returns up to ten results formatted for LLM consumption. The Search API
responds with a JSON envelope whose ``rawData`` field is a base64-encoded XML
document; this module decodes and parses it into the same line-per-result
format the agents already expect.

Throughput is capped by a module-level semaphore so parallel finders cannot
burst past Yandex's per-second rate limit. Transient errors (429, 5xx,
network) are surfaced to the caller as a plain string so the ReAct loop
treats them as "this search came back empty" rather than crashing the
whole strategy.
"""

import base64
import logging
import xml.etree.ElementTree as ET

import httpx

from news_service.core.concurrency import search_semaphore
from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_YANDEX_SEARCH_ENDPOINT = "https://searchapi.api.cloud.yandex.net/v2/web/search"
_MAX_RESULTS = 10


def _element_text(element: ET.Element | None) -> str:
    """Return concatenated text of an element, stripping nested markup such as ``<hlword>``."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


async def search_web(query: str) -> str:
    """Query the Yandex Search API and return results formatted for LLM consumption."""
    payload = {
        "query": {
            "searchType": f"SEARCH_TYPE_{settings.yandex_search_type}",
            "queryText": query,
        },
    }
    headers = {
        "Authorization": f"Api-Key {settings.yandex_search_api_key}",
        "Content-Type": "application/json",
    }

    client_kwargs: dict[str, object] = {"timeout": settings.http_timeout_seconds}
    if settings.proxy_url:
        client_kwargs["proxy"] = settings.proxy_url

    try:
        async with (
            search_semaphore,
            httpx.AsyncClient(**client_kwargs) as client,  # type: ignore[arg-type]
        ):
            response = await client.post(_YANDEX_SEARCH_ENDPOINT, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("Yandex search network error for query %r: %s", query[:60], exc)
        return f"Search temporarily unavailable (network error) for: {query}"

    if response.status_code == 429:
        logger.warning("Yandex search rate-limited for query %r", query[:60])
        return f"Search rate-limited; try a different phrasing later. Query was: {query}"
    if 500 <= response.status_code < 600:
        logger.warning(
            "Yandex search upstream error %s for query %r", response.status_code, query[:60]
        )
        return f"Search temporarily unavailable (upstream {response.status_code}) for: {query}"
    if response.status_code != 200:
        logger.warning(
            "Yandex search unexpected status %s for query %r", response.status_code, query[:60]
        )
        return f"Search failed (HTTP {response.status_code}) for: {query}"

    raw_data = response.json().get("rawData")
    if not raw_data:
        return f"No search results found for: {query}"

    root = ET.fromstring(base64.b64decode(raw_data))

    lines: list[str] = []
    for group in root.iter("group"):
        if len(lines) >= _MAX_RESULTS:
            break
        doc = group.find("doc")
        if doc is None:
            continue
        url = _element_text(doc.find("url"))
        title = _element_text(doc.find("title"))
        if not url or not title:
            continue
        passages = doc.find("passages")
        snippet = _element_text(passages.find("passage")) if passages is not None else ""
        lines.append(f"- {title}: {url}\n  {snippet}")

    if not lines:
        return f"No search results found for: {query}"
    return "\n".join(lines)
