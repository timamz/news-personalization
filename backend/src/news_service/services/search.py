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
import time
import xml.etree.ElementTree as ET

import httpx

from news_service.core.concurrency import search_semaphore
from news_service.core.config import get_settings
from news_service.core.llm_usage import record_web_search
from news_service.core.provider_errors import classify_search_status
from news_service.services.admin_alerts import notify_provider_limit

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
    """Query the Yandex Search API and return results formatted for LLM consumption.

    Every dispatch -- successful, rate-limited, or network-failed --
    writes one ``call_type='web_search'`` row into the llm_usage ledger
    via ``record_web_search``. Rows carry the ``error`` string for
    non-200 outcomes so the economics report can separate real searches
    (which cost money on the Yandex side) from transient failures.
    """
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

    started = time.monotonic()
    error: str | None = None
    try:
        try:
            async with (
                search_semaphore,
                httpx.AsyncClient(**client_kwargs) as client,  # type: ignore[arg-type]
            ):
                response = await client.post(_YANDEX_SEARCH_ENDPOINT, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("Yandex search network error for query %r: %s", query[:60], exc)
            error = f"network_error: {exc}"
            return f"Search temporarily unavailable (network error) for: {query}"

        if response.status_code == 429:
            logger.warning("Yandex search rate-limited for query %r", query[:60])
            error = "rate_limited"
            body_snippet = _safe_body_snippet(response)
            limit_err = classify_search_status(status_code=429, body_snippet=body_snippet)
            if limit_err is not None:
                await notify_provider_limit(limit_err)
            return f"Search rate-limited; try a different phrasing later. Query was: {query}"
        if response.status_code in (401, 402, 403):
            body_snippet = _safe_body_snippet(response)
            logger.warning(
                "Yandex search auth/balance error %s for query %r: %s",
                response.status_code,
                query[:60],
                body_snippet[:200],
            )
            error = f"http_{response.status_code}"
            limit_err = classify_search_status(
                status_code=response.status_code, body_snippet=body_snippet
            )
            if limit_err is not None:
                await notify_provider_limit(limit_err)
                raise limit_err
            return f"Search failed (HTTP {response.status_code}) for: {query}"
        if 500 <= response.status_code < 600:
            logger.warning(
                "Yandex search upstream error %s for query %r", response.status_code, query[:60]
            )
            error = f"upstream_{response.status_code}"
            return f"Search temporarily unavailable (upstream {response.status_code}) for: {query}"
        if response.status_code != 200:
            logger.warning(
                "Yandex search unexpected status %s for query %r",
                response.status_code,
                query[:60],
            )
            error = f"http_{response.status_code}"
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
    finally:
        await record_web_search(
            latency_ms=int((time.monotonic() - started) * 1000),
            error=error,
        )


def _safe_body_snippet(response: httpx.Response, *, max_chars: int = 500) -> str:
    """Read up to ``max_chars`` of the response body without raising.

    Search errors typically include a short JSON envelope like
    ``{"code": ..., "message": ...}``; surfacing that to the admin
    alert gives them the actual reason (revoked key, missing role,
    insufficient balance) rather than a bare HTTP code.
    """
    try:
        text = response.text
    except Exception:
        return ""
    return text[:max_chars]
