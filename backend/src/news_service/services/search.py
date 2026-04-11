"""Provider-agnostic web search.

Supports SearXNG (self-hosted, default) and OpenAI Responses API (legacy fallback).
The active provider is selected via the ``web_search_provider`` setting.
"""

import asyncio
import logging

import httpx

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SEARXNG_MAX_RESULTS = 10


async def search_web(query: str) -> str:
    """Search the web and return formatted results.

    Delegates to the provider configured in ``settings.web_search_provider``.
    """
    match settings.web_search_provider:
        case "searxng":
            return await _search_searxng(query)
        case "openai":
            return await _search_openai(query)
        case other:
            raise ValueError(
                f"Unknown web_search_provider: {other!r}. Supported values: 'searxng', 'openai'."
            )


async def _search_searxng(query: str) -> str:
    """Query a SearXNG instance and return results formatted for LLM consumption."""
    async with httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        proxy=settings.proxy_url,
    ) as client:
        response = await client.get(
            f"{settings.searxng_url}/search",
            params={
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
            },
        )
        response.raise_for_status()

    data = response.json()
    results = data.get("results", [])
    if not results:
        return f"No search results found for: {query}"

    lines: list[str] = []
    for r in results[:_SEARXNG_MAX_RESULTS]:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"- {title}: {url}\n  {snippet}")
    return "\n".join(lines)


async def _search_openai(query: str) -> str:
    """Legacy fallback: search via OpenAI Responses API with web_search tool."""
    from openai import OpenAI

    def _do_search() -> str:
        http_client = httpx.Client(proxy=settings.proxy_url) if settings.proxy_url else None
        client = OpenAI(
            api_key=settings.openai_api_key,
            http_client=http_client,
        )
        response = client.responses.create(
            model=settings.litellm_model.removeprefix("openai/"),
            tools=[{"type": "web_search"}],
            input=query,
        )
        return response.output_text

    return await asyncio.to_thread(_do_search)
