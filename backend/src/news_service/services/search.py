"""Web search via a self-hosted SearXNG instance."""

import logging

import httpx

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SEARXNG_MAX_RESULTS = 10


async def search_web(query: str) -> str:
    """Query SearXNG and return results formatted for LLM consumption."""
    # SearXNG is reached over the internal Docker network (e.g. http://searxng:8080),
    # so the SOCKS5 proxy used for external traffic must NOT be applied here -- it
    # cannot resolve the compose-internal hostname.
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
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
