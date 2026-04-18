"""Fetch an article URL and extract its readable body text.

Centralises the HTTP-get + HTML-strip logic used by both RSS ingestion
(where every polled entry is enriched with the full linked article) and
any other consumer that needs article body text. Callers supply the
character cap so ingest (which stores raw text) and tighter consumers
(e.g. an LLM tool) can share the same implementation.

Example::

    text = await fetch_article_text(
        "https://example.com/article",
        timeout_seconds=15.0,
        max_chars=50_000,
    )
    if text is None:
        # fall back to whatever stub the source gave us
        ...
"""

import logging

import httpx
from bs4 import BeautifulSoup

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def fetch_article_text(
    url: str,
    *,
    timeout_seconds: float,
    max_chars: int,
) -> str | None:
    """Download ``url``, extract readable text, truncate to ``max_chars``.

    Returns the cleaned text, or ``None`` when the page cannot be fetched,
    returns a non-200 status, or yields no extractable text. The helper
    never raises on network or parse errors -- callers fall back to
    whatever stub they already have.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            proxy=settings.proxy_url,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None
            html = response.text
    except Exception:
        logger.debug("Article fetch failed for %s", url)
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        logger.debug("Article parse failed for %s", url)
        return None

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
