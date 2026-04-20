"""Shared web tools for any ADK agent that searches the web.

Any agent that can call ``search_web`` should also be able to click
through to a result and read the page body -- otherwise it only ever
sees the search provider's short snippets and has to guess at what the source
actually contains. ``fetch_page`` is that counterpart: give the LLM
a URL, get back the readable article text.

Both the Source Finder and the Digest Writer wrap this helper as an
ADK tool. The ADK-facing docstring is written from the model's point
of view so the prompt surface is consistent across agents.
"""

import logging

from news_service.core.config import get_settings
from news_service.services.article_fetch import fetch_article_text

logger = logging.getLogger(__name__)
settings = get_settings()


async def fetch_page(url: str) -> str:
    """Fetch a web page and return its readable body text.

    Pair with search_web: when a search result looks promising (e.g.
    a "best RSS feeds for X" listicle, a curator's blog post, a news
    article you need full context on) call this to read the full page
    instead of relying on the short search-result snippet. Pass a single URL
    per call.

    Args:
        url: The page URL to fetch. Must be a full http(s) URL.

    Returns:
        Cleaned article text, truncated if long, or an error note on
        failure ("could not fetch <url>"). Never raises.
    """
    cleaned = (url or "").strip()
    if not cleaned.startswith(("http://", "https://")):
        return f"could not fetch {url}: must be a full http(s) URL."
    text = await fetch_article_text(
        cleaned,
        timeout_seconds=settings.article_fetch_timeout_seconds,
        max_chars=settings.article_body_max_chars,
        use_browser_fallback=True,
    )
    if text is None or not text.strip():
        return f"could not fetch {url}."
    return text
