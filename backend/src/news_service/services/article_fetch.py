"""Fetch an article URL and extract its readable body text.

Centralises the HTTP-get + HTML-strip logic used by both RSS ingestion
(where every polled entry is enriched with the full linked article) and
any other consumer that needs article body text. Callers supply the
character cap so ingest (which stores raw text) and tighter consumers
(e.g. an LLM tool) can share the same implementation.

Two fetch paths are available:

1. A plain ``httpx`` GET with a realistic Firefox ``User-Agent``. This is
   cheap, parallel-safe, and works for the majority of news sites.
2. A headless-Firefox fallback for JS-rendered pages, Cloudflare
   challenges, and aggressive UA-filter gates. It is opt-in via
   ``use_browser_fallback=True`` because each browser launch costs
   several seconds; ingest paths keep it off, the agent-facing
   ``fetch_page`` tool turns it on.

Example::

    text = await fetch_article_text(
        "https://example.com/article",
        timeout_seconds=15.0,
        max_chars=50_000,
        use_browser_fallback=True,
    )
    if text is None:
        # fall back to whatever stub the source gave us
        ...
"""

import asyncio
import contextlib
import logging
import shutil

import httpx
from bs4 import BeautifulSoup

from news_service.core.config import get_settings
from news_service.services.browser import (
    BROWSER_USER_AGENT,
    build_firefox_driver,
    create_socks_proxy_addon,
)

logger = logging.getLogger(__name__)
settings = get_settings()


async def fetch_article_text(
    url: str,
    *,
    timeout_seconds: float,
    max_chars: int,
    use_browser_fallback: bool = False,
) -> str | None:
    """Download ``url``, extract readable text, truncate to ``max_chars``.

    Returns the cleaned text, or ``None`` when the page cannot be fetched,
    returns a non-200 status, or yields no extractable text. The helper
    never raises on network or parse errors -- callers fall back to
    whatever stub they already have.

    When ``use_browser_fallback`` is ``True`` and the ``httpx`` path returns
    nothing, a headless Firefox render is attempted as a last resort. This
    costs several seconds per call and is therefore kept opt-in.
    """
    text = await _fetch_via_httpx(url, timeout_seconds=timeout_seconds, max_chars=max_chars)
    if text is not None:
        return text

    if not use_browser_fallback:
        return None

    return await asyncio.to_thread(
        _fetch_via_browser,
        url,
        timeout_seconds,
        max_chars,
    )


async def _fetch_via_httpx(
    url: str,
    *,
    timeout_seconds: float,
    max_chars: int,
) -> str | None:
    """Plain HTTP GET + HTML extract. Returns ``None`` on any failure."""
    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            proxy=settings.proxy_url,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any(
                marker in content_type for marker in ("html", "xml", "text/plain")
            ):
                return None
            html = response.text
    except Exception:
        logger.debug("Article fetch (httpx) failed for %s", url)
        return None

    return _extract_text(html, max_chars=max_chars)


def _fetch_via_browser(
    url: str,
    timeout_seconds: float,
    max_chars: int,
) -> str | None:
    """Headless-Firefox fallback for JS-rendered or bot-filtered pages.

    Always returns ``None`` instead of raising. Synchronous on purpose --
    Selenium's API is blocking; callers wrap this in ``asyncio.to_thread``.
    """
    driver = None
    addon_dir: str | None = None
    try:
        driver = build_firefox_driver(timeout_seconds)
        if settings.proxy_url:
            addon_dir = create_socks_proxy_addon(settings.proxy_url)
            driver.install_addon(addon_dir, temporary=True)
        driver.get(url)
        html = driver.page_source
    except Exception:
        logger.debug("Article fetch (browser) failed for %s", url)
        return None
    finally:
        if driver is not None:
            with contextlib.suppress(Exception):
                driver.quit()
        if addon_dir:
            shutil.rmtree(addon_dir, ignore_errors=True)

    return _extract_text(html, max_chars=max_chars)


def _extract_text(html: str, *, max_chars: int) -> str | None:
    """Run BeautifulSoup cleanup on raw HTML and return truncated text."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
