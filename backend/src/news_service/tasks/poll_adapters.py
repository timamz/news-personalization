"""Source adapters that normalize posts from different source types into a uniform format.

Each adapter wraps a source-specific fetch function and converts its output
into NormalizedPost instances that the generic polling loop can process
identically regardless of source type.
"""

import asyncio
import email.utils
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import feedparser
import httpx

from news_service.core.config import get_settings
from news_service.models.source import Source
from news_service.services.article_fetch import fetch_article_text
from news_service.services.reddit import fetch_reddit_posts
from news_service.services.telegram import fetch_telegram_posts

logger = logging.getLogger(__name__)
settings = get_settings()

RSS_FETCH_TIMEOUT_SECONDS = settings.http_timeout_seconds
RSS_FETCH_ATTEMPTS = 2


@dataclass(slots=True)
class NormalizedPost:
    """Uniform representation of a post from any source type."""

    url: str
    headline: str
    body: str
    text_to_embed: str
    published_at: datetime | None


class SourceAdapter(Protocol):
    """Protocol for source-specific fetch and normalization logic."""

    async def fetch_posts(self) -> list[NormalizedPost]: ...

    def source_name(self) -> str: ...

    def log_label(self) -> str: ...


class RssAdapter:
    """Adapter for RSS/Atom feeds via feedparser."""

    def __init__(self, src: Source) -> None:
        self._src = src
        self._url = src.url

    def source_name(self) -> str:
        return self._src.title or self._url

    def log_label(self) -> str:
        return f"RSS feed {self._url}"

    async def fetch_posts(self) -> list[NormalizedPost]:
        content = await _fetch_rss_feed_content(self._url)
        parsed = feedparser.parse(content)

        entries: list[tuple[str, str, str, datetime | None]] = []
        for entry in parsed.entries:
            url = entry.get("link", "")
            if not url:
                continue
            headline = entry.get("title", "")
            stub = entry.get("summary", entry.get("description", ""))
            entries.append((url, headline, stub, _published_at_from_rss_entry(entry)))

        sem = asyncio.Semaphore(settings.article_fetch_concurrency)

        async def _enrich(url: str) -> str | None:
            async with sem:
                return await fetch_article_text(
                    url,
                    timeout_seconds=settings.article_fetch_timeout_seconds,
                    max_chars=settings.article_body_max_chars,
                )

        article_bodies = await asyncio.gather(*(_enrich(u) for u, *_ in entries))

        posts: list[NormalizedPost] = []
        for (url, headline, stub, published_at), article in zip(
            entries, article_bodies, strict=True
        ):
            body = article if article else stub
            posts.append(
                NormalizedPost(
                    url=url,
                    headline=headline,
                    body=body,
                    text_to_embed=f"{headline}\n\n{body}".strip(),
                    published_at=published_at,
                )
            )
        return posts


class TelegramAdapter:
    """Adapter for public Telegram channels."""

    def __init__(self, src: Source, channel_handle: str) -> None:
        self._src = src
        self._channel = channel_handle

    def source_name(self) -> str:
        return self._src.title or f"Telegram @{self._channel}"

    def log_label(self) -> str:
        return f"Telegram channel @{self._channel}"

    async def fetch_posts(self) -> list[NormalizedPost]:
        raw_posts = await fetch_telegram_posts(self._channel)
        posts: list[NormalizedPost] = []
        for post in raw_posts:
            lines = post.body.splitlines()
            headline = lines[0][:200] if lines else f"Telegram post from @{self._channel}"
            posts.append(
                NormalizedPost(
                    url=post.url,
                    headline=headline,
                    body=post.body,
                    text_to_embed=post.body,
                    published_at=post.published_at,
                )
            )
        return posts


class RedditAdapter:
    """Adapter for Reddit subreddits."""

    def __init__(self, src: Source, subreddit: str) -> None:
        self._src = src
        self._subreddit = subreddit

    def source_name(self) -> str:
        return self._src.title or f"Reddit r/{self._subreddit}"

    def log_label(self) -> str:
        return f"Reddit subreddit r/{self._subreddit}"

    async def fetch_posts(self) -> list[NormalizedPost]:
        raw_posts = await fetch_reddit_posts(self._subreddit)

        sem = asyncio.Semaphore(settings.article_fetch_concurrency)

        async def _enrich(post) -> str | None:
            if post.body or not post.external_url:
                return None
            async with sem:
                return await fetch_article_text(
                    post.external_url,
                    timeout_seconds=settings.article_fetch_timeout_seconds,
                    max_chars=settings.article_body_max_chars,
                )

        article_bodies = await asyncio.gather(*(_enrich(p) for p in raw_posts))

        posts: list[NormalizedPost] = []
        for post, article in zip(raw_posts, article_bodies, strict=True):
            body = post.body or (article or "")
            headline = post.title[:200] or f"Reddit post from r/{self._subreddit}"
            text_to_embed = "\n\n".join(part for part in [post.title, body] if part).strip()
            posts.append(
                NormalizedPost(
                    url=post.url,
                    headline=headline,
                    body=body,
                    text_to_embed=text_to_embed,
                    published_at=post.published_at,
                )
            )
        return posts


def _published_at_from_rss_entry(entry: object) -> datetime | None:
    """Parse publication date from an RSS entry."""
    published_str = entry.get("published", None)
    if not published_str:
        return None
    try:
        parsed_date = email.utils.parsedate_to_datetime(published_str)
        return parsed_date.astimezone(UTC)
    except (ValueError, TypeError):
        return None


async def _fetch_rss_feed_content(url: str) -> bytes:
    """Fetch RSS feed content with retry logic."""
    async with httpx.AsyncClient(
        timeout=RSS_FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        proxy=settings.proxy_url,
    ) as client:
        last_error: httpx.HTTPError | None = None
        for attempt in range(1, RSS_FETCH_ATTEMPTS + 1):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == RSS_FETCH_ATTEMPTS:
                    break
                logger.warning(
                    "RSS fetch attempt %d/%d failed for %s; retrying",
                    attempt,
                    RSS_FETCH_ATTEMPTS,
                    url,
                )

    if last_error is None:
        raise RuntimeError(f"RSS fetch failed without HTTP error for {url}")
    raise last_error
