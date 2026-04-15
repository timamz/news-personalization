"""Source adapters that normalize posts from different source types into a uniform format.

Each adapter wraps a source-specific fetch function and converts its output
into NormalizedPost instances that the generic polling loop can process
identically regardless of source type.
"""

import email.utils
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import feedparser
import httpx

from news_service.core.config import get_settings
from news_service.models.source import Source
from news_service.services.reddit import fetch_reddit_posts
from news_service.services.telegram import fetch_telegram_posts
from news_service.services.twitter import fetch_twitter_posts

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
        posts: list[NormalizedPost] = []
        for entry in parsed.entries:
            url = entry.get("link", "")
            if not url:
                continue
            headline = entry.get("title", "")
            body = entry.get("summary", entry.get("description", ""))
            published_at = _published_at_from_rss_entry(entry)
            posts.append(
                NormalizedPost(
                    url=url,
                    headline=headline,
                    body=body,
                    text_to_embed=f"{headline} {body}",
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
        posts: list[NormalizedPost] = []
        for post in raw_posts:
            headline = post.title[:200] or f"Reddit post from r/{self._subreddit}"
            text_to_embed = "\n\n".join(part for part in [post.title, post.body] if part).strip()
            posts.append(
                NormalizedPost(
                    url=post.url,
                    headline=headline,
                    body=post.body,
                    text_to_embed=text_to_embed,
                    published_at=post.published_at,
                )
            )
        return posts


class TwitterAdapter:
    """Adapter for Twitter/X accounts."""

    def __init__(self, src: Source, account: str) -> None:
        self._src = src
        self._account = account

    def source_name(self) -> str:
        return self._src.title or f"X @{self._account}"

    def log_label(self) -> str:
        return f"Twitter/X account @{self._account}"

    async def fetch_posts(self) -> list[NormalizedPost]:
        raw_posts = await fetch_twitter_posts(self._account)
        posts: list[NormalizedPost] = []
        for post in raw_posts:
            lines = post.body.splitlines()
            headline = lines[0][:200] if lines else f"Post from @{self._account}"
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
