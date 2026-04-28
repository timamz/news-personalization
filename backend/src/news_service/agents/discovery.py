import logging
from typing import Literal

import feedparser
import httpx

from news_service.core.config import get_settings
from news_service.services.reddit import (
    build_reddit_subreddit_url,
    extract_reddit_subreddit_from_url,
    fetch_reddit_posts,
    normalize_reddit_subreddit,
)
from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channel_from_url,
    fetch_telegram_posts,
    normalize_telegram_channel,
)

logger = logging.getLogger(__name__)

settings = get_settings()

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit"]


def normalize_source_url(url: str, *, source_kind: SourceKind) -> str | None:
    if source_kind == "telegram_channel":
        candidate = url.strip()
        channel = extract_telegram_channel_from_url(candidate)
        if channel is None and candidate.startswith("@"):
            try:
                channel = normalize_telegram_channel(candidate)
            except ValueError:
                return None
        if channel is None:
            return None
        return build_telegram_channel_url(channel)

    if source_kind == "reddit_subreddit":
        candidate = url.strip()
        subreddit = extract_reddit_subreddit_from_url(candidate)
        if subreddit is None:
            try:
                subreddit = normalize_reddit_subreddit(candidate)
            except ValueError:
                return None
        return build_reddit_subreddit_url(subreddit)

    normalized = url.strip()
    if (
        not normalized
        or extract_telegram_channel_from_url(normalized) is not None
        or extract_reddit_subreddit_from_url(normalized) is not None
    ):
        return None
    return normalized


async def validate_source_url(url: str, *, source_kind: SourceKind) -> bool:
    if source_kind == "telegram_channel":
        channel = extract_telegram_channel_from_url(url)
        if channel is None:
            return False
        return await validate_telegram_channel(channel)
    if source_kind == "reddit_subreddit":
        subreddit = extract_reddit_subreddit_from_url(url)
        if subreddit is None:
            return False
        return await validate_reddit_subreddit(subreddit)
    return await validate_feed_url(url)


async def validate_feed_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(
            timeout=settings.http_timeout_seconds,
            proxy=settings.proxy_url,
        ) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code != 200:
                return False

        parsed = feedparser.parse(response.text)
        return len(parsed.entries) > 0
    except Exception:
        logger.exception("Feed validation failed for %s", url)
        return False


async def validate_telegram_channel(channel: str) -> bool:
    try:
        posts = await fetch_telegram_posts(channel)
        return len(posts) > 0
    except Exception:
        logger.exception("Telegram channel validation failed for @%s", channel)
        return False


async def validate_reddit_subreddit(subreddit: str) -> bool:
    try:
        posts = await fetch_reddit_posts(subreddit)
        return len(posts) > 0
    except Exception:
        logger.exception("Reddit subreddit validation failed for r/%s", subreddit)
        return False
