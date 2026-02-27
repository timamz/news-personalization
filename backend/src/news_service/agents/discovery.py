import logging

import feedparser
import httpx
from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channel_from_url,
    fetch_telegram_posts,
)

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

SYSTEM_PROMPT = """\
You are a source discovery agent. Given a list of news topics, find real, working sources that \
cover those topics.

Rules:
- Return a mix of:
  1) RSS/Atom feeds from well-known sources.
  2) Public Telegram channels using URL format https://t.me/s/<channel>.
- For each topic, try to find 3-6 total sources.
- Prefer high-signal, active sources.
- For RSS sources, return full RSS/Atom feed URLs (not homepages).
- For Telegram sources, return public channel archive URLs only (https://t.me/s/<channel>).
"""


class DiscoveredFeedItem(BaseModel):
    url: str = Field(..., description="RSS feed URL")
    topic_tags: list[str] = Field(..., description="Topics this feed covers")
    title: str = Field(default="", description="Feed title")


class DiscoveredFeedList(BaseModel):
    feeds: list[DiscoveredFeedItem] = Field(..., description="List of discovered sources")


async def discover_feeds(topics: list[str]) -> list[DiscoveredFeedItem]:
    topics_str = ", ".join(topics)

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Find RSS feeds for these topics: {topics_str}"},
        ],
        response_format=DiscoveredFeedList,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"LLM returned empty response for topics: {topics_str}")

    validated = []
    for feed in result.feeds:
        normalized_url = normalize_source_url(feed.url)
        if normalized_url is None:
            logger.warning("Invalid source URL discarded: %s", feed.url)
            continue

        is_valid = await validate_source_url(normalized_url)
        if is_valid:
            validated.append(feed.model_copy(update={"url": normalized_url}))
            logger.info("Validated source: %s (%s)", normalized_url, feed.title)
            continue

        logger.warning("Invalid source URL discarded: %s", normalized_url)

    return validated


def normalize_source_url(url: str) -> str | None:
    channel = extract_telegram_channel_from_url(url)
    if channel is not None:
        return build_telegram_channel_url(channel)
    return url.strip()


async def validate_source_url(url: str) -> bool:
    channel = extract_telegram_channel_from_url(url)
    if channel is not None:
        return await validate_telegram_channel(channel)
    return await validate_feed_url(url)


async def validate_feed_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code != 200:
                return False

        parsed = feedparser.parse(response.text)
        return len(parsed.entries) > 0
    except (httpx.HTTPError, Exception):
        logger.exception("Feed validation failed for %s", url)
        return False


async def validate_telegram_channel(channel: str) -> bool:
    try:
        posts = await fetch_telegram_posts(channel)
        return len(posts) > 0
    except Exception:
        logger.exception("Telegram channel validation failed for @%s", channel)
        return False
