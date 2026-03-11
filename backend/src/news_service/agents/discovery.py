import asyncio
import logging
from typing import Literal

import feedparser
import httpx
from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.services.telegram import (
    build_telegram_channel_url,
    extract_telegram_channel_from_url,
    fetch_telegram_posts,
    normalize_telegram_channel,
)

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

type SourceKind = Literal["rss", "telegram_channel"]

RSS_SYSTEM_PROMPT = """\
You are an RSS source discovery agent. Given a list of news topics, find real, working RSS or \
Atom feeds that cover those topics.

Rules:
- Return only RSS/Atom feed URLs, never homepages.
- Prefer high-signal, active sources from established publishers or niche experts.
- Find 2-4 good feeds per topic when possible.
- Do not return Telegram URLs or social profiles.
"""

TELEGRAM_SYSTEM_PROMPT = """\
You are a Telegram source discovery agent. Given a list of news topics, find real, working \
public Telegram channels that cover those topics.

Rules:
- Return only public Telegram channel archive URLs in the format https://t.me/s/<channel>.
- Prefer high-signal, active channels with frequent news or analysis posts.
- Find 2-4 good channels per topic when possible.
- Do not return RSS feeds, websites, invite links, group chats, or private channels.
"""


class DiscoveredSourceItem(BaseModel):
    url: str = Field(..., description="Canonical source URL")
    topic_tags: list[str] = Field(..., description="Topics this source covers")
    title: str = Field(default="", description="Human-readable source title")
    source_kind: SourceKind = Field(..., description="Source type")


class DiscoveredSourceList(BaseModel):
    sources: list[DiscoveredSourceItem] = Field(..., description="List of discovered sources")


# Backward-compatible aliases for older imports.
DiscoveredFeedItem = DiscoveredSourceItem
DiscoveredFeedList = DiscoveredSourceList


async def discover_sources(topics: list[str]) -> list[DiscoveredSourceItem]:
    rss_sources, telegram_sources = await asyncio.gather(
        discover_rss_feeds(topics),
        discover_telegram_channels(topics),
    )
    merged: dict[str, DiscoveredSourceItem] = {}
    for source in [*rss_sources, *telegram_sources]:
        merged.setdefault(source.url, source)
    return list(merged.values())


async def discover_feeds(topics: list[str]) -> list[DiscoveredSourceItem]:
    return await discover_sources(topics)


async def discover_rss_feeds(topics: list[str]) -> list[DiscoveredSourceItem]:
    return await _discover_sources_for_kind(
        topics,
        source_kind="rss",
        system_prompt=RSS_SYSTEM_PROMPT,
        user_prompt_prefix="Find RSS feeds for these topics:",
    )


async def discover_telegram_channels(topics: list[str]) -> list[DiscoveredSourceItem]:
    return await _discover_sources_for_kind(
        topics,
        source_kind="telegram_channel",
        system_prompt=TELEGRAM_SYSTEM_PROMPT,
        user_prompt_prefix="Find public Telegram channels for these topics:",
    )


async def _discover_sources_for_kind(
    topics: list[str],
    *,
    source_kind: SourceKind,
    system_prompt: str,
    user_prompt_prefix: str,
) -> list[DiscoveredSourceItem]:
    topics_str = ", ".join(topics)
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt_prefix} {topics_str}"},
        ],
        response_format=DiscoveredSourceList,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"LLM returned empty response for topics: {topics_str}")

    validated: list[DiscoveredSourceItem] = []
    seen_urls: set[str] = set()
    for source in result.sources:
        if source.source_kind != source_kind:
            logger.warning(
                "Discarded source with mismatched kind %s for requested kind %s: %s",
                source.source_kind,
                source_kind,
                source.url,
            )
            continue

        normalized_url = normalize_source_url(source.url, source_kind=source_kind)
        if normalized_url is None or normalized_url in seen_urls:
            logger.warning("Invalid source URL discarded: %s", source.url)
            continue

        is_valid = await validate_source_url(normalized_url, source_kind=source_kind)
        if not is_valid:
            logger.warning("Invalid source URL discarded: %s", normalized_url)
            continue

        seen_urls.add(normalized_url)
        validated.append(source.model_copy(update={"url": normalized_url}))
        logger.info("Validated %s source: %s (%s)", source_kind, normalized_url, source.title)

    return validated


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

    normalized = url.strip()
    if not normalized or extract_telegram_channel_from_url(normalized) is not None:
        return None
    return normalized


async def validate_source_url(url: str, *, source_kind: SourceKind) -> bool:
    if source_kind == "telegram_channel":
        channel = extract_telegram_channel_from_url(url)
        if channel is None:
            return False
        return await validate_telegram_channel(channel)
    return await validate_feed_url(url)


async def validate_feed_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
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
