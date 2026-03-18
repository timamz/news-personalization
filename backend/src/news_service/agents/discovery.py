import asyncio
import logging
from typing import Literal

import feedparser
import httpx
from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry
from news_service.core.openai_client import openai_client
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
from news_service.services.twitter import (
    build_twitter_account_url,
    extract_twitter_account_from_url,
    fetch_twitter_posts,
    normalize_twitter_account,
)

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit", "twitter_account"]

RSS_SYSTEM_PROMPT = """\
You are an RSS source discovery agent. Given a user's original news request, find real, working \
RSS or Atom feeds that are relevant to it.

Rules:
- Return only RSS/Atom feed URLs, never homepages.
- Prefer high-signal, active sources from established publishers or niche experts.
- Find 3-6 good feeds per topic when possible.
- Do not return Telegram URLs or social profiles.
"""

TELEGRAM_SYSTEM_PROMPT = """\
You are a Telegram source discovery agent. Given a user's original news request, find real, \
working public Telegram channels that are relevant to it.

Rules:
- Return only public Telegram channel archive URLs in the format https://t.me/s/<channel>.
- Prefer high-signal, active channels with frequent news or analysis posts.
- Find 3-6 good channels per topic when possible.
- Do not return RSS feeds, websites, invite links, group chats, or private channels.
"""

REDDIT_SYSTEM_PROMPT = """\
You are a Reddit source discovery agent. Given a user's original news request, find real, active \
public Reddit subreddits that are relevant to it.

Rules:
- Return only subreddit URLs in the format https://www.reddit.com/r/<subreddit>/new/.
- Prefer active communities where new topical posts appear regularly.
- Find 3-6 good subreddits per topic when possible.
- Do not return Reddit post URLs, users, or non-Reddit websites.
"""

TWITTER_SYSTEM_PROMPT = """\
You are a Twitter/X source discovery agent. Given a user's original news request, find real, \
active public Twitter/X accounts that are relevant to it.

Rules:
- Return only profile URLs in the format https://x.com/<account>.
- Prefer active accounts that post original news, announcements, or analysis.
- Find 3-6 good accounts per topic when possible.
- Do not return tweet URLs, lists, hashtags, or non-X websites.
"""


class DiscoveredSourceItem(BaseModel):
    url: str = Field(..., description="Canonical source URL")
    title: str = Field(default="", description="Human-readable source title")
    source_kind: SourceKind = Field(..., description="Source type")


class DiscoveredSourceList(BaseModel):
    sources: list[DiscoveredSourceItem] = Field(..., description="List of discovered sources")


# Backward-compatible aliases for older imports.
DiscoveredFeedItem = DiscoveredSourceItem
DiscoveredFeedList = DiscoveredSourceList


async def discover_sources(raw_prompt: str) -> list[DiscoveredSourceItem]:
    rss_sources, telegram_sources, reddit_sources, twitter_sources = await asyncio.gather(
        discover_rss_feeds(raw_prompt),
        discover_telegram_channels(raw_prompt),
        discover_reddit_subreddits(raw_prompt),
        discover_twitter_accounts(raw_prompt),
    )
    merged: dict[str, DiscoveredSourceItem] = {}
    for source in [*rss_sources, *telegram_sources, *reddit_sources, *twitter_sources]:
        merged.setdefault(source.url, source)
    return list(merged.values())


async def discover_feeds(raw_prompt: str) -> list[DiscoveredSourceItem]:
    return await discover_sources(raw_prompt)


async def discover_rss_feeds(raw_prompt: str) -> list[DiscoveredSourceItem]:
    return await _discover_sources_for_kind(
        raw_prompt,
        source_kind="rss",
        system_prompt=RSS_SYSTEM_PROMPT,
        user_prompt_prefix="The user wants to get:",
    )


async def discover_telegram_channels(raw_prompt: str) -> list[DiscoveredSourceItem]:
    return await _discover_sources_for_kind(
        raw_prompt,
        source_kind="telegram_channel",
        system_prompt=TELEGRAM_SYSTEM_PROMPT,
        user_prompt_prefix="The user wants to get:",
    )


async def discover_reddit_subreddits(raw_prompt: str) -> list[DiscoveredSourceItem]:
    return await _discover_sources_for_kind(
        raw_prompt,
        source_kind="reddit_subreddit",
        system_prompt=REDDIT_SYSTEM_PROMPT,
        user_prompt_prefix="The user wants to get:",
    )


async def discover_twitter_accounts(raw_prompt: str) -> list[DiscoveredSourceItem]:
    return await _discover_sources_for_kind(
        raw_prompt,
        source_kind="twitter_account",
        system_prompt=TWITTER_SYSTEM_PROMPT,
        user_prompt_prefix="The user wants to get:",
    )


@with_llm_retry()
async def _discover_sources_for_kind(
    raw_prompt: str,
    *,
    source_kind: SourceKind,
    system_prompt: str,
    user_prompt_prefix: str,
) -> list[DiscoveredSourceItem]:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt_prefix} {raw_prompt}"},
        ],
        response_format=DiscoveredSourceList,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"LLM returned empty response for prompt: {raw_prompt}")

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

    if source_kind == "reddit_subreddit":
        candidate = url.strip()
        subreddit = extract_reddit_subreddit_from_url(candidate)
        if subreddit is None:
            try:
                subreddit = normalize_reddit_subreddit(candidate)
            except ValueError:
                return None
        return build_reddit_subreddit_url(subreddit)

    if source_kind == "twitter_account":
        candidate = url.strip()
        account = extract_twitter_account_from_url(candidate)
        if account is None:
            try:
                account = normalize_twitter_account(candidate)
            except ValueError:
                return None
        return build_twitter_account_url(account)

    normalized = url.strip()
    if (
        not normalized
        or extract_telegram_channel_from_url(normalized) is not None
        or extract_reddit_subreddit_from_url(normalized) is not None
        or extract_twitter_account_from_url(normalized) is not None
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
    if source_kind == "twitter_account":
        account = extract_twitter_account_from_url(url)
        if account is None:
            return False
        return await validate_twitter_account(account)
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


async def validate_reddit_subreddit(subreddit: str) -> bool:
    try:
        posts = await fetch_reddit_posts(subreddit)
        return len(posts) > 0
    except Exception:
        logger.exception("Reddit subreddit validation failed for r/%s", subreddit)
        return False


async def validate_twitter_account(account: str) -> bool:
    try:
        posts = await fetch_twitter_posts(account, limit=1)
        return len(posts) > 0
    except Exception:
        logger.exception("Twitter/X account validation failed for @%s", account)
        return False
