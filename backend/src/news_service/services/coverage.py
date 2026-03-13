import asyncio
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import (
    DiscoveredSourceItem,
    SourceKind,
    discover_reddit_subreddits,
    discover_rss_feeds,
    discover_telegram_channels,
    discover_twitter_accounts,
)
from news_service.db.vector_store import embed_text, find_similar_feeds
from news_service.models.rss_feed import RssFeed
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.source_descriptions import describe_source
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.twitter import build_twitter_account_url

logger = logging.getLogger(__name__)


async def ensure_prompt_coverage(
    session: AsyncSession,
    raw_prompt: str,
    raw_prompt_embedding: list[float],
) -> list[RssFeed]:
    selected: dict[uuid.UUID, RssFeed] = {}
    matching_feeds = await find_similar_feeds(session, raw_prompt_embedding, limit=3)

    if matching_feeds:
        logger.info(
            "Prompt '%s' already covered by %d feed(s)",
            raw_prompt,
            len(matching_feeds),
        )
        for feed in matching_feeds:
            selected[feed.id] = feed

    for feed in selected.values():
        feed.subscriber_count += 1

    if not selected:
        logger.info("Discovering feeds for prompt: %s", raw_prompt)
        rss_sources, telegram_sources, reddit_sources, twitter_sources = await asyncio.gather(
            discover_rss_feeds(raw_prompt),
            discover_telegram_channels(raw_prompt),
            discover_reddit_subreddits(raw_prompt),
            discover_twitter_accounts(raw_prompt),
        )
        discovered = [*rss_sources, *telegram_sources, *reddit_sources, *twitter_sources]
        selected_urls = {feed.url for feed in selected.values()}
        deduplicated_discovered: dict[str, DiscoveredSourceItem] = {}

        for feed_info in discovered:
            if feed_info.url in selected_urls or feed_info.url in deduplicated_discovered:
                continue
            deduplicated_discovered[feed_info.url] = feed_info

        for feed_info in deduplicated_discovered.values():
            feed = await _register_feed(session, feed_info)
            selected[feed.id] = feed

    return list(selected.values())


async def ensure_telegram_channel_coverage(
    session: AsyncSession,
    channels: list[str],
) -> list[RssFeed]:
    if not channels:
        return []

    resolved: dict[uuid.UUID, RssFeed] = {}
    for channel in channels:
        source_url = build_telegram_channel_url(channel)
        result = await session.execute(select(RssFeed).where(RssFeed.url == source_url))
        existing_feed = result.scalar_one_or_none()
        if existing_feed is not None:
            existing_feed.subscriber_count += 1
            existing_feed.is_active = True
            await _ensure_feed_profile(
                existing_feed,
                source_kind="telegram_channel",
                fallback_title=f"Telegram @{channel}",
            )
            resolved[existing_feed.id] = existing_feed
            logger.info("Telegram channel source already exists: %s", source_url)
            continue

        title = f"Telegram @{channel}"
        description, embedding = await _build_feed_profile(
            source_kind="telegram_channel",
            title=title,
            url=source_url,
        )
        feed = RssFeed(
            url=source_url,
            title=title,
            source_description=description,
            source_description_embedding=embedding,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        resolved[feed.id] = feed
        logger.info("Registered Telegram channel source: %s", source_url)

    return list(resolved.values())


async def ensure_reddit_subreddit_coverage(
    session: AsyncSession,
    subreddits: list[str],
) -> list[RssFeed]:
    if not subreddits:
        return []

    resolved: dict[uuid.UUID, RssFeed] = {}
    for subreddit in subreddits:
        source_url = build_reddit_subreddit_url(subreddit)
        result = await session.execute(select(RssFeed).where(RssFeed.url == source_url))
        existing_feed = result.scalar_one_or_none()
        if existing_feed is not None:
            existing_feed.subscriber_count += 1
            existing_feed.is_active = True
            await _ensure_feed_profile(
                existing_feed,
                source_kind="reddit_subreddit",
                fallback_title=f"Reddit r/{subreddit}",
            )
            resolved[existing_feed.id] = existing_feed
            logger.info("Reddit subreddit source already exists: %s", source_url)
            continue

        title = f"Reddit r/{subreddit}"
        description, embedding = await _build_feed_profile(
            source_kind="reddit_subreddit",
            title=title,
            url=source_url,
        )
        feed = RssFeed(
            url=source_url,
            title=title,
            source_description=description,
            source_description_embedding=embedding,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        resolved[feed.id] = feed
        logger.info("Registered Reddit subreddit source: %s", source_url)

    return list(resolved.values())


async def ensure_twitter_account_coverage(
    session: AsyncSession,
    accounts: list[str],
) -> list[RssFeed]:
    if not accounts:
        return []

    resolved: dict[uuid.UUID, RssFeed] = {}
    for account in accounts:
        source_url = build_twitter_account_url(account)
        result = await session.execute(select(RssFeed).where(RssFeed.url == source_url))
        existing_feed = result.scalar_one_or_none()
        if existing_feed is not None:
            existing_feed.subscriber_count += 1
            existing_feed.is_active = True
            await _ensure_feed_profile(
                existing_feed,
                source_kind="twitter_account",
                fallback_title=f"X @{account}",
            )
            resolved[existing_feed.id] = existing_feed
            logger.info("Twitter/X account source already exists: %s", source_url)
            continue

        title = f"X @{account}"
        description, embedding = await _build_feed_profile(
            source_kind="twitter_account",
            title=title,
            url=source_url,
        )
        feed = RssFeed(
            url=source_url,
            title=title,
            source_description=description,
            source_description_embedding=embedding,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        resolved[feed.id] = feed
        logger.info("Registered Twitter/X account source: %s", source_url)

    return list(resolved.values())


async def _register_feed(session: AsyncSession, feed_info: DiscoveredSourceItem) -> RssFeed:
    existing_result = await session.execute(select(RssFeed).where(RssFeed.url == feed_info.url))
    existing_feed = existing_result.scalar_one_or_none()
    if existing_feed is not None:
        existing_feed.subscriber_count += 1
        existing_feed.is_active = True
        await _ensure_feed_profile(
            existing_feed,
            source_kind=feed_info.source_kind,
            fallback_title=feed_info.title,
        )
        logger.info("Discovered source already exists: %s", feed_info.url)
        return existing_feed

    description, embedding = await _build_feed_profile(
        source_kind=feed_info.source_kind,
        title=feed_info.title,
        url=feed_info.url,
    )
    feed = RssFeed(
        url=feed_info.url,
        title=feed_info.title,
        source_description=description,
        source_description_embedding=embedding,
        is_active=True,
        subscriber_count=1,
    )
    session.add(feed)
    await session.flush()
    logger.info("Registered new source: %s (%s)", feed_info.url, feed_info.title)
    return feed


async def _ensure_feed_profile(
    feed: RssFeed,
    *,
    source_kind: SourceKind,
    fallback_title: str,
) -> None:
    if feed.source_description and feed.source_description_embedding is not None:
        if not feed.title:
            feed.title = fallback_title
        return

    title = feed.title or fallback_title
    description, embedding = await _build_feed_profile(
        source_kind=source_kind,
        title=title,
        url=feed.url,
    )
    feed.title = title
    feed.source_description = description
    feed.source_description_embedding = embedding


async def _build_feed_profile(
    *,
    source_kind: SourceKind,
    title: str,
    url: str,
) -> tuple[str, list[float]]:
    description = await describe_source(source_kind=source_kind, title=title, url=url)
    embedding = await embed_text(description)
    return description, embedding
