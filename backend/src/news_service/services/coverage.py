from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import SourceKind
from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text
from news_service.models.rss_feed import RssFeed
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.source_descriptions import describe_source
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.twitter import build_twitter_account_url

if TYPE_CHECKING:
    from news_service.agents.source_discovery import ScoredSource

logger = logging.getLogger(__name__)
settings = get_settings()


async def ensure_prompt_coverage(
    session: AsyncSession,
    raw_prompt: str,
    raw_prompt_embedding: list[float],
) -> list[RssFeed]:
    from news_service.agents.source_discovery import run_source_discovery

    try:
        result = await run_source_discovery(
            session=session,
            raw_prompt=raw_prompt,
            prompt_embedding=raw_prompt_embedding,
        )
    except Exception:
        logger.exception("Source discovery agent failed for prompt: %s", raw_prompt)
        return []

    selected: list[RssFeed] = []
    for source in result.sources:
        feed = await _register_or_reuse_source(session, source)
        if feed is not None:
            selected.append(feed)

    logger.info(
        "Selected %d sources for prompt (scores: %s)",
        len(selected),
        ", ".join(f"{s.relevance_score:.3f}" for s in result.sources),
    )
    return selected


async def _register_or_reuse_source(
    session: AsyncSession,
    source: ScoredSource,
) -> RssFeed | None:
    existing_result = await session.execute(select(RssFeed).where(RssFeed.url == source.url))
    existing_feed = existing_result.scalar_one_or_none()
    if existing_feed is not None:
        existing_feed.subscriber_count += 1
        existing_feed.is_active = True
        await _ensure_feed_profile(
            existing_feed,
            source_kind=source.source_kind,
            fallback_title=source.title or source.url,
        )
        return existing_feed

    description, embedding = await _build_feed_profile(
        source_kind=source.source_kind,
        title=source.title or source.url,
        url=source.url,
    )
    feed = RssFeed(
        url=source.url,
        title=source.title or source.url,
        source_description=description,
        source_description_embedding=embedding,
        is_active=True,
        subscriber_count=1,
    )
    session.add(feed)
    await session.flush()
    logger.info("Registered new source: %s (%s)", source.url, source.title)
    return feed


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


async def _ensure_feed_profile(
    feed: RssFeed,
    *,
    source_kind: SourceKind,
    fallback_title: str,
    sample_content: list[str] | None = None,
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
        sample_content=sample_content,
    )
    feed.title = title
    feed.source_description = description
    feed.source_description_embedding = embedding


async def _build_feed_profile(
    *,
    source_kind: SourceKind,
    title: str,
    url: str,
    sample_content: list[str] | None = None,
) -> tuple[str, list[float]]:
    description = await describe_source(
        source_kind=source_kind,
        title=title,
        url=url,
        sample_content=sample_content,
    )
    embedding = await embed_text(description)
    return description, embedding
