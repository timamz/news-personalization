import asyncio
import logging
import uuid
from dataclasses import dataclass

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
from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text, find_similar_feeds
from news_service.models.rss_feed import RssFeed
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.relevance import score_candidate
from news_service.services.source_descriptions import describe_source
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.twitter import build_twitter_account_url

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class _ScoredCandidate:
    feed: RssFeed | None  # set if already in DB
    discovered: DiscoveredSourceItem | None  # set if from web discovery
    url: str
    source_kind: SourceKind
    score: float
    sampled_texts: list[str]


async def ensure_prompt_coverage(
    session: AsyncSession,
    raw_prompt: str,
    raw_prompt_embedding: list[float],
) -> list[RssFeed]:
    target = settings.source_target_count

    # Phase 1: gather candidates from DB and web in parallel
    db_candidates, web_candidates = await asyncio.gather(
        find_similar_feeds(
            session,
            raw_prompt_embedding,
            threshold=settings.content_db_candidate_threshold,
            limit=target * 2,
        ),
        _discover_all(raw_prompt),
    )

    # Build candidate pool, dedupe by URL
    candidates: dict[str, _ScoredCandidate] = {}
    for feed in db_candidates:
        candidates[feed.url] = _ScoredCandidate(
            feed=feed,
            discovered=None,
            url=feed.url,
            source_kind=_feed_source_kind(feed.url),
            score=0.0,
            sampled_texts=[],
        )
    for item in web_candidates:
        if item.url not in candidates:
            candidates[item.url] = _ScoredCandidate(
                feed=None,
                discovered=item,
                url=item.url,
                source_kind=item.source_kind,
                score=0.0,
                sampled_texts=[],
            )

    if not candidates:
        logger.warning("No candidate sources found for prompt: %s", raw_prompt)
        return []

    # Phase 2: score all candidates in parallel
    candidate_list = list(candidates.values())
    scores = await asyncio.gather(
        *(score_candidate(c.url, c.source_kind, raw_prompt_embedding) for c in candidate_list)
    )
    for candidate, (score, sampled_texts) in zip(candidate_list, scores, strict=True):
        candidate.score = score
        candidate.sampled_texts = sampled_texts

    # Phase 3: rank and pick top K
    ranked = sorted(candidate_list, key=lambda c: c.score, reverse=True)
    selected: list[RssFeed] = []

    for candidate in ranked[:target]:
        if candidate.feed is not None:
            candidate.feed.subscriber_count += 1
            candidate.feed.is_active = True
            await _ensure_feed_profile(
                candidate.feed,
                source_kind=candidate.source_kind,
                fallback_title=candidate.feed.title or candidate.url,
                sample_content=candidate.sampled_texts,
            )
            selected.append(candidate.feed)
        elif candidate.discovered is not None:
            feed = await _register_feed(
                session, candidate.discovered, sample_content=candidate.sampled_texts
            )
            selected.append(feed)

    logger.info(
        "Selected %d/%d sources for prompt (top scores: %s)",
        len(selected),
        target,
        ", ".join(f"{c.score:.3f}" for c in ranked[:target]),
    )
    return selected


async def _discover_all(raw_prompt: str) -> list[DiscoveredSourceItem]:
    rss, telegram, reddit, twitter = await asyncio.gather(
        discover_rss_feeds(raw_prompt),
        discover_telegram_channels(raw_prompt),
        discover_reddit_subreddits(raw_prompt),
        discover_twitter_accounts(raw_prompt),
    )
    merged: dict[str, DiscoveredSourceItem] = {}
    for item in [*rss, *telegram, *reddit, *twitter]:
        merged.setdefault(item.url, item)
    return list(merged.values())


def _feed_source_kind(url: str) -> SourceKind:
    from news_service.services.reddit import extract_reddit_subreddit_from_url
    from news_service.services.telegram import extract_telegram_channel_from_url
    from news_service.services.twitter import extract_twitter_account_from_url

    if extract_telegram_channel_from_url(url) is not None:
        return "telegram_channel"
    if extract_reddit_subreddit_from_url(url) is not None:
        return "reddit_subreddit"
    if extract_twitter_account_from_url(url) is not None:
        return "twitter_account"
    return "rss"


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


async def _register_feed(
    session: AsyncSession,
    feed_info: DiscoveredSourceItem,
    *,
    sample_content: list[str] | None = None,
) -> RssFeed:
    existing_result = await session.execute(select(RssFeed).where(RssFeed.url == feed_info.url))
    existing_feed = existing_result.scalar_one_or_none()
    if existing_feed is not None:
        existing_feed.subscriber_count += 1
        existing_feed.is_active = True
        await _ensure_feed_profile(
            existing_feed,
            source_kind=feed_info.source_kind,
            fallback_title=feed_info.title,
            sample_content=sample_content,
        )
        logger.info("Discovered source already exists: %s", feed_info.url)
        return existing_feed

    description, embedding = await _build_feed_profile(
        source_kind=feed_info.source_kind,
        title=feed_info.title,
        url=feed_info.url,
        sample_content=sample_content,
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
