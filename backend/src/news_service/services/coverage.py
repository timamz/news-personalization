import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import DiscoveredFeedItem, discover_feeds
from news_service.db.vector_store import embed_text, find_similar_feeds
from news_service.models.rss_feed import RssFeed
from news_service.services.telegram import build_telegram_channel_url

logger = logging.getLogger(__name__)


async def ensure_topic_coverage(session: AsyncSession, topics: list[str]) -> list[RssFeed]:
    """Resolve and return fixed sources for the provided topics."""
    uncovered: list[str] = []
    selected: dict[uuid.UUID, RssFeed] = {}

    for topic in topics:
        topic_embedding = await embed_text(topic)
        matching_feeds = await find_similar_feeds(session, topic_embedding, limit=3)

        if matching_feeds:
            logger.info("Topic '%s' already covered by %d feed(s)", topic, len(matching_feeds))
            for feed in matching_feeds:
                selected[feed.id] = feed
        else:
            uncovered.append(topic)

    for feed in selected.values():
        feed.subscriber_count += 1

    if uncovered:
        logger.info("Discovering feeds for uncovered topics: %s", uncovered)
        discovered = await discover_feeds(uncovered)

        for feed_info in discovered:
            feed = await _register_feed(session, feed_info)
            selected[feed.id] = feed

    return list(selected.values())

async def ensure_telegram_channel_coverage(
    session: AsyncSession,
    channels: list[str],
    topics: list[str],
) -> list[RssFeed]:
    if not channels:
        return []

    topic_tags = topics or ["telegram"]
    topic_str = " ".join(topic_tags)
    topic_embedding = await embed_text(topic_str)
    resolved: dict[uuid.UUID, RssFeed] = {}

    for channel in channels:
        source_url = build_telegram_channel_url(channel)
        result = await session.execute(select(RssFeed).where(RssFeed.url == source_url))
        existing_feed = result.scalar_one_or_none()
        if existing_feed is not None:
            existing_feed.subscriber_count += 1
            existing_feed.is_active = True
            resolved[existing_feed.id] = existing_feed
            logger.info("Telegram channel source already exists: %s", source_url)
            continue

        feed = RssFeed(
            url=source_url,
            title=f"Telegram @{channel}",
            topic_tags=topic_tags,
            topic_embedding=topic_embedding,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        resolved[feed.id] = feed
        logger.info("Registered Telegram channel source: %s", source_url)

    return list(resolved.values())


async def _register_feed(session: AsyncSession, feed_info: DiscoveredFeedItem) -> RssFeed:
    existing_result = await session.execute(select(RssFeed).where(RssFeed.url == feed_info.url))
    existing_feed = existing_result.scalar_one_or_none()
    if existing_feed is not None:
        existing_feed.subscriber_count += 1
        existing_feed.is_active = True
        logger.info("Discovered source already exists: %s", feed_info.url)
        return existing_feed

    topic_str = " ".join(feed_info.topic_tags)
    embedding = await embed_text(topic_str)

    feed = RssFeed(
        url=feed_info.url,
        title=feed_info.title,
        topic_tags=feed_info.topic_tags,
        topic_embedding=embedding,
        is_active=True,
        subscriber_count=1,
    )
    session.add(feed)
    await session.flush()
    logger.info("Registered new source: %s (%s)", feed_info.url, feed_info.title)
    return feed
