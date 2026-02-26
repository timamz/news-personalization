import logging

from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import DiscoveredFeedItem, discover_feeds
from news_service.db.vector_store import embed_text, find_similar_feeds
from news_service.models.rss_feed import RssFeed

logger = logging.getLogger(__name__)


async def ensure_topic_coverage(session: AsyncSession, topics: list[str]) -> None:
    """Check each topic for existing RSS feed coverage; discover new feeds for gaps."""
    uncovered: list[str] = []

    for topic in topics:
        topic_embedding = await embed_text(topic)
        matching_feeds = await find_similar_feeds(session, topic_embedding, limit=3)

        if matching_feeds:
            logger.info("Topic '%s' already covered by %d feed(s)", topic, len(matching_feeds))
            for feed in matching_feeds:
                feed.subscriber_count += 1
        else:
            uncovered.append(topic)

    if not uncovered:
        return

    logger.info("Discovering feeds for uncovered topics: %s", uncovered)
    discovered = await discover_feeds(uncovered)

    for feed_info in discovered:
        await _register_feed(session, feed_info)


async def _register_feed(session: AsyncSession, feed_info: DiscoveredFeedItem) -> RssFeed:
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
    logger.info("Registered new RSS feed: %s (%s)", feed_info.url, feed_info.title)
    return feed
