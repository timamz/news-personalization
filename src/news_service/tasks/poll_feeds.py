import asyncio
import logging
from datetime import UTC, datetime

import feedparser
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.db.vector_store import embed_texts, upsert_news_item
from news_service.models.rss_feed import RssFeed
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="news_service.tasks.poll_feeds.poll_all_feeds")
def poll_all_feeds() -> dict:
    return asyncio.run(_poll_all_feeds())


async def _poll_all_feeds() -> dict:
    async with async_session_factory() as session:
        result = await session.execute(select(RssFeed).where(RssFeed.is_active.is_(True)))
        feeds = list(result.scalars().all())

        total_new = 0
        for feed in feeds:
            count = await _poll_single_feed(session, feed)
            total_new += count

        await session.commit()

    return {"feeds_polled": len(feeds), "new_items": total_new}


async def _poll_single_feed(session, feed: RssFeed) -> int:  # noqa: ANN001
    try:
        parsed = feedparser.parse(feed.url)
    except Exception:
        logger.exception("Failed to parse feed %s", feed.url, extra={"feed_id": str(feed.id)})
        return 0

    entries = parsed.entries
    if not entries:
        return 0

    headlines = [e.get("title", "") for e in entries]
    bodies = [e.get("summary", e.get("description", "")) for e in entries]
    texts_to_embed = [f"{h} {b}" for h, b in zip(headlines, bodies, strict=True)]
    embeddings = await embed_texts(texts_to_embed)

    new_count = 0
    for entry, embedding in zip(entries, embeddings, strict=True):
        url = entry.get("link", "")
        if not url:
            continue

        published_str = entry.get("published", None)
        published_at = None
        if published_str:
            try:
                import email.utils

                parsed_date = email.utils.parsedate_to_datetime(published_str)
                published_at = parsed_date.astimezone(UTC)
            except (ValueError, TypeError):
                pass

        item = await upsert_news_item(
            session,
            feed_id=feed.id,
            headline=entry.get("title", "Untitled"),
            body=entry.get("summary", entry.get("description", "")),
            url=url,
            source=feed.title or feed.url,
            published_at=published_at,
            fetched_at=datetime.now(UTC),
            embedding=embedding,
        )
        if item is not None:
            new_count += 1

    feed.last_polled_at = datetime.now(UTC)
    logger.info(
        "Polled feed %s: %d new items from %d entries",
        feed.url,
        new_count,
        len(entries),
        extra={"feed_id": str(feed.id)},
    )
    return new_count
