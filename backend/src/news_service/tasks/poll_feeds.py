import asyncio
import logging
from datetime import UTC, datetime

import feedparser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.db.session import get_task_session
from news_service.db.vector_store import embed_texts, upsert_news_item
from news_service.models.rss_feed import RssFeed
from news_service.services.telegram import extract_telegram_channel_from_url, fetch_telegram_posts
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="news_service.tasks.poll_feeds.poll_all_feeds")
def poll_all_feeds() -> dict:
    return asyncio.run(_poll_all_feeds())


async def _poll_all_feeds() -> dict:
    async with get_task_session() as session:
        result = await session.execute(select(RssFeed).where(RssFeed.is_active.is_(True)))
        feeds = list(result.scalars().all())

        total_new = 0
        for feed in feeds:
            count = await _poll_single_feed(session, feed)
            total_new += count

        await session.commit()

    return {"feeds_polled": len(feeds), "new_items": total_new}


async def _poll_single_feed(session: AsyncSession, feed: RssFeed) -> int:
    channel_handle = extract_telegram_channel_from_url(feed.url)
    if channel_handle is not None:
        return await _poll_single_telegram_channel(session, feed, channel_handle)

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
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed entries for feed %s",
            feed.url,
            extra={"feed_id": str(feed.id)},
        )
        return 0

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


async def _poll_single_telegram_channel(
    session: AsyncSession,
    feed: RssFeed,
    channel_handle: str,
) -> int:
    try:
        posts = await fetch_telegram_posts(channel_handle)
    except Exception:
        logger.exception(
            "Failed to parse Telegram channel @%s",
            channel_handle,
            extra={"feed_id": str(feed.id)},
        )
        return 0

    if not posts:
        return 0

    texts_to_embed = [post.body for post in posts]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed Telegram posts for @%s",
            channel_handle,
            extra={"feed_id": str(feed.id)},
        )
        return 0

    source_name = feed.title or f"Telegram @{channel_handle}"
    now = datetime.now(UTC)
    new_count = 0
    for post, embedding in zip(posts, embeddings, strict=True):
        headline = post.body.splitlines()[0][:200]
        item = await upsert_news_item(
            session,
            feed_id=feed.id,
            headline=headline or f"Telegram post from @{channel_handle}",
            body=post.body,
            url=post.url,
            source=source_name,
            published_at=post.published_at,
            fetched_at=now,
            embedding=embedding,
        )
        if item is not None:
            new_count += 1

    feed.last_polled_at = now
    logger.info(
        "Polled Telegram channel @%s: %d new items from %d posts",
        channel_handle,
        new_count,
        len(posts),
        extra={"feed_id": str(feed.id)},
    )
    return new_count
