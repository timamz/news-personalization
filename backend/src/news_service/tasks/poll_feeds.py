import asyncio
import logging
from datetime import UTC, datetime

import feedparser
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.event import extract_upcoming_event
from news_service.core.config import get_settings
from news_service.db.session import get_task_session
from news_service.db.vector_store import embed_texts, upsert_news_item
from news_service.models.news_item import NewsItem
from news_service.models.rss_feed import RssFeed
from news_service.services.telegram import extract_telegram_channel_from_url, fetch_telegram_posts
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

DELIVER_EVENTS_TASK = "news_service.tasks.deliver_events.deliver_event_notifications"
RSS_FETCH_TIMEOUT_SECONDS = settings.http_timeout_seconds
RSS_FETCH_ATTEMPTS = 2


@celery_app.task(name="news_service.tasks.poll_feeds.poll_all_feeds")
def poll_all_feeds() -> dict:
    return asyncio.run(_poll_all_feeds())


async def _poll_all_feeds() -> dict:
    async with get_task_session() as session:
        session.info["event_item_ids"] = []
        result = await session.execute(select(RssFeed).where(RssFeed.is_active.is_(True)))
        feeds = list(result.scalars().all())

        total_new = 0
        for feed in feeds:
            count = await _poll_single_feed(session, feed)
            total_new += count

        await session.commit()

        event_item_ids = list(dict.fromkeys(session.info.pop("event_item_ids", [])))

    for item_id in event_item_ids:
        celery_app.send_task(DELIVER_EVENTS_TASK, args=[str(item_id)])

    return {
        "feeds_polled": len(feeds),
        "new_items": total_new,
        "event_notifications_queued": len(event_item_ids),
    }


async def _poll_single_feed(session: AsyncSession, feed: RssFeed) -> int:
    channel_handle = extract_telegram_channel_from_url(feed.url)
    if channel_handle is not None:
        return await _poll_single_telegram_channel(session, feed, channel_handle)

    try:
        content = await _fetch_rss_feed_content(feed.url)
        parsed = feedparser.parse(content)
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
            if isinstance(item, NewsItem):
                await _maybe_store_upcoming_event(session, item)

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
            if isinstance(item, NewsItem):
                await _maybe_store_upcoming_event(session, item)

    feed.last_polled_at = now
    logger.info(
        "Polled Telegram channel @%s: %d new items from %d posts",
        channel_handle,
        new_count,
        len(posts),
        extra={"feed_id": str(feed.id)},
    )
    return new_count


async def _maybe_store_upcoming_event(session: AsyncSession, item: NewsItem) -> None:
    try:
        event = await extract_upcoming_event(item.headline, item.body, item.published_at)
    except Exception:
        logger.exception(
            "Failed to extract upcoming event from news item %s",
            item.id,
            extra={"news_item_id": str(item.id)},
        )
        return

    if event is None:
        return

    item.event_title = event.title or item.headline
    item.event_summary = event.summary or item.headline
    item.event_starts_at = event.starts_at
    session.info.setdefault("event_item_ids", []).append(item.id)


async def _fetch_rss_feed_content(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=RSS_FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        last_error: httpx.HTTPError | None = None
        for attempt in range(1, RSS_FETCH_ATTEMPTS + 1):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == RSS_FETCH_ATTEMPTS:
                    break
                logger.warning(
                    "RSS fetch attempt %d/%d failed for %s; retrying",
                    attempt,
                    RSS_FETCH_ATTEMPTS,
                    url,
                )

    if last_error is None:
        raise RuntimeError(f"RSS fetch failed without HTTP error for {url}")
    raise last_error
