"""Source polling task — fetches new items from all active sources.

Uses the adapter pattern to handle different source types (RSS, Telegram,
Reddit, Twitter) through a single generic polling loop.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.guardrails import cap_text_for_embedding, scan_for_injection
from news_service.db.session import get_task_session
from news_service.db.vector_store import embed_texts, upsert_news_item
from news_service.models.news_item import NewsItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.reddit import extract_reddit_subreddit_from_url
from news_service.services.telegram import extract_telegram_channel_from_url
from news_service.services.twitter import extract_twitter_account_from_url
from news_service.tasks.celery_app import celery_app
from news_service.tasks.poll_adapters import (
    RedditAdapter,
    RssAdapter,
    SourceAdapter,
    TelegramAdapter,
    TwitterAdapter,
)

logger = logging.getLogger(__name__)
settings = get_settings()

DELIVER_EVENTS_BATCH_TASK = "news_service.tasks.deliver_events.deliver_event_notifications_batch"


@celery_app.task(name="news_service.tasks.poll_feeds.poll_all_feeds")
def poll_all_feeds() -> dict:
    return asyncio.run(_poll_all_feeds())


async def _poll_all_feeds() -> dict:
    async with get_task_session() as session:
        session.info["event_item_ids"] = []

        event_source_result = await session.execute(
            select(SubscriptionSource.source_id)
            .join(Subscription, Subscription.id == SubscriptionSource.subscription_id)
            .where(Subscription.is_active.is_(True), Subscription.delivery_mode == "event")
            .distinct()
        )
        session.info["event_source_ids"] = set(event_source_result.scalars().all())

        result = await session.execute(select(Source).where(Source.is_active.is_(True)))
        all_sources = list(result.scalars().all())

        total_new = 0
        for src in all_sources:
            try:
                count = await _poll_single_source(session, src)
            except Exception:
                await session.rollback()
                logger.exception(
                    "Unexpected failure while polling source %s",
                    src.url,
                    extra={"source_id": str(src.id)},
                )
                continue
            await session.commit()
            total_new += count

        event_item_ids = list(dict.fromkeys(session.info.pop("event_item_ids", [])))

    if event_item_ids:
        celery_app.send_task(
            DELIVER_EVENTS_BATCH_TASK,
            args=[[str(item_id) for item_id in event_item_ids]],
        )

    return {
        "feeds_polled": len(all_sources),
        "new_items": total_new,
        "event_notifications_queued": len(event_item_ids),
    }


async def _poll_single_source(session: AsyncSession, src: Source) -> int:
    """Route a source to its adapter and poll through the generic loop."""
    channel_handle = extract_telegram_channel_from_url(src.url)
    if channel_handle is not None:
        return await _poll_typed_source(session, src, TelegramAdapter(src, channel_handle))

    subreddit = extract_reddit_subreddit_from_url(src.url)
    if subreddit is not None:
        return await _poll_typed_source(session, src, RedditAdapter(src, subreddit))

    twitter_account = extract_twitter_account_from_url(src.url)
    if twitter_account is not None:
        return await _poll_typed_source(session, src, TwitterAdapter(src, twitter_account))

    return await _poll_typed_source(session, src, RssAdapter(src))


async def _poll_typed_source(
    session: AsyncSession,
    src: Source,
    adapter: SourceAdapter,
) -> int:
    """Generic polling loop for any source type via adapter."""
    try:
        posts = await adapter.fetch_posts()
    except Exception:
        logger.exception(
            "Failed to fetch from %s",
            adapter.log_label(),
            extra={"source_id": str(src.id)},
        )
        return 0

    now = datetime.now(UTC)
    fresh_posts = [p for p in posts if _is_fresh_news_item(p.published_at, now)]
    if not fresh_posts:
        src.last_polled_at = now
        logger.info(
            "Polled %s: 0 new items from %d posts",
            adapter.log_label(),
            len(posts),
            extra={"source_id": str(src.id)},
        )
        return 0

    for post in fresh_posts:
        injection_flags = scan_for_injection(post.body)
        if injection_flags:
            logger.warning(
                "Injection detected in post %s: %s",
                post.url,
                injection_flags[:3],
            )

    texts_to_embed = [cap_text_for_embedding(p.text_to_embed) for p in fresh_posts]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed posts for %s",
            adapter.log_label(),
            extra={"source_id": str(src.id)},
        )
        return 0

    new_count = 0
    for post, embedding in zip(fresh_posts, embeddings, strict=True):
        item = await upsert_news_item(
            session,
            source_id=src.id,
            headline=post.headline or adapter.source_name(),
            body=post.body,
            url=post.url,
            source=adapter.source_name(),
            published_at=post.published_at,
            fetched_at=now,
            embedding=embedding,
        )
        if item is not None:
            new_count += 1
            if isinstance(item, NewsItem) and item.source_id in session.info.get(
                "event_source_ids", set()
            ):
                session.info.setdefault("event_item_ids", []).append(item.id)

    src.last_polled_at = now
    logger.info(
        "Polled %s: %d new items from %d posts",
        adapter.log_label(),
        new_count,
        len(fresh_posts),
        extra={"source_id": str(src.id)},
    )
    return new_count


def _is_fresh_news_item(published_at: datetime | None, now: datetime) -> bool:
    if published_at is None:
        return True
    return published_at >= now - timedelta(days=settings.news_item_max_age_days)
