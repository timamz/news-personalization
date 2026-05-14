"""Source polling task — fetches new items from all active sources.

Uses the adapter pattern to handle different source types (RSS, Telegram,
Reddit) through a single generic polling loop.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.guardrails import (
    cap_text_for_embedding,
    classify_injection,
    scan_for_injection,
)
from news_service.core.provider_errors import ProviderLimitError
from news_service.db.session import get_task_session
from news_service.db.vector_store import embed_texts, upsert_news_item
from news_service.models.news_item import NewsItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.reddit import extract_reddit_subreddit_from_url
from news_service.services.telegram import extract_telegram_channel_from_url
from news_service.tasks.celery_app import celery_app
from news_service.tasks.poll_adapters import (
    RedditAdapter,
    RssAdapter,
    SourceAdapter,
    TelegramAdapter,
)
from news_service.tasks.retry_policy import retry_on_provider_limit

logger = logging.getLogger(__name__)
settings = get_settings()

DELIVER_EVENTS_BATCH_TASK = "news_service.tasks.deliver_events.deliver_event_notifications_batch"


@celery_app.task(bind=True, name="news_service.tasks.poll_feeds.poll_all_feeds")
def poll_all_feeds(self) -> dict:
    try:
        return asyncio.run(_poll_all_feeds())
    except ProviderLimitError as exc:
        raise retry_on_provider_limit(self, exc) from exc


async def _poll_all_feeds() -> dict:
    """Poll every active source under bounded concurrency, fair ordering, per-source timeout.

    Sources are picked in ascending ``last_polled_at`` order with NULLs first
    so unpolled sources always make progress and a hot subset cannot starve
    the long tail. Concurrency is capped at ``poll_max_concurrency`` so a
    burst of 200+ active sources does not saturate the provider, network
    egress, or the embedder. Each source has its own session and a hard
    ``poll_per_source_timeout_seconds`` deadline; a single slow adapter
    cannot eat the entire 30-minute polling cycle.
    """
    async with get_task_session() as session:
        event_source_result = await session.execute(
            select(SubscriptionSource.source_id)
            .join(Subscription, Subscription.id == SubscriptionSource.subscription_id)
            .where(Subscription.is_active.is_(True), Subscription.delivery_mode == "event")
            .distinct()
        )
        event_source_ids = set(event_source_result.scalars().all())

        result = await session.execute(
            select(Source.id)
            .where(Source.is_active.is_(True))
            .order_by(Source.last_polled_at.asc().nulls_first(), Source.id)
        )
        source_ids = list(result.scalars().all())

    semaphore = asyncio.Semaphore(settings.poll_max_concurrency)

    async def _run_one(source_id) -> tuple[int, list]:
        async with semaphore, get_task_session() as task_session:
            task_session.info["event_source_ids"] = event_source_ids
            task_session.info["event_item_ids"] = []
            src = await task_session.get(Source, source_id)
            if src is None or not src.is_active:
                return 0, []
            try:
                count = await asyncio.wait_for(
                    _poll_single_source(task_session, src),
                    timeout=settings.poll_per_source_timeout_seconds,
                )
            except TimeoutError:
                await task_session.rollback()
                logger.warning(
                    "Polling timed out after %.0fs for %s",
                    settings.poll_per_source_timeout_seconds,
                    src.url,
                    extra={"source_id": str(source_id)},
                )
                return 0, []
            except ProviderLimitError:
                await task_session.rollback()
                raise
            except Exception:
                await task_session.rollback()
                logger.exception(
                    "Unexpected failure while polling source %s",
                    src.url,
                    extra={"source_id": str(source_id)},
                )
                return 0, []
            await task_session.commit()
            return count, list(task_session.info.get("event_item_ids", []))

    results = await asyncio.gather(
        *(_run_one(sid) for sid in source_ids),
        return_exceptions=True,
    )

    total_new = 0
    event_item_ids: list = []
    provider_limit: ProviderLimitError | None = None
    for r in results:
        if isinstance(r, ProviderLimitError):
            provider_limit = provider_limit or r
            continue
        if isinstance(r, BaseException):
            logger.exception("Unexpected polling task error", exc_info=r)
            continue
        count, items = r
        total_new += count
        event_item_ids.extend(items)

    if provider_limit is not None:
        raise provider_limit

    deduped_event_ids = list(dict.fromkeys(event_item_ids))
    if deduped_event_ids:
        celery_app.send_task(
            DELIVER_EVENTS_BATCH_TASK,
            args=[[str(item_id) for item_id in deduped_event_ids]],
        )

    return {
        "feeds_polled": len(source_ids),
        "new_items": total_new,
        "event_notifications_queued": len(deduped_event_ids),
    }


async def _poll_single_source(session: AsyncSession, src: Source) -> int:
    """Route a source to its adapter and poll through the generic loop."""
    channel_handle = extract_telegram_channel_from_url(src.url)
    if channel_handle is not None:
        return await _poll_typed_source(session, src, TelegramAdapter(src, channel_handle))

    subreddit = extract_reddit_subreddit_from_url(src.url)
    if subreddit is not None:
        return await _poll_typed_source(session, src, RedditAdapter(src, subreddit))

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
        ml_score = classify_injection(post.body)
        if ml_score is not None and ml_score >= settings.injection_classifier_threshold:
            injection_flags.append(f"classifier:{ml_score:.2f}")
        if injection_flags:
            logger.warning(
                "Injection detected in post %s: %s",
                post.url,
                injection_flags[:3],
            )

    texts_to_embed = [cap_text_for_embedding(p.text_to_embed) for p in fresh_posts]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except ProviderLimitError:
        raise
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
