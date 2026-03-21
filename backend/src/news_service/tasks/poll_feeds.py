import asyncio
import logging
from datetime import UTC, datetime, timedelta

import feedparser
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.db.session import get_task_session
from news_service.db.vector_store import embed_texts, upsert_news_item
from news_service.models.news_item import NewsItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.reddit import extract_reddit_subreddit_from_url, fetch_reddit_posts
from news_service.services.telegram import extract_telegram_channel_from_url, fetch_telegram_posts
from news_service.services.twitter import extract_twitter_account_from_url, fetch_twitter_posts
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

    for item_id in event_item_ids:
        celery_app.send_task(DELIVER_EVENTS_TASK, args=[str(item_id)])

    return {
        "feeds_polled": len(all_sources),
        "new_items": total_new,
        "event_notifications_queued": len(event_item_ids),
    }


async def _poll_single_source(session: AsyncSession, src: Source) -> int:
    channel_handle = extract_telegram_channel_from_url(src.url)
    if channel_handle is not None:
        return await _poll_single_telegram_channel(session, src, channel_handle)

    subreddit = extract_reddit_subreddit_from_url(src.url)
    if subreddit is not None:
        return await _poll_single_reddit_subreddit(session, src, subreddit)

    twitter_account = extract_twitter_account_from_url(src.url)
    if twitter_account is not None:
        return await _poll_single_twitter_account(session, src, twitter_account)

    try:
        content = await _fetch_rss_feed_content(src.url)
        parsed = feedparser.parse(content)
    except Exception:
        logger.exception("Failed to parse feed %s", src.url, extra={"source_id": str(src.id)})
        return 0

    entries = parsed.entries
    if not entries:
        return 0

    now = datetime.now(UTC)
    recent_entries: list[tuple[object, str, str, datetime | None]] = []
    for entry in entries:
        headline = entry.get("title", "")
        body = entry.get("summary", entry.get("description", ""))
        published_at = _published_at_from_rss_entry(entry)
        if not _is_fresh_news_item(published_at, now):
            continue
        recent_entries.append((entry, headline, body, published_at))
    if not recent_entries:
        src.last_polled_at = now
        logger.info(
            "Polled feed %s: 0 new items from %d recent entries",
            src.url,
            len(entries),
            extra={"source_id": str(src.id)},
        )
        return 0

    texts_to_embed = [f"{headline} {body}" for _, headline, body, _ in recent_entries]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed entries for feed %s",
            src.url,
            extra={"source_id": str(src.id)},
        )
        return 0

    new_count = 0
    for (entry, headline, body, published_at), embedding in zip(
        recent_entries,
        embeddings,
        strict=True,
    ):
        url = entry.get("link", "")
        if not url:
            continue

        item = await upsert_news_item(
            session,
            source_id=src.id,
            headline=headline or "Untitled",
            body=body,
            url=url,
            source=src.title or src.url,
            published_at=published_at,
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
        "Polled feed %s: %d new items from %d recent entries",
        src.url,
        new_count,
        len(recent_entries),
        extra={"source_id": str(src.id)},
    )
    return new_count


async def _poll_single_telegram_channel(
    session: AsyncSession,
    src: Source,
    channel_handle: str,
) -> int:
    try:
        posts = await fetch_telegram_posts(channel_handle)
    except Exception:
        logger.exception(
            "Failed to parse Telegram channel @%s",
            channel_handle,
            extra={"source_id": str(src.id)},
        )
        return 0

    now = datetime.now(UTC)
    fresh_posts = [post for post in posts if _is_fresh_news_item(post.published_at, now)]
    if not fresh_posts:
        src.last_polled_at = now
        return 0

    texts_to_embed = [post.body for post in fresh_posts]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed Telegram posts for @%s",
            channel_handle,
            extra={"source_id": str(src.id)},
        )
        return 0

    source_name = src.title or f"Telegram @{channel_handle}"
    new_count = 0
    for post, embedding in zip(fresh_posts, embeddings, strict=True):
        headline = post.body.splitlines()[0][:200]
        item = await upsert_news_item(
            session,
            source_id=src.id,
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
            if isinstance(item, NewsItem) and item.source_id in session.info.get(
                "event_source_ids", set()
            ):
                session.info.setdefault("event_item_ids", []).append(item.id)

    src.last_polled_at = now
    logger.info(
        "Polled Telegram channel @%s: %d new items from %d posts",
        channel_handle,
        new_count,
        len(fresh_posts),
        extra={"source_id": str(src.id)},
    )
    return new_count


async def _poll_single_reddit_subreddit(
    session: AsyncSession,
    src: Source,
    subreddit: str,
) -> int:
    try:
        posts = await fetch_reddit_posts(subreddit)
    except Exception:
        logger.exception(
            "Failed to parse Reddit subreddit r/%s",
            subreddit,
            extra={"source_id": str(src.id)},
        )
        return 0

    now = datetime.now(UTC)
    fresh_posts = [post for post in posts if _is_fresh_news_item(post.published_at, now)]
    if not fresh_posts:
        src.last_polled_at = now
        return 0

    texts_to_embed = [
        "\n\n".join(part for part in [post.title, post.body] if part).strip()
        for post in fresh_posts
    ]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed Reddit posts for r/%s",
            subreddit,
            extra={"source_id": str(src.id)},
        )
        return 0

    source_name = src.title or f"Reddit r/{subreddit}"
    new_count = 0
    for post, embedding in zip(fresh_posts, embeddings, strict=True):
        item = await upsert_news_item(
            session,
            source_id=src.id,
            headline=post.title[:200] or f"Reddit post from r/{subreddit}",
            body=post.body,
            url=post.url,
            source=source_name,
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
        "Polled Reddit subreddit r/%s: %d new items from %d posts",
        subreddit,
        new_count,
        len(fresh_posts),
        extra={"source_id": str(src.id)},
    )
    return new_count


async def _poll_single_twitter_account(
    session: AsyncSession,
    src: Source,
    account: str,
) -> int:
    try:
        posts = await fetch_twitter_posts(account)
    except Exception:
        logger.exception(
            "Failed to parse Twitter/X account @%s",
            account,
            extra={"source_id": str(src.id)},
        )
        return 0

    now = datetime.now(UTC)
    fresh_posts = [post for post in posts if _is_fresh_news_item(post.published_at, now)]
    if not fresh_posts:
        src.last_polled_at = now
        return 0

    texts_to_embed = [post.body for post in fresh_posts]
    try:
        embeddings = await embed_texts(texts_to_embed)
    except Exception:
        logger.exception(
            "Failed to embed Twitter/X posts for @%s",
            account,
            extra={"source_id": str(src.id)},
        )
        return 0

    source_name = src.title or f"X @{account}"
    new_count = 0
    for post, embedding in zip(fresh_posts, embeddings, strict=True):
        headline = post.body.splitlines()[0][:200]
        item = await upsert_news_item(
            session,
            source_id=src.id,
            headline=headline or f"Post from @{account}",
            body=post.body,
            url=post.url,
            source=source_name,
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
        "Polled Twitter/X account @%s: %d new items from %d posts",
        account,
        new_count,
        len(fresh_posts),
        extra={"source_id": str(src.id)},
    )
    return new_count


def _published_at_from_rss_entry(entry: object) -> datetime | None:
    published_str = entry.get("published", None)
    if not published_str:
        return None
    try:
        import email.utils

        parsed_date = email.utils.parsedate_to_datetime(published_str)
        return parsed_date.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def _is_fresh_news_item(published_at: datetime | None, now: datetime) -> bool:
    if published_at is None:
        return True
    return published_at >= now - timedelta(days=settings.news_item_max_age_days)


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
