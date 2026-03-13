import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.db.vector_store import embed_text, find_similar_news
from news_service.models.news_item import NewsItem
from news_service.models.rss_feed import RssFeed
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.reddit import (
    extract_reddit_subreddit_from_url,
    extract_reddit_subreddits,
)

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client


async def generate_digest(session: AsyncSession, subscription: Subscription) -> str | None:
    sent_result = await session.execute(
        select(SentItem.news_item_id, SentItem.sent_at).where(
            SentItem.subscription_id == subscription.id
        )
    )
    sent_rows = list(sent_result.all())
    sent_ids: set[uuid.UUID] = {news_item_id for news_item_id, _ in sent_rows}
    last_sent_at = max((sent_at for _, sent_at in sent_rows), default=None)

    source_feed_ids = await _source_feed_ids_for_digest(session, subscription)
    if not source_feed_ids:
        logger.warning(
            "No fixed sources configured for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    query_embedding = getattr(subscription, "raw_prompt_embedding", None)
    if query_embedding is None:
        query_text = subscription.raw_prompt.strip() or subscription.prompt_summary.strip()
        query_embedding = await embed_text(query_text)
        subscription.raw_prompt_embedding = query_embedding

    news_items = await find_similar_news(
        session,
        query_embedding,
        exclude_ids=sent_ids,
        allowed_feed_ids=source_feed_ids,
        published_after=_published_after_for_digest(last_sent_at),
        limit=15,
    )

    if not news_items:
        logger.info(
            "No unseen news items for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    digest_text = await _compose_digest(
        news_items,
        subscription.format_instructions,
        subscription.digest_language,
    )

    await _mark_as_sent(session, subscription.id, [item.id for item in news_items])

    logger.info(
        "Generated digest with %d items for subscription %s",
        len(news_items),
        subscription.id,
        extra={"subscription_id": str(subscription.id)},
    )
    return digest_text


async def _compose_digest(
    items: list[NewsItem],
    format_instructions: str,
    digest_language: str,
) -> str:
    source_label = "Источник" if _is_russian_language(digest_language) else "Source"
    news_block = "\n\n".join(
        (
            f"**{item.headline}**\n"
            f"Published at: {_published_marker(item)}\n"
            f"{item.body}\n"
            f"Link: {item.url}"
        )
        for item in items
    )

    completion = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a news digest writer. Format the following news items "
                    f"according to these instructions: {format_instructions}\n\n"
                    f"Write the digest in language '{digest_language}'. "
                    f"Make it well-structured, readable, and engaging. "
                    f"Prioritize the newest and most substantive items. "
                    f"Ignore stale items and low-signal community chatter such as "
                    f"personal requests, endorsement requests, generic questions, "
                    f"and self-promotional posts unless they are clearly newsworthy. "
                    f"For every item, end with the exact line "
                    f"'{source_label}: <original link>' using the digest language label. "
                    f"Use exactly '{source_label}:' and never switch to a different language. "
                    f"Keep the link exactly as provided and include no extra text on that line. "
                    f"Do not mention feed names, channel names, site names, or labels other than "
                    f"the required '{source_label}:' line. "
                    f"Return only the digest itself. Do not add assistant-style "
                    f"introductions, closings, commentary, or offers to help."
                ),
            },
            {"role": "user", "content": news_block},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def _is_russian_language(digest_language: str) -> bool:
    return digest_language.strip().lower().split("-", maxsplit=1)[0] == "ru"


def _published_after_for_digest(last_sent_at: datetime | None) -> datetime:
    if last_sent_at is not None:
        return last_sent_at
    return datetime.now(UTC) - timedelta(days=settings.news_item_max_age_days)


def _published_marker(item: NewsItem) -> str:
    timestamp = getattr(item, "published_at", None) or getattr(item, "fetched_at", None)
    if timestamp is None:
        return "unknown"
    return timestamp.astimezone(UTC).isoformat()


async def _source_feed_ids_for_digest(
    session: AsyncSession,
    subscription: Subscription,
) -> set[uuid.UUID]:
    source_result = await session.execute(
        select(RssFeed.id, RssFeed.url)
        .join(SubscriptionSource, SubscriptionSource.feed_id == RssFeed.id)
        .where(SubscriptionSource.subscription_id == subscription.id)
    )
    source_rows = list(source_result.all())
    if not source_rows:
        return set()
    if not _should_exclude_reddit_sources(subscription):
        return {feed_id for feed_id, _ in source_rows}

    filtered_rows = [
        (feed_id, url)
        for feed_id, url in source_rows
        if extract_reddit_subreddit_from_url(url) is None
    ]
    if filtered_rows:
        return {feed_id for feed_id, _ in filtered_rows}
    return {feed_id for feed_id, _ in source_rows}


def _should_exclude_reddit_sources(subscription: Subscription) -> bool:
    return _is_research_focused_subscription(subscription) and not extract_reddit_subreddits(
        subscription.raw_prompt
    )


def _is_research_focused_subscription(subscription: Subscription) -> bool:
    text = " ".join(
        part
        for part in [subscription.raw_prompt, getattr(subscription, "prompt_summary", "")]
        if part
    ).lower()
    return any(
        keyword in text
        for keyword in (
            "research paper",
            "research papers",
            "scientific article",
            "scientific articles",
            "paper",
            "papers",
            "study",
            "studies",
            "arxiv",
            "научн",
            "исследован",
            "стат",
        )
    )


async def _mark_as_sent(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    news_item_ids: list[uuid.UUID],
) -> None:
    for item_id in news_item_ids:
        session.add(SentItem(subscription_id=subscription_id, news_item_id=item_id))
    await session.flush()
