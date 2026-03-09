import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.db.vector_store import embed_text, find_similar_news
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client


async def generate_digest(session: AsyncSession, subscription: Subscription) -> str | None:
    sent_result = await session.execute(
        select(SentItem.news_item_id).where(SentItem.subscription_id == subscription.id)
    )
    sent_ids: set[uuid.UUID] = set(sent_result.scalars().all())

    source_result = await session.execute(
        select(SubscriptionSource.feed_id).where(
            SubscriptionSource.subscription_id == subscription.id
        )
    )
    source_feed_ids: set[uuid.UUID] = set(source_result.scalars().all())
    if not source_feed_ids:
        logger.warning(
            "No fixed sources configured for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    query_embedding = getattr(subscription, "raw_prompt_embedding", None)
    if query_embedding is None:
        query_text = subscription.raw_prompt.strip() or " ".join(subscription.topics)
        query_embedding = await embed_text(query_text)
        subscription.raw_prompt_embedding = query_embedding

    news_items = await find_similar_news(
        session,
        query_embedding,
        exclude_ids=sent_ids,
        allowed_feed_ids=source_feed_ids,
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
    news_block = "\n\n".join(
        f"**{item.headline}**\n{item.body}\nLink: {item.url}" for item in items
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
                    f"When referencing a source, keep only the hyperlink. "
                    f"Do not mention source labels, feed names, channel names, site names, "
                    f"or words like 'source', 'channel', 'website', 'post', "
                    f"'Источник', 'канал', or 'сайт'. "
                    f"Keep the link exactly as provided. "
                    f"Return only the digest itself. Do not add assistant-style "
                    f"introductions, closings, commentary, or offers to help."
                ),
            },
            {"role": "user", "content": news_block},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


async def _mark_as_sent(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    news_item_ids: list[uuid.UUID],
) -> None:
    for item_id in news_item_ids:
        session.add(SentItem(subscription_id=subscription_id, news_item_id=item_id))
    await session.flush()
