import logging
import uuid
from datetime import datetime

from openai import OpenAIError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.models.news_item import NewsItem
from news_service.models.rss_feed import RssFeed

settings = get_settings()
_client = openai_client
logger = logging.getLogger(__name__)

EMBEDDING_MAX_CHARS = 4000
EMBEDDING_BATCH_SIZE = 6


def _normalize_embedding_text(content: str) -> str:
    normalized = " ".join(content.split())
    return normalized[:EMBEDDING_MAX_CHARS]


async def embed_text(content: str) -> list[float]:
    response = await _client.embeddings.create(
        input=_normalize_embedding_text(content),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


async def embed_texts(contents: list[str]) -> list[list[float]]:
    if not contents:
        return []

    normalized_contents = [_normalize_embedding_text(content) for content in contents]
    embeddings: list[list[float]] = []

    for i in range(0, len(normalized_contents), EMBEDDING_BATCH_SIZE):
        batch = normalized_contents[i : i + EMBEDDING_BATCH_SIZE]
        try:
            response = await _client.embeddings.create(
                input=batch,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
            embeddings.extend(item.embedding for item in response.data)
        except OpenAIError:
            logger.exception(
                "Batch embedding failed; retrying per-item for batch size=%d",
                len(batch),
            )
            for content in batch:
                embeddings.append(await embed_text(content))

    return embeddings


async def find_similar_feeds(
    session: AsyncSession,
    query_embedding: list[float],
    threshold: float | None = None,
    limit: int = 5,
) -> list[RssFeed]:
    """Find feeds whose topic embedding is within cosine similarity threshold.

    pgvector cosine_distance = 1 - cosine_similarity,
    so similarity >= threshold  ⟺  distance <= (1 - threshold).
    """
    if threshold is None:
        threshold = settings.topic_similarity_threshold

    max_distance = 1.0 - threshold

    stmt = (
        select(RssFeed)
        .where(
            RssFeed.topic_embedding.isnot(None),
            RssFeed.is_active.is_(True),
            RssFeed.topic_embedding.cosine_distance(query_embedding) <= max_distance,
        )
        .order_by(RssFeed.topic_embedding.cosine_distance(query_embedding))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def find_similar_news(
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_feed_ids: set[uuid.UUID] | None = None,
    limit: int = 20,
) -> list[NewsItem]:
    if allowed_feed_ids is not None and not allowed_feed_ids:
        return []

    exclude_list = list(exclude_ids) if exclude_ids else [uuid.uuid4()]
    where_clauses = [
        NewsItem.embedding.isnot(None),
        NewsItem.id.notin_(exclude_list),
    ]
    if allowed_feed_ids is not None:
        where_clauses.append(NewsItem.feed_id.in_(list(allowed_feed_ids)))

    stmt = (
        select(NewsItem)
        .where(*where_clauses)
        .order_by(NewsItem.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def upsert_news_item(
    session: AsyncSession,
    *,
    feed_id: uuid.UUID,
    headline: str,
    body: str,
    url: str,
    source: str,
    published_at: datetime | None,
    fetched_at: datetime,
    embedding: list[float] | None = None,
) -> NewsItem | None:
    existing = await session.execute(select(NewsItem).where(NewsItem.url == url))
    if existing.scalar_one_or_none() is not None:
        return None

    item = NewsItem(
        feed_id=feed_id,
        headline=headline,
        body=body,
        url=url,
        source=source,
        published_at=published_at,
        fetched_at=fetched_at,
        embedding=embedding,
    )
    session.add(item)
    return item
