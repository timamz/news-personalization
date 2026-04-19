import logging
import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.llm import embed_text, embed_texts
from news_service.models.news_item import NewsItem
from news_service.models.source import Source

settings = get_settings()
logger = logging.getLogger(__name__)

__all__ = [
    "embed_text",
    "embed_texts",
    "find_similar_news",
    "find_similar_sources",
    "upsert_news_item",
]


async def find_similar_sources(
    session: AsyncSession,
    query_embedding: list[float],
    threshold: float | None = None,
    limit: int = 5,
) -> list[Source]:
    """Find sources whose topic embedding is within cosine similarity threshold.

    pgvector cosine_distance = 1 - cosine_similarity,
    so similarity >= threshold  <=>  distance <= (1 - threshold).
    """
    if threshold is None:
        threshold = settings.topic_similarity_threshold

    max_distance = 1.0 - threshold

    stmt = (
        select(Source)
        .where(
            Source.source_description_embedding.isnot(None),
            Source.is_active.is_(True),
            Source.source_description_embedding.cosine_distance(query_embedding) <= max_distance,
        )
        .order_by(Source.source_description_embedding.cosine_distance(query_embedding))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def find_similar_news(
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_source_ids: set[uuid.UUID] | None = None,
    published_after: datetime | None = None,
    limit: int = 20,
) -> list[NewsItem]:
    if allowed_source_ids is not None and not allowed_source_ids:
        return []

    exclude_list = list(exclude_ids) if exclude_ids else [uuid.uuid4()]
    recent_marker = func.coalesce(NewsItem.published_at, NewsItem.fetched_at)
    where_clauses = [
        NewsItem.embedding.isnot(None),
        NewsItem.id.notin_(exclude_list),
    ]
    if allowed_source_ids is not None:
        where_clauses.append(NewsItem.source_id.in_(list(allowed_source_ids)))
    if published_after is not None:
        where_clauses.append(recent_marker >= published_after)

    stmt = (
        select(NewsItem)
        .where(*where_clauses)
        .order_by(
            NewsItem.embedding.cosine_distance(query_embedding),
            recent_marker.desc(),
            NewsItem.fetched_at.desc(),
        )
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def upsert_news_item(
    session: AsyncSession,
    *,
    source_id: uuid.UUID,
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
        source_id=source_id,
        headline=_sanitize_text(headline),
        body=_sanitize_text(body),
        url=url,
        source=_sanitize_text(source),
        published_at=published_at,
        fetched_at=fetched_at,
        embedding=embedding,
    )
    session.add(item)
    await session.flush()
    return item


def _sanitize_text(value: str) -> str:
    """Strip NUL bytes (Postgres TEXT cannot hold them).

    Adapters occasionally pull binary blobs (e.g. an image masquerading as an
    article body) and pass them through here; without this the whole polling
    batch aborts on ``CharacterNotInRepertoireError`` for one bad item.
    """
    if not value:
        return value
    if "\x00" in value:
        return value.replace("\x00", "")
    return value
