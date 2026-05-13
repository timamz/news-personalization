"""Candidate fetching for digest generation — pure DB queries, no LLM."""

import uuid
from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.guardrails import wrap_untrusted_content
from news_service.db.vector_store import find_similar_news
from news_service.models.news_item import NewsItem
from news_service.services.relevance import cosine_similarity


async def fetch_candidate_items(
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_source_ids: set[uuid.UUID],
    published_after: datetime,
    fetched_after: datetime | None = None,
) -> list[NewsItem]:
    """Fetch candidates by relevance and recency, merge, sort by cosine similarity.

    ``fetched_after`` is OR'd with the publish-cutoff filter so items that
    were ingested after the previous digest still qualify even when their
    ``published_at`` predates the cutoff -- the typical late-fetch case
    where an RSS source surfaces an older article only after a delayed poll.
    """
    relevance_items = await find_similar_news(
        session,
        query_embedding,
        exclude_ids=exclude_ids,
        allowed_source_ids=allowed_source_ids,
        published_after=published_after,
        fetched_after=fetched_after,
        limit=50,
    )

    exclude_list = list(exclude_ids) if exclude_ids else [uuid.uuid4()]
    recent_marker = func.coalesce(NewsItem.published_at, NewsItem.fetched_at)
    where_clauses = [
        NewsItem.embedding.isnot(None),
        NewsItem.id.notin_(exclude_list),
        NewsItem.source_id.in_(list(allowed_source_ids)),
    ]
    if fetched_after is not None:
        where_clauses.append(
            or_(recent_marker >= published_after, NewsItem.fetched_at >= fetched_after)
        )
    else:
        where_clauses.append(recent_marker >= published_after)
    stmt = (
        select(NewsItem)
        .where(*where_clauses)
        .order_by(recent_marker.desc(), NewsItem.fetched_at.desc())
        .limit(30)
    )
    result = await session.execute(stmt)
    recency_items = list(result.scalars().all())

    seen_ids: set[uuid.UUID] = set()
    merged: list[NewsItem] = []
    for item in [*relevance_items, *recency_items]:
        if item.id not in seen_ids:
            seen_ids.add(item.id)
            merged.append(item)

    merged.sort(
        key=lambda item: (
            cosine_similarity(list(item.embedding), query_embedding)
            if item.embedding is not None
            else 0.0
        ),
        reverse=True,
    )

    return merged


def format_news_item(item: NewsItem) -> str:
    published = getattr(item, "published_at", None) or getattr(item, "fetched_at", None)
    pub_str = published.isoformat() if published else "unknown"
    body_preview = item.body or ""
    return (
        f"[ID: {item.id}]\n"
        f"Headline: {wrap_untrusted_content(item.headline)}\n"
        f"Published: {pub_str}\n"
        f"Body: {wrap_untrusted_content(body_preview)}\n"
        f"Link: {item.url}"
    )


def build_items_text(items: list[NewsItem], max_chars: int) -> str:
    """Format items until hitting the context budget."""
    parts: list[str] = []
    total = 0
    for item in items:
        formatted = format_news_item(item)
        if total + len(formatted) > max_chars:
            break
        parts.append(formatted)
        total += len(formatted)
    return "\n\n".join(parts)
