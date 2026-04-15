"""Candidate fetching for digest generation — pure DB queries, no LLM."""

import math
import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.db.vector_store import find_similar_news
from news_service.models.news_item import NewsItem
from news_service.orchestration.guardrails import wrap_untrusted_content


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def fetch_candidate_items(
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_source_ids: set[uuid.UUID],
    published_after: datetime,
) -> list[NewsItem]:
    """Fetch candidates by relevance and recency, merge, sort by cosine similarity."""
    relevance_items = await find_similar_news(
        session,
        query_embedding,
        exclude_ids=exclude_ids,
        allowed_source_ids=allowed_source_ids,
        published_after=published_after,
        limit=50,
    )

    exclude_list = list(exclude_ids) if exclude_ids else [uuid.uuid4()]
    recent_marker = func.coalesce(NewsItem.published_at, NewsItem.fetched_at)
    stmt = (
        select(NewsItem)
        .where(
            NewsItem.embedding.isnot(None),
            NewsItem.id.notin_(exclude_list),
            NewsItem.source_id.in_(list(allowed_source_ids)),
            recent_marker >= published_after,
        )
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
            _cosine_similarity(list(item.embedding), query_embedding)
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
