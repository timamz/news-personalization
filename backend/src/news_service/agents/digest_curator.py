"""Single-shot digest curation using Chat Completions API."""

import logging
import math
import uuid
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry
from news_service.core.openai_client import openai_client
from news_service.db.vector_store import find_similar_news
from news_service.models.news_item import NewsItem

logger = logging.getLogger(__name__)
settings = get_settings()
_client = openai_client


class DigestCurationResult(BaseModel):
    digest_text: str = Field(..., description="The formatted news digest")
    used_item_ids: list[str] = Field(..., description="IDs of news items included in the digest")


def _format_news_item(item: NewsItem) -> str:
    published = getattr(item, "published_at", None) or getattr(item, "fetched_at", None)
    pub_str = published.isoformat() if published else "unknown"
    body_preview = item.body or ""
    return (
        f"[ID: {item.id}]\n"
        f"Headline: {item.headline}\n"
        f"Published: {pub_str}\n"
        f"Body: {body_preview}\n"
        f"Link: {item.url}"
    )


def _is_russian_language(digest_language: str) -> bool:
    return digest_language.strip().lower().split("-", maxsplit=1)[0] == "ru"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _fetch_candidate_items(
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


def _build_items_text(items: list[NewsItem], max_chars: int) -> str:
    """Format items until hitting the context budget."""
    parts: list[str] = []
    total = 0
    for item in items:
        formatted = _format_news_item(item)
        if total + len(formatted) > max_chars:
            break
        parts.append(formatted)
        total += len(formatted)
    return "\n\n".join(parts)


async def run_digest_curator(
    *,
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_source_ids: set[uuid.UUID],
    published_after: datetime,
    format_instructions: str,
    digest_language: str,
) -> DigestCurationResult | None:
    """Curate a news digest from available items.

    Pre-fetches candidates from the DB, ranks by cosine similarity, and passes
    them to a single LLM call for selection and composition.

    Returns a DigestCurationResult with the formatted digest and used item IDs,
    or None if no usable items were found.
    """
    candidates = await _fetch_candidate_items(
        session,
        query_embedding,
        exclude_ids=exclude_ids,
        allowed_source_ids=allowed_source_ids,
        published_after=published_after,
    )
    if not candidates:
        return None

    source_label = "Источник" if _is_russian_language(digest_language) else "Source"
    items_text = _build_items_text(candidates, settings.llm_max_context_chars)

    system_prompt = (
        "You are a news digest curator. Create a well-structured, readable digest "
        "by selecting the best items from the candidates below.\n\n"
        "Quality criteria:\n"
        "- Prioritize the most substantive items.\n"
        "- Skip stale items, low-signal community chatter, personal requests, "
        "endorsement requests, generic questions, and self-promotional posts.\n"
        "- If multiple items cover the same story, include only the most informative one.\n\n"
        f"Format: {format_instructions}\n"
        f"Language: {digest_language}\n"
        f"For every item in the digest, end with the exact line "
        f"'{source_label}: <original link>' using exactly that label.\n"
        f"Never switch to a different language for the source label.\n"
        "Do not mention feed names, channel names, site names, or labels "
        f"other than the required '{source_label}:' line.\n"
        "Return only the digest. No introductions, closings, commentary, "
        "or offers to help.\n\n"
        "IMPORTANT: In used_item_ids, list the UUIDs of every news item you included "
        "in the digest."
    )

    output = await _parse_digest(system_prompt, items_text)
    if not output.used_item_ids:
        return None
    return output


@with_llm_retry()
async def _parse_digest(system_prompt: str, items_text: str) -> DigestCurationResult:
    response = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Candidate news items:\n\n{items_text}"},
        ],
        response_format=DigestCurationResult,
        temperature=0.3,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for digest curation")
    return result
