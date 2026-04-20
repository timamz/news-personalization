"""Shared ``fetch_source_items`` ADK tool builder.

Both the Digest Pipeline Reflector and the Event Verifier need the same
read-only view over a subscription's recent source items, scoped to the
source_ids the agent is allowed to touch. The Reflector additionally
annotates each item with its cosine similarity to the subscription's
topic embedding; the Verifier does not need this.

``build_fetch_source_items_tool`` produces a closure suitable for direct
inclusion in an ADK ``Agent.tools`` list. The closure's ``__name__`` is
configurable because the two call sites historically exposed the tool
under different names to the LLM (``fetch_source_items`` for the
Reflector, ``fetch_source_items_tool`` for the Verifier) and ADK surfaces
the function's ``__name__`` as the tool name to the model.

Usage::

    tool = build_fetch_source_items_tool(
        db_session=session,
        allowed_source_ids={sid_1, sid_2},
        topic_embedding=embedding,  # optional, enables cos= annotation
    )
    agent = Agent(..., tools=[tool, ...])
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.models.news_item import NewsItem
from news_service.services.relevance import cosine_similarity

settings = get_settings()


def build_fetch_source_items_tool(
    *,
    db_session: AsyncSession,
    allowed_source_ids: set[uuid.UUID],
    topic_embedding: list[float] | None = None,
    name: str = "fetch_source_items",
) -> Callable[..., Awaitable[str]]:
    """Build an ADK tool closure that reads recent items from an allowed source.

    Args:
        db_session: Async session the closure will query.
        allowed_source_ids: Source IDs the caller is permitted to inspect;
            any request for a source_id outside this set is refused.
        topic_embedding: When provided, each returned item is annotated
            with its cosine similarity to this vector (rendered as
            ``cos=0.42``). When ``None``, the annotation is omitted.
        name: Value assigned to the returned closure's ``__name__``; ADK
            exposes this as the tool name to the LLM.
    """

    async def _tool(
        source_id: str,
        since_days_ago: int = 14,
        limit: int = 10,
    ) -> str:
        try:
            sid = uuid.UUID(str(source_id).strip())
        except (ValueError, AttributeError):
            return f"Invalid source_id: {source_id!r}."
        if sid not in allowed_source_ids:
            return f"Source {sid} is not linked to this subscription."

        max_limit = settings.reflector_fetch_source_items_max_limit
        effective_limit = max(1, min(int(limit or 10), max_limit))
        effective_days = max(1, int(since_days_ago or 14))
        cutoff = datetime.now(UTC) - timedelta(days=effective_days)

        stmt = (
            select(NewsItem)
            .where(
                NewsItem.source_id == sid,
                NewsItem.published_at.is_not(None),
                NewsItem.published_at >= cutoff,
            )
            .order_by(NewsItem.published_at.desc())
            .limit(effective_limit)
        )
        result = await db_session.execute(stmt)
        items = list(result.scalars().all())
        if not items:
            return f"No items from source {sid} in the last {effective_days} days."

        lines: list[str] = [f"Items from source {sid} (last {effective_days} days):"]
        for item in items:
            published = (
                item.published_at.isoformat() if item.published_at is not None else "unknown"
            )
            body_snippet = (item.body or "")[:300].replace("\n", " ").strip()
            if topic_embedding is not None:
                if item.embedding is not None:
                    try:
                        sim = cosine_similarity(list(item.embedding), topic_embedding)
                        sim_str = f"{sim:.2f}"
                    except Exception:
                        sim_str = "n/a"
                else:
                    sim_str = "n/a"
                lines.append(
                    f"- [{published}] cos={sim_str} | {item.headline}\n    {body_snippet}"
                )
            else:
                lines.append(f"- [{published}] {item.headline}\n    {body_snippet}")
        return "\n".join(lines)

    _tool.__name__ = name
    _tool.__qualname__ = name
    _tool.__doc__ = (
        "Fetch recent items from a linked source for inspection.\n\n"
        "Args:\n"
        "    source_id: UUID of the source to inspect.\n"
        "    since_days_ago: How far back to look, in days. Default 14.\n"
        "    limit: Max number of items to return. Capped at the server limit.\n\n"
        "Returns:\n"
        "    Formatted list of items with headline, body snippet, and\n"
        "    published_at; cosine similarity to the subscription topic is\n"
        "    included when a topic embedding is configured."
    )
    return _tool
