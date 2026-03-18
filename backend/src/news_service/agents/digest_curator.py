"""Agentic digest curation using OpenAI Agents SDK with tool-calling."""

import logging
import uuid
from datetime import datetime

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.openai_client import agents_model
from news_service.db.vector_store import find_similar_news
from news_service.models.news_item import NewsItem

logger = logging.getLogger(__name__)
settings = get_settings()


class DigestCurationResult(BaseModel):
    digest_text: str = Field(..., description="The formatted news digest")
    used_item_ids: list[str] = Field(..., description="IDs of news items included in the digest")


def _format_news_item(item: NewsItem) -> str:
    published = getattr(item, "published_at", None) or getattr(item, "fetched_at", None)
    pub_str = published.isoformat() if published else "unknown"
    body_preview = (item.body or "")[:500]
    return (
        f"[ID: {item.id}]\n"
        f"Headline: {item.headline}\n"
        f"Published: {pub_str}\n"
        f"Body: {body_preview}\n"
        f"Link: {item.url}"
    )


def _create_digest_curator_agent(
    *,
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_feed_ids: set[uuid.UUID],
    published_after: datetime,
    format_instructions: str,
    digest_language: str,
) -> Agent[None]:
    source_label = "Источник" if _is_russian_language(digest_language) else "Source"

    @function_tool
    async def search_news_by_relevance(limit: int = 15) -> str:
        """Search for news items most relevant to the subscription topic.

        Returns items ranked by semantic similarity to the user's subscription.

        Args:
            limit: Maximum number of items to return (default 15).
        """
        items = await find_similar_news(
            session,
            query_embedding,
            exclude_ids=exclude_ids,
            allowed_feed_ids=allowed_feed_ids,
            published_after=published_after,
            limit=limit,
        )
        if not items:
            return "No relevant news items found."
        return "\n\n".join(_format_news_item(item) for item in items)

    @function_tool
    async def search_news_by_recency(limit: int = 10) -> str:
        """Search for the most recent news items from subscription sources.

        Returns items sorted by publication date, newest first.
        Useful for finding recent items that semantic search might miss.

        Args:
            limit: Maximum number of items to return (default 10).
        """
        exclude_list = list(exclude_ids) if exclude_ids else [uuid.uuid4()]
        recent_marker = func.coalesce(NewsItem.published_at, NewsItem.fetched_at)
        where_clauses = [
            NewsItem.embedding.isnot(None),
            NewsItem.id.notin_(exclude_list),
            NewsItem.feed_id.in_(list(allowed_feed_ids)),
            recent_marker >= published_after,
        ]
        stmt = (
            select(NewsItem)
            .where(*where_clauses)
            .order_by(recent_marker.desc(), NewsItem.fetched_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        items = list(result.scalars().all())
        if not items:
            return "No recent news items found."
        return "\n\n".join(_format_news_item(item) for item in items)

    @function_tool
    async def get_article_details(item_id: str) -> str:
        """Get the full details of a specific news item by its ID.

        Args:
            item_id: The UUID of the news item to retrieve.
        """
        try:
            parsed_id = uuid.UUID(item_id)
        except ValueError:
            return f"Invalid item ID: {item_id}"
        result = await session.execute(select(NewsItem).where(NewsItem.id == parsed_id))
        item = result.scalar_one_or_none()
        if item is None:
            return f"Item not found: {item_id}"
        published = getattr(item, "published_at", None) or getattr(item, "fetched_at", None)
        pub_str = published.isoformat() if published else "unknown"
        return (
            f"[ID: {item.id}]\n"
            f"Headline: {item.headline}\n"
            f"Published: {pub_str}\n"
            f"Full body:\n{item.body}\n"
            f"Link: {item.url}"
        )

    instructions = (
        f"You are a news digest curator. Create a well-structured, readable digest "
        f"from available news items.\n\n"
        f"Your workflow:\n"
        f"1. Search for relevant news items using the available tools.\n"
        f"2. Review the results and assess quality, diversity, and relevance.\n"
        f"3. If results are sparse or repetitive, try different search strategies "
        f"(by relevance vs recency).\n"
        f"4. Compose a digest following the format instructions below.\n\n"
        f"Quality criteria:\n"
        f"- Prioritize the newest and most substantive items.\n"
        f"- Skip stale items, low-signal community chatter, personal requests, "
        f"endorsement requests, generic questions, and self-promotional posts.\n"
        f"- If multiple items cover the same story, include only the most informative one.\n\n"
        f"Format: {format_instructions}\n"
        f"Language: {digest_language}\n"
        f"For every item in the digest, end with the exact line "
        f"'{source_label}: <original link>' using exactly that label.\n"
        f"Never switch to a different language for the source label.\n"
        f"Do not mention feed names, channel names, site names, or labels "
        f"other than the required '{source_label}:' line.\n"
        f"Return only the digest. No introductions, closings, commentary, "
        f"or offers to help.\n\n"
        f"IMPORTANT: In used_item_ids, list the UUIDs of every news item you included "
        f"in the digest."
    )

    return Agent(
        name="digest_curator",
        instructions=instructions,
        tools=[search_news_by_relevance, search_news_by_recency, get_article_details],
        model=agents_model,
        output_type=DigestCurationResult,
        model_settings=ModelSettings(temperature=0.3),
    )


def _is_russian_language(digest_language: str) -> bool:
    return digest_language.strip().lower().split("-", maxsplit=1)[0] == "ru"


async def run_digest_curator(
    *,
    session: AsyncSession,
    query_embedding: list[float],
    exclude_ids: set[uuid.UUID],
    allowed_feed_ids: set[uuid.UUID],
    published_after: datetime,
    format_instructions: str,
    digest_language: str,
) -> DigestCurationResult | None:
    """Run the agentic digest curation flow.

    Returns a DigestCurationResult with the formatted digest and used item IDs,
    or None if no usable items were found.
    """
    agent = _create_digest_curator_agent(
        session=session,
        query_embedding=query_embedding,
        exclude_ids=exclude_ids,
        allowed_feed_ids=allowed_feed_ids,
        published_after=published_after,
        format_instructions=format_instructions,
        digest_language=digest_language,
    )
    result = await Runner.run(
        agent,
        input="Curate a news digest from available items.",
        run_config=RunConfig(tracing_disabled=True),
        max_turns=10,
    )
    output: DigestCurationResult = result.final_output  # type: ignore[assignment]
    if not output.used_item_ids:
        return None
    return output
