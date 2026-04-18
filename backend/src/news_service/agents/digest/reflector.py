"""Pipeline Reflector -- ADK agent that reviews pipeline health and self-heals.

Runs after digest delivery when data-driven triggers (drift, staleness, REVISE
after max revisions, periodic) indicate a health issue. The Reflector receives
the specific reasons it was invoked plus rich per-source metadata so it can
make targeted decisions rather than rediscovering signals.

Tools:
- fetch_source_items: inspect a source's recent items before deciding
- remove_source: delete dead or off-topic auto-discovered sources
- trigger_source_discovery: queue discovery to find new sources
- emit_status: emit a progress status to the user
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent_text
from news_service.core.config import get_settings
from news_service.models.news_item import NewsItem
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.relevance import cosine_similarity

if TYPE_CHECKING:
    from news_service.agents.digest.pipeline import ReflectorSourceContext

logger = logging.getLogger(__name__)
settings = get_settings()

REFLECTOR_PROMPT = """\
You are a pipeline quality reflector. After a digest was generated and \
delivered, review how the pipeline performed and take corrective action on \
the source pool backing this subscription.

You are given:
- The specific reasons you were invoked (act on these first).
- A rich line per linked source: URL, user-specified flag, contribution \
count in this digest, cosine similarity to the subscription topic, and \
days since last published item.
- The user's preferences (user_spec), the delivered digest, and judge scores \
if any.

You have four tools:
1. **fetch_source_items(source_id, since_days_ago, limit)** -- Read recent \
items from a specific linked source to examine its content before deciding. \
Always call this before removing a source whose content you have not seen.
2. **remove_source(url, reason)** -- Remove a dead or off-topic \
auto-discovered source. NEVER attempt to remove sources marked \
[user-specified]. Remove only when you have concrete evidence (drift, \
prolonged silence, or content inspection via fetch_source_items).
3. **trigger_source_discovery(reason)** -- Request new sources. Use after \
removing sources, or when the subscription's source pool is clearly thin \
or off-topic.
4. **emit_status(message)** -- Emit a short progress status to the user.

Guidelines:
- Start from the invocation reasons. They name the specific sources you \
should investigate first.
- Prefer inspect before remove: fetch_source_items to verify, then remove.
- Be conservative. Do not remove a user-specified source or a source that \
looks healthy on inspection.
- After removing sources, trigger discovery with a reason naming what was \
lost so the finder knows what to look for.
"""


def _format_source_contexts(contexts: list["ReflectorSourceContext"]) -> str:
    """Render the per-source metadata list passed to the Reflector."""
    lines: list[str] = []
    for ctx in contexts:
        label = "user-specified, DO NOT remove" if ctx.is_user_specified else "auto-discovered"
        display = f"{ctx.title} ({ctx.url})" if ctx.title else ctx.url
        drift = f"cos={ctx.cosine_to_topic:.2f}" if ctx.cosine_to_topic is not None else "cos=n/a"
        if ctx.days_since_last_published is None:
            last = "last item: never"
        else:
            last = f"last item: {ctx.days_since_last_published}d ago"
        lines.append(
            f"- [source_id={ctx.source_id}] {display} [{label}] | "
            f"contributed: {ctx.contribution_count} items | "
            f"topic: {drift} | {last}"
        )
    return "\n".join(lines) if lines else "(no sources linked)"


async def run_reflector(
    *,
    db_session: AsyncSession,
    subscription: Subscription,
    digest_text: str,
    user_spec: str,
    quality_scores: dict,
    trigger_reasons: list[str],
    source_contexts: list["ReflectorSourceContext"],
    allowed_source_ids: set[uuid.UUID],
    topic_embedding: list[float],
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the Reflector ADK agent; return shared state recording side effects.

    Returns a dict with:
        discovery_triggered: bool
        discovery_reason: str
        observations: str -- agent's final text response
    """
    shared_state: dict[str, Any] = {
        "discovery_triggered": False,
        "discovery_reason": "",
    }

    async def remove_source(url: str, reason: str) -> str:
        """Remove a dead or useless source from this subscription.

        Only works for auto-discovered sources. User-specified sources cannot
        be removed by the reflector.

        Args:
            url: The canonical URL of the source to remove.
            reason: Why the source should be removed.

        Returns:
            Confirmation or rejection message.
        """
        link_result = await db_session.execute(
            select(SubscriptionSource)
            .join(Source, Source.id == SubscriptionSource.source_id)
            .where(
                SubscriptionSource.subscription_id == subscription.id,
                Source.url == url,
            )
        )
        link = link_result.scalar_one_or_none()
        if link is None:
            return f"Source {url} is not linked to this subscription."

        if link.is_user_specified:
            return f"Cannot remove {url}: this is a user-specified source."

        source_result = await db_session.execute(select(Source).where(Source.id == link.source_id))
        source = source_result.scalar_one()

        await db_session.delete(link)
        source.subscriber_count = max(source.subscriber_count - 1, 0)
        if source.subscriber_count == 0:
            source.is_active = False

        db_session.add(
            SourceRemovalLog(
                subscription_id=subscription.id,
                source_url=url,
                removed_at=datetime.now(UTC),
                removal_reason=reason,
            )
        )
        await db_session.flush()

        logger.info(
            "Reflector removed source %s from subscription %s: %s",
            url,
            subscription.id,
            reason,
        )
        return f"Removed source {url} (reason: {reason})."

    async def trigger_source_discovery(reason: str) -> str:
        """Request that the discovery pipeline find new sources.

        Does not execute discovery immediately -- sets a flag so the caller
        knows to queue a discovery task after the reflector finishes.

        Args:
            reason: Why new sources are needed.

        Returns:
            Confirmation that discovery was requested.
        """
        shared_state["discovery_triggered"] = True
        shared_state["discovery_reason"] = reason
        return f"Source discovery requested: {reason}"

    async def fetch_source_items(
        source_id: str,
        since_days_ago: int = 14,
        limit: int = 10,
    ) -> str:
        """Fetch recent items from a specific linked source for inspection.

        Useful before removing a source or to verify drift. The source must
        be linked to this subscription; cross-subscription access is refused.

        Args:
            source_id: UUID of the source to inspect.
            since_days_ago: How far back to look, in days. Default 14.
            limit: Max number of items to return. Capped at the server limit.

        Returns:
            Formatted list of items (headline, body snippet, published_at,
            cosine similarity to the subscription topic) or an error message.
        """
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
            if item.embedding is not None:
                try:
                    sim = cosine_similarity(list(item.embedding), topic_embedding)
                    sim_str = f"{sim:.2f}"
                except Exception:
                    sim_str = "n/a"
            else:
                sim_str = "n/a"
            body_snippet = (item.body or "")[:300].replace("\n", " ").strip()
            lines.append(f"- [{published}] cos={sim_str} | {item.headline}\n    {body_snippet}")
        return "\n".join(lines)

    async def emit_status(message: str) -> str:
        """Emit a progress status message to the user.

        Args:
            message: A short, friendly progress message.

        Returns:
            Confirmation that the status was emitted.
        """
        if status_queue is not None:
            status_queue.put_nowait(
                {
                    "event": "status",
                    "status_key": "status_agent_progress",
                    "status_text": message,
                }
            )
        return "Status emitted."

    reasons_block = (
        "\n".join(f"- {reason}" for reason in trigger_reasons)
        if trigger_reasons
        else "- (none given)"
    )
    source_block = _format_source_contexts(source_contexts)

    input_message = (
        f"You were invoked because:\n{reasons_block}\n\n"
        f"Linked sources:\n{source_block}\n\n"
        f"User preferences (user_spec):\n{user_spec}\n\n"
        f"Quality scores: {quality_scores}\n\n"
        f"Delivered digest:\n{digest_text}"
    )

    agent = Agent(
        name=f"reflector_{uuid.uuid4().hex[:6]}",
        model=LiteLlm(model=settings.litellm_model),
        instruction=REFLECTOR_PROMPT,
        tools=[fetch_source_items, remove_source, trigger_source_discovery, emit_status],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    response = await run_agent_text(
        agent=agent,
        message=input_message,
        user_id=str(subscription.user_id),
    )

    shared_state["observations"] = response
    logger.info(
        "Reflector completed for subscription %s: discovery_triggered=%s",
        subscription.id,
        shared_state["discovery_triggered"],
    )
    return shared_state
