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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent_text
from news_service.agents.tools.source_inspection import build_fetch_source_items_tool
from news_service.core.config import get_settings
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

if TYPE_CHECKING:
    from news_service.agents.digest.pipeline import ReflectorSourceContext

logger = logging.getLogger(__name__)
settings = get_settings()

REFLECTOR_PROMPT = """\
# Context

This is a personalized news-digest service. A user describes what they want \
to follow in a freeform spec (topic, tone, exclusions, format). The backend \
keeps, per subscription, a small pool of sources (RSS feeds, Telegram \
channels, subreddits, X accounts) -- some added explicitly by the user, \
others auto-discovered by a separate agent to fill out the pool. Every 30 \
minutes those sources are polled; new items are embedded and stored.

On the user's schedule (e.g. daily at 8am), the digest pipeline runs:
  1. Fetch candidate items from the pool via cosine similarity + recency.
  2. A Writer agent composes a draft digest from those candidates.
  3. A Judge grades the draft; up to 3 revisions are attempted.
  4. The digest is delivered to the user via webhook.
  5. YOU run next, but only when a health signal fires (drift, silent \
source, REVISE verdict after max revisions, or periodic check).

# Your role

You are the Pipeline Reflector. You do not rewrite digests and you do not \
touch the user_spec. Your job is to keep the subscription's SOURCE POOL \
healthy so tomorrow's digest has better material to pull from. Everything \
else upstream (Writer, Judge, retrieval) re-runs on the next schedule and \
gets whatever pool you leave behind.

You are given:
- The specific reasons you were invoked (act on these first).
- A rich line per linked source with: URL, user-specified flag, today's \
contribution count, cosine of the source's long-run content to the \
subscription topic, days since last published item, how many of the last \
30 digests this source contributed to, its contribution rate \
(digest-included items / items published in the last 30 days), how many \
consecutive digests have passed since it last contributed, and a three-\
number cosine distribution of its recent items (p50, p90, std).
- The user's preferences (user_spec), the delivered digest, and judge \
scores if any.

How to read the source metrics:
- Low aggregate cosine + high p90 means the source drifts on average but \
still lands on-topic items occasionally -- consider keeping.
- High contribution-streak (digests since last contribution) means the \
Writer keeps skipping this source despite it publishing -- strong \
removal candidate.
- Low contribution_rate on a high-volume source signals noise; combined \
with a high streak, it's a clean remove.

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
- Never emit Markdown bold syntax (**...**) in emit_status messages or any \
other user-visible text. The frontend does not render it and the asterisks \
appear literally. Use plain text -- no bold markers at all.
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
        dist_parts: list[str] = []
        if ctx.item_cosine_p50 is not None:
            dist_parts.append(f"p50={ctx.item_cosine_p50:.2f}")
        if ctx.item_cosine_p90 is not None:
            dist_parts.append(f"p90={ctx.item_cosine_p90:.2f}")
        if ctx.item_cosine_std is not None:
            dist_parts.append(f"std={ctx.item_cosine_std:.2f}")
        dist = ",".join(dist_parts) if dist_parts else "n/a"
        lines.append(
            f"- [source_id={ctx.source_id}] {display} [{label}] | "
            f"today: {ctx.contribution_count} items | "
            f"topic: {drift} | item-cos: {dist} | {last} | "
            f"30d-digests: {ctx.contributed_last_30_digests} | "
            f"rate: {ctx.contribution_rate:.2f} | "
            f"streak: {ctx.digests_since_last_contribution}"
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

    # ADK dispatches tool calls in parallel when the LLM returns multiple
    # function_call parts in a single turn. Every reflector tool touches
    # the same AsyncSession, and asyncpg does not support overlapping
    # operations on one connection. Without this lock the session hits
    # "Session is already flushing" / "another operation is in progress"
    # and the session's connection ends up in a bad state that then
    # poisons the next task sharing the pool. Serializing every
    # DB-touching tool keeps the connection usage single-threaded.
    session_lock = asyncio.Lock()

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
        async with session_lock:
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

            source_result = await db_session.execute(
                select(Source).where(Source.id == link.source_id)
            )
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

    fetch_source_items = build_fetch_source_items_tool(
        db_session=db_session,
        allowed_source_ids=allowed_source_ids,
        topic_embedding=topic_embedding,
        name="fetch_source_items",
        session_lock=session_lock,
    )

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
