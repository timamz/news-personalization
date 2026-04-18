"""Pipeline Reflector -- ADK agent that reviews pipeline health and self-heals.

Runs after digest delivery when data-driven triggers (drift, staleness, REVISE
after max revisions, periodic) indicate a health issue. The Reflector receives
the specific reasons it was invoked plus rich per-source metadata so it can
make targeted decisions rather than rediscovering signals.

Tools:
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

You have three tools:
1. **remove_source(url, reason)** -- Remove a dead or off-topic \
auto-discovered source. NEVER attempt to remove sources marked \
[user-specified]. Remove only when you have concrete evidence (drift, \
prolonged silence).
2. **trigger_source_discovery(reason)** -- Request new sources. Use after \
removing sources, or when the subscription's source pool is clearly thin \
or off-topic.
3. **emit_status(message)** -- Emit a short progress status to the user.

Guidelines:
- Start from the invocation reasons. They name the specific sources you \
should investigate first.
- Be conservative. Do not remove a user-specified source.
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
        tools=[remove_source, trigger_source_discovery, emit_status],
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
