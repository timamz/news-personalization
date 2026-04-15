"""Pipeline Reflector — ADK agent that reviews pipeline health and self-heals.

Converted from single-shot structured output to an ADK agent with tools:
- remove_source: hard-deletes dead auto-discovered sources (guards user-specified)
- trigger_source_discovery: flags that new sources should be found

Runs after digest delivery when data-driven triggers indicate health issues.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

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

logger = logging.getLogger(__name__)
settings = get_settings()

REFLECTOR_PROMPT = """\
You are a pipeline quality reflector. After a digest was generated and delivered, \
review how the pipeline performed and take corrective action.

You receive:
- The digest that was delivered
- The user's preferences (user_spec)
- Quality scores from the judge (empty if judge failed)
- Source contribution data showing which sources contributed items

You have two tools:
1. **remove_source(url, reason)** — Remove a dead or useless auto-discovered source. \
Only use this for sources marked [auto-discovered, removable]. \
NEVER attempt to remove sources marked [user-specified, DO NOT remove]. \
Only remove sources that contributed 0 items for 3+ weeks.
2. **trigger_source_discovery(reason)** — Request that new sources be found. \
Use after removing sources to find replacements, or when coverage is thin \
even without removals.

Guidelines:
- Be conservative with removals — only remove clearly dead sources.
- After removing sources, trigger discovery with a reason explaining what was lost.
- If coverage is thin but no sources need removal, still trigger discovery.
- If the composer ignored user preferences, mention it in your response \
so the pipeline can update user_spec.
"""


async def run_reflector(
    *,
    db_session: AsyncSession,
    subscription: Subscription,
    digest_text: str,
    user_spec: str,
    quality_scores: dict,
    source_info: str,
) -> dict[str, Any]:
    """Run the reflector ADK agent and return shared state with side effects.

    Returns a dict with:
        discovery_triggered: bool
        discovery_reason: str
        observations: str — reflector's text response (pipeline health notes)
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
            reason: Why the source should be removed (e.g. 'no content for 3+ weeks').

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

        Does not execute discovery immediately — sets a flag so the caller
        knows to queue a discovery task after the reflector finishes.

        Args:
            reason: Why new sources are needed (e.g. 'removed dead ML feed, need replacement').

        Returns:
            Confirmation that discovery was requested.
        """
        shared_state["discovery_triggered"] = True
        shared_state["discovery_reason"] = reason
        return f"Source discovery requested: {reason}"

    message = (
        f"User preferences (user_spec):\n{user_spec}\n\n"
        f"Quality scores: {quality_scores}\n\n"
        f"Linked sources:\n{source_info}\n\n"
        f"Delivered digest:\n{digest_text}"
    )

    agent = Agent(
        name=f"reflector_{uuid.uuid4().hex[:6]}",
        model=LiteLlm(model=settings.litellm_model),
        instruction=REFLECTOR_PROMPT,
        tools=[remove_source, trigger_source_discovery],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    response = await run_agent_text(
        agent=agent,
        message=message,
        user_id=str(subscription.user_id),
    )

    shared_state["observations"] = response
    logger.info(
        "Reflector completed for subscription %s: discovery_triggered=%s",
        subscription.id,
        shared_state["discovery_triggered"],
    )
    return shared_state
