"""Conversational agent — single agent per user for all subscription management.

The agent maintains user_spec as the primary representation of user intent.
All subscription management (create, edit, feedback, source discovery) happens
through this agent's tools. Conversation state persists in Redis between turns;
user_spec and structured fields persist in the database.
"""

import logging
import uuid
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent_text
from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.models.subscription import Subscription
from news_service.models.user import User
from news_service.models.user_spec import validate_user_spec

logger = logging.getLogger(__name__)
settings = get_settings()

CONVERSATIONAL_AGENT_PROMPT = """\
You are a personal news assistant. You help the user set up and manage \
news subscriptions through casual conversation.

You have access to the user's current preferences (user_spec) and a summary \
of past conversations. Use this context to understand what the user wants \
without asking them to repeat themselves.

When the user wants to create or edit a subscription, gather:
1. **Topic** — what they want to follow.
2. **Delivery mode** — periodic digest or instant event alerts.
3. **Schedule** (digest only) — convert to 5-field cron. Never show cron to the user.
4. **Language** — detect from conversation or ask if ambiguous.
5. **Sources** — ask if they follow specific Telegram/Reddit/Twitter sources.

Behavior rules:
- Be friendly and concise. Ask at most ONE question per turn.
- Keep it fully text-based — no buttons, no structured choices.
- Respond in the same language as the user.
- Never show cron expressions, technical names, or config internals.
- If the user provides enough info in one message, act immediately.
- When you modify preferences, always call update_user_spec with the full updated spec.
- When creating a subscription, call create_subscription with the structured fields, \
then call discover_sources if the user wants automatic source discovery.
- When the user gives feedback about digest quality, update user_spec with their preferences.

{context_section}\
"""


def _build_instruction(
    user_spec: str,
    conversation_summary: str,
    user_language: str | None,
    user_timezone: str | None,
) -> str:
    parts: list[str] = []
    if user_language:
        parts.append(f"User's preferred language: {user_language}.")
    if user_timezone:
        parts.append(f"User's timezone: {user_timezone}.")
    if user_spec:
        parts.append(f"Current user_spec:\n{user_spec}")
    if conversation_summary:
        parts.append(f"Conversation summary:\n{conversation_summary}")
    context_section = ""
    if parts:
        context_section = "Context:\n" + "\n\n".join(parts) + "\n"
    return CONVERSATIONAL_AGENT_PROMPT.format(context_section=context_section)


def create_conversational_agent(
    *,
    db_session: AsyncSession,
    user: User,
    user_spec: str,
    conversation_summary: str,
    user_language: str | None = None,
    subscription_id: str | None = None,
) -> tuple[Agent, dict[str, Any]]:
    """Create a conversational agent with tools bound to the current DB session.

    Returns the agent and a shared state dict that tools write side effects to.
    """
    shared_state: dict[str, Any] = {
        "user_spec_updated": False,
        "new_user_spec": user_spec,
        "subscription_created": False,
        "created_subscription_id": None,
        "discovery_triggered": False,
    }

    async def create_subscription(
        delivery_mode: str,
        schedule_cron: str = "",
        digest_language: str = "en",
        format_instructions: str = "brief summary",
    ) -> str:
        """Create a new subscription with the given structured settings.

        Call this when the user has confirmed they want a new subscription.

        Args:
            delivery_mode: Either 'digest' for periodic summaries or 'event' for instant alerts.
            schedule_cron: 5-field cron expression for digest schedule. Empty for event mode.
            digest_language: Language code for the digest content (e.g. 'en', 'ru').
            format_instructions: How to format the digest (e.g. 'brief summary', 'detailed').

        Returns:
            Confirmation message with the subscription ID.
        """
        sub = Subscription(
            user_id=user.id,
            raw_prompt=shared_state["new_user_spec"][:500],
            delivery_mode=delivery_mode,
            schedule_cron=schedule_cron or None,
            digest_language=digest_language,
            format_instructions=format_instructions,
            user_spec=shared_state["new_user_spec"],
        )
        db_session.add(sub)
        await db_session.flush()
        shared_state["subscription_created"] = True
        shared_state["created_subscription_id"] = str(sub.id)
        return f"Subscription created with ID {sub.id}."

    async def update_user_spec(content: str) -> str:
        """Update the user_spec document with the full new content.

        Always pass the COMPLETE user_spec, not just the changed section.
        The user_spec should contain sections like ## Topic, ## Sources,
        ## Schedule, ## Preferences, ## Feedback, etc.

        Args:
            content: The complete updated user_spec markdown text.

        Returns:
            Confirmation that the spec was updated.
        """
        try:
            validated = validate_user_spec(content)
        except (ValueError, Exception) as exc:
            return f"Invalid user spec: {exc}"

        shared_state["user_spec_updated"] = True
        shared_state["new_user_spec"] = validated

        if subscription_id:
            from sqlalchemy import select

            result = await db_session.execute(
                select(Subscription).where(Subscription.id == uuid.UUID(subscription_id))
            )
            sub = result.scalar_one_or_none()
            if sub:
                sub.user_spec = validated
                await db_session.flush()

        return "User spec updated."

    async def validate_source(url: str, source_kind: str) -> str:
        """Check if a source URL is reachable and has content.

        Args:
            url: Full source URL to validate.
            source_kind: One of: rss, telegram_channel, reddit_subreddit, twitter_account.

        Returns:
            Whether the source is valid and has content.
        """
        try:
            is_valid = await _validate_source_url(url, source_kind=source_kind)
            if is_valid:
                return f"Source {url} ({source_kind}): valid and has content."
            return f"Source {url} ({source_kind}): could not fetch content or is empty."
        except Exception as exc:
            return f"Source {url} ({source_kind}): validation error: {exc}"

    async def discover_sources(topic: str) -> str:
        """Signal that automatic source discovery should run after this conversation turn.

        This does NOT execute discovery immediately. It sets a flag so the caller
        knows to kick off the discovery pipeline after the turn completes.

        Args:
            topic: The topic to find sources for (from user_spec).

        Returns:
            Confirmation that the discovery request was recorded.
        """
        shared_state["discovery_triggered"] = True
        shared_state["discovery_topic"] = topic
        return (
            "Source discovery request recorded. "
            "The system will search for relevant sources after this conversation turn."
        )

    async def list_subscriptions() -> str:
        """List all active subscriptions for the current user.

        Returns:
            Formatted list of subscriptions with their topics and settings.
        """
        from sqlalchemy import select

        result = await db_session.execute(
            select(Subscription).where(
                Subscription.user_id == user.id,
                Subscription.is_active.is_(True),
            )
        )
        subs = list(result.scalars().all())
        if not subs:
            return "No active subscriptions."
        lines = []
        for s in subs:
            topic = (s.user_spec or s.raw_prompt)[:80]
            lines.append(
                f"- [{s.id}] {s.delivery_mode}: {topic} "
                f"(schedule: {s.schedule_cron or 'event mode'})"
            )
        return f"Active subscriptions:\n{chr(10).join(lines)}"

    async def trigger_digest_now(subscription_id: str) -> str:
        """Queue an immediate digest delivery for a subscription.

        Args:
            subscription_id: UUID of the subscription to deliver.

        Returns:
            Confirmation that the digest was queued.
        """
        from news_service.tasks.deliver_digest import deliver_digest

        deliver_digest.delay(subscription_id, notify_if_empty=True)
        return f"Digest queued for delivery (subscription {subscription_id})."

    async def delete_subscription(subscription_id: str) -> str:
        """Delete a subscription by ID.

        Args:
            subscription_id: UUID of the subscription to delete.

        Returns:
            Confirmation that the subscription was deleted.
        """
        from sqlalchemy import select

        result = await db_session.execute(
            select(Subscription).where(
                Subscription.id == uuid.UUID(subscription_id),
                Subscription.user_id == user.id,
            )
        )
        sub = result.scalar_one_or_none()
        if sub is None:
            return f"Subscription {subscription_id} not found."
        sub.is_active = False
        await db_session.flush()
        return f"Subscription {subscription_id} deleted."

    instruction = _build_instruction(
        user_spec=user_spec,
        conversation_summary=conversation_summary,
        user_language=user_language,
        user_timezone=user.timezone,
    )

    agent = Agent(
        name="conversational_agent",
        model=LiteLlm(model=settings.litellm_model),
        instruction=instruction,
        tools=[
            create_subscription,
            update_user_spec,
            validate_source,
            discover_sources,
            list_subscriptions,
            trigger_digest_now,
            delete_subscription,
        ],
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )

    return agent, shared_state


async def run_conversational_turn(
    *,
    db_session: AsyncSession,
    user: User,
    user_message: str,
    user_spec: str,
    conversation_summary: str,
    user_language: str | None = None,
    subscription_id: str | None = None,
) -> dict[str, Any]:
    """Run a single conversational turn and return the result.

    Returns a dict with:
        agent_message: str — what to show the user
        user_spec_updated: bool — whether user_spec changed
        new_user_spec: str — the updated user_spec (if changed)
        subscription_created: bool
        created_subscription_id: str | None
        discovery_triggered: bool
    """
    agent, shared_state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec=user_spec,
        conversation_summary=conversation_summary,
        user_language=user_language,
        subscription_id=subscription_id,
    )

    agent_message = await run_agent_text(
        agent=agent,
        message=user_message,
        user_id=str(user.id),
    )

    return {
        "agent_message": agent_message,
        "user_spec_updated": shared_state["user_spec_updated"],
        "new_user_spec": shared_state["new_user_spec"],
        "subscription_created": shared_state["subscription_created"],
        "created_subscription_id": shared_state["created_subscription_id"],
        "discovery_triggered": shared_state["discovery_triggered"],
    }
