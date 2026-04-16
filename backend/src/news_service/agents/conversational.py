"""Conversational agent -- single agent per user for all subscription management.

The agent maintains user_spec as the primary representation of user intent.
All subscription management (create, edit, feedback, source discovery) happens
through this agent's tools. Conversation state persists in Redis between turns;
user_spec and structured fields persist in the database.

This module also provides a streaming entry point for the subscription setup
conversation flow (formerly in subscription_parser.py). The streaming function
yields status events as tools execute and a final done event with the result.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent, run_agent_text
from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.models.subscription import Subscription
from news_service.models.user import User
from news_service.models.user_spec import extract_topic, validate_user_spec
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ExistingSubscriptionContext,
    FinalizedSubscriptionConfig,
)

logger = logging.getLogger(__name__)
settings = get_settings()

CONVERSATIONAL_AGENT_PROMPT = """\
You are a personal news assistant. You help the user set up and manage \
news subscriptions through casual conversation.

You have access to the user's current preferences (user_spec) and a summary \
of past conversations. Use this context to understand what the user wants \
without asking them to repeat themselves.

When the user wants to create or edit a subscription, gather:
1. **Topic** -- what they want to follow.
2. **Delivery mode** -- does the user want a periodic summary ("digest") or instant alerts \
about specific events like releases, premieres, or concerts ("event")? Default to digest \
unless they clearly want event alerts.
3. **Schedule** (digest only) -- ask when they'd like to receive updates (e.g. "every \
morning", "twice a day"). Offer the option of only getting updates on demand. Internally \
convert to a 5-field cron expression. Never show cron to the user.
   - "every morning" -> "0 8 * * *"
   - "every evening at 9pm" -> "0 21 * * *"
   - "every Saturday morning" -> "0 8 * * 6"
   - "every third day" -> "0 8 */3 * *"
   - "every hour" -> "0 * * * *"
   - "every weekday at 9" -> "0 9 * * 1-5"
   - "twice a day at 8 and 18" -> "0 8,18 * * *"
4. **Language** -- detect from the user's message language. Use the context preference if \
provided. Only ask if ambiguous (e.g. user writes in English about Russian-language content).
5. **Sources** -- ask if they follow any specific Telegram channels, Reddit communities, or \
X/Twitter accounts on this topic. Extract source identifiers:
   - Telegram channels: @channel or t.me/channel -> store as "channel" (no @ prefix)
   - Reddit subreddits: r/sub or reddit.com/r/sub -> store as "sub" (no r/ prefix)
   - Twitter/X accounts: x.com/handle -> store as "handle" (no @ prefix)
   When asking the user about sources, tell them to use @channel for Telegram and x.com/handle \
for X/Twitter to avoid ambiguity.
   Use the validate_source tool to verify sources are reachable when the user provides them. \
Build the full URL for validation: https://t.me/s/channel, https://www.reddit.com/r/sub/new/, \
https://x.com/handle.
6. **Source discovery** -- if user provided sources, ask whether to also find additional \
sources automatically, or stick with only theirs.
7. **Format** -- default to "brief summary" unless the user specifies preferences.

Behavior rules:
- Be friendly and concise. Ask at most ONE question per turn.
- Keep the conversation fully text-based -- no buttons, no structured choices.
- Respond in the same language as the user.
- Never show cron expressions, technical field names, or internal config details to the user.
- If the user provides enough information in a single message, act immediately.
- Accommodate mid-conversation changes (e.g. "actually make it weekly").
- When you modify preferences, always call update_user_spec with the full updated spec.
- When all information is gathered, call finalize_subscription with the configuration, \
then respond with a friendly confirmation message.
- When the user gives feedback about digest quality, update user_spec with their preferences.

{context_section}\
"""


SUBSCRIPTION_EDIT_CONTEXT = """\
You are editing an EXISTING subscription, not creating a new one. The user wants to change \
something about their current subscription.

Current subscription state:
- Topic: {user_spec_topic}
- Delivery mode: {delivery_mode}
- Schedule: {schedule_cron}
- Language: {digest_language}
- Format: {format_instructions}
- Telegram channels: {telegram_channels}
- Reddit subreddits: {reddit_subreddits}
- Twitter/X accounts: {twitter_accounts}

Edit rules:
- Treat the user's message as an incremental change to the existing subscription.
- Preserve ALL fields the user does not mention changing.
- The user can change any aspect: topic, schedule, sources, format, delivery mode.
- For fields the user does NOT mention, carry forward the current values exactly.
- When the edit is clear, call finalize_subscription with the complete updated config.
- If the user's request is ambiguous, ask ONE clarifying question.
"""


def _source_display_name(url: str, source_kind: str) -> str:
    """Extract a user-friendly name from a source URL."""
    if source_kind == "telegram_channel":
        name = url.rstrip("/").split("/")[-1]
        return f"@{name}"
    if source_kind == "reddit_subreddit":
        parts = url.rstrip("/").split("/")
        for i, p in enumerate(parts):
            if p == "r" and i + 1 < len(parts):
                return f"r/{parts[i + 1]}"
        return url
    if source_kind == "twitter_account":
        name = url.rstrip("/").split("/")[-1]
        return f"x.com/{name}"
    return url


def _build_instruction(
    user_spec: str,
    conversation_summary: str,
    user_language: str | None,
    user_timezone: str | None,
    conversation_history: list[dict] | None = None,
    existing_config: ExistingSubscriptionContext | None = None,
) -> str:
    parts: list[str] = []
    if user_language:
        parts.append(f"User's preferred language: {user_language}.")
    if user_timezone:
        parts.append(f"User's timezone: {user_timezone}.")
    if existing_config is not None:
        topic = extract_topic(existing_config.user_spec)
        parts.append(
            SUBSCRIPTION_EDIT_CONTEXT.format(
                user_spec_topic=topic,
                delivery_mode=existing_config.delivery_mode,
                schedule_cron=existing_config.schedule_cron or "none (manual only)",
                digest_language=existing_config.digest_language,
                format_instructions=existing_config.format_instructions,
                telegram_channels=", ".join(existing_config.fixed_telegram_channels) or "none",
                reddit_subreddits=", ".join(existing_config.fixed_reddit_subreddits) or "none",
                twitter_accounts=", ".join(existing_config.fixed_twitter_accounts) or "none",
            )
        )
    if user_spec:
        parts.append(f"Current user_spec:\n{user_spec}")
    if conversation_summary:
        parts.append(f"Conversation summary:\n{conversation_summary}")
    if conversation_history:
        history_lines: list[str] = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                history_lines.append(f"{role.capitalize()}: {content}")
        if history_lines:
            parts.append("Previous conversation:\n" + "\n".join(history_lines))
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
    conversation_history: list[dict] | None = None,
    existing_config: ExistingSubscriptionContext | None = None,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
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
        "status": "in_progress",
        "finalized_config": None,
    }

    async def finalize_subscription(
        delivery_mode: str = "digest",
        schedule_cron: str = "",
        digest_language: str = "en",
        format_instructions: str = "brief summary",
        fixed_telegram_channels: str = "",
        fixed_reddit_subreddits: str = "",
        fixed_twitter_accounts: str = "",
        include_discovered_sources: bool = True,
        manual_only: bool = False,
    ) -> str:
        """Call this when all subscription information is gathered and the user has confirmed.

        Args:
            delivery_mode: Either 'digest' for periodic summaries or 'event' for instant alerts.
            schedule_cron: 5-field cron expression for digest schedule. Empty for event or manual.
            digest_language: Language code for content (e.g. 'en', 'ru').
            format_instructions: How to format the digest (e.g. 'brief summary', 'detailed').
            fixed_telegram_channels: Comma-separated channel names (no @ prefix).
            fixed_reddit_subreddits: Comma-separated subreddit names (no r/ prefix).
            fixed_twitter_accounts: Comma-separated account handles (no @ prefix).
            include_discovered_sources: Whether to auto-discover additional sources.
            manual_only: If true, digest is only sent on explicit request.

        Returns:
            Confirmation that the configuration was saved.
        """
        telegram = [c.strip() for c in fixed_telegram_channels.split(",") if c.strip()]
        reddit = [s.strip() for s in fixed_reddit_subreddits.split(",") if s.strip()]
        twitter = [a.strip() for a in fixed_twitter_accounts.split(",") if a.strip()]

        shared_state["status"] = "ready"
        shared_state["finalized_config"] = FinalizedSubscriptionConfig(
            delivery_mode=delivery_mode,
            schedule_cron=schedule_cron or None,
            manual_only=manual_only,
            format_instructions=format_instructions,
            digest_language=digest_language,
            fixed_telegram_channels=telegram,
            fixed_reddit_subreddits=reddit,
            fixed_twitter_accounts=twitter,
            include_discovered_sources=include_discovered_sources,
        )
        return "Configuration saved. Now respond to the user with a friendly confirmation."

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

    async def emit_status(message: str) -> str:
        """Emit a progress status message to the user.

        Call this when starting a significant operation to keep the user informed.
        Write in the same language as the conversation.

        Args:
            message: A short, friendly progress message
                (e.g. "Looking for Telegram channels about AI...")

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

    instruction = _build_instruction(
        user_spec=user_spec,
        conversation_summary=conversation_summary,
        user_language=user_language,
        user_timezone=user.timezone,
        conversation_history=conversation_history,
        existing_config=existing_config,
    )

    agent = Agent(
        name="conversational_agent",
        model=LiteLlm(model=settings.litellm_model),
        instruction=instruction,
        tools=[
            finalize_subscription,
            create_subscription,
            update_user_spec,
            validate_source,
            discover_sources,
            list_subscriptions,
            trigger_digest_now,
            delete_subscription,
            emit_status,
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
        agent_message: str -- what to show the user
        user_spec_updated: bool -- whether user_spec changed
        new_user_spec: str -- the updated user_spec (if changed)
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


async def run_conversation_turn_streaming(
    messages: list[dict],
    *,
    db_session: AsyncSession,
    user: User,
    user_spec: str,
    conversation_summary: str,
    user_language: str | None = None,
    subscription_id: str | None = None,
    existing_config: ExistingSubscriptionContext | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming variant that yields status events and a final done event.

    Events:
      {"event": "status", "status_key": "...", ...optional kwargs}
      {"event": "done", "output": {...}, "new_messages": [...]}
      {"event": "error", "detail": "..."}
    """
    previous_messages = messages[:-1] if len(messages) > 1 else []
    current_message = messages[-1]["content"] if messages else ""

    status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    agent, shared_state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec=user_spec,
        conversation_summary=conversation_summary,
        user_language=user_language,
        subscription_id=subscription_id,
        conversation_history=previous_messages,
        existing_config=existing_config,
        status_queue=status_queue,
    )

    agent_text = ""
    try:
        async for event in run_agent(
            agent=agent,
            message=current_message,
            user_id=str(user.id),
        ):
            while not status_queue.empty():
                yield status_queue.get_nowait()
            if event["type"] == "tool_call":
                tool_name = event["name"]
                if tool_name == "validate_source":
                    args = event.get("args", {})
                    url = args.get("url", "")
                    source_kind = args.get("source_kind", "")
                    display = _source_display_name(url, source_kind)
                    yield {
                        "event": "status",
                        "status_key": "status_checking_source",
                        "source": display,
                    }
            elif event["type"] == "final_response":
                agent_text = event["text"]
        while not status_queue.empty():
            yield status_queue.get_nowait()
    except Exception as exc:
        logger.exception("Conversational agent streaming failed")
        yield {"event": "error", "detail": f"Agent error: {exc}"}
        return

    output = AgentTurnOutput(
        message=agent_text,
        status=shared_state["status"],
        finalized_config=shared_state["finalized_config"],
    )
    new_messages = [{"role": "assistant", "content": agent_text}]
    yield {
        "event": "done",
        "output": output.model_dump(),
        "new_messages": new_messages,
    }
