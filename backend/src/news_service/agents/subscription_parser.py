"""Conversational subscription parser using Google ADK.

Uses ADK for the tool-calling loop (validate_source_url) and a finalize_subscription
tool to capture structured config. Conversation history is passed in the agent
instruction since the actual history lives in Redis (managed by the routes layer),
not in ADK sessions.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from news_service.agents.adk_runner import run_agent
from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ExistingSubscriptionContext,
    FinalizedSubscriptionConfig,
)

logger = logging.getLogger(__name__)
settings = get_settings()


SUBSCRIPTION_PARSER_PROMPT = """\
You are a friendly assistant helping the user set up a news subscription through a casual \
chat. Talk like a helpful friend, not a configuration wizard. Never mention technical \
terms like cron, RSS, feeds, webhooks, or API to the user.

Gather the following (internally, without exposing the technical names):
1. **Topic** -- always present in the initial message.
2. **Delivery mode** -- does the user want a periodic summary ("digest") or instant alerts \
about specific events like releases, premieres, or concerts ("event")? Default to digest \
unless they clearly want event alerts.
3. **Schedule** (digest only) -- ask when they'd like to receive updates (e.g. "every \
morning", "twice a day"). Offer the option of only getting updates on demand. Internally \
convert to a 5-field cron expression:
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
- If the user provides enough information in a single message to finalize, do so immediately.
- When all information is gathered, call finalize_subscription with the configuration, \
then respond with a confirmation message.
- Accommodate mid-conversation changes (e.g. "actually make it weekly").

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


def _extract_topic_from_user_spec(user_spec: str) -> str:
    """Extract the topic line from user_spec markdown."""
    for line in user_spec.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## Topic"):
            continue
        if stripped.startswith("##"):
            break
        if stripped:
            return stripped
    return user_spec[:200]


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


def _build_system_prompt(
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
        topic = _extract_topic_from_user_spec(existing_config.user_spec)
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
        context_section = "Context:\n" + "\n".join(parts) + "\n"
    return SUBSCRIPTION_PARSER_PROMPT.format(context_section=context_section)


def _create_parser_agent(
    *,
    user_language: str | None = None,
    user_timezone: str | None = None,
    conversation_history: list[dict] | None = None,
    existing_config: ExistingSubscriptionContext | None = None,
) -> tuple[Agent, dict[str, Any]]:
    """Create the subscription parser ADK agent with tools.

    Returns the agent and a shared_state dict that the finalize tool writes to.
    """
    shared_state: dict[str, Any] = {
        "status": "in_progress",
        "finalized_config": None,
    }

    async def validate_source(url: str, source_kind: str) -> str:
        """Check if a source URL is reachable and has content.

        Use this to verify sources the user provides.

        Args:
            url: The source URL to validate.
            source_kind: One of: rss, telegram_channel, reddit_subreddit, twitter_account.

        Returns:
            Whether the source is valid and reachable.
        """
        try:
            is_valid = await _validate_source_url(url, source_kind=source_kind)
            if is_valid:
                return f"Source {url} ({source_kind}): valid and has content."
            return f"Source {url} ({source_kind}): could not fetch content or is empty."
        except Exception as exc:
            return f"Source {url} ({source_kind}): validation error: {exc}"

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

    instruction = _build_system_prompt(
        user_language, user_timezone, conversation_history, existing_config
    )

    agent = Agent(
        name="subscription_parser",
        model=LiteLlm(model=settings.litellm_model),
        instruction=instruction,
        tools=[validate_source, finalize_subscription],
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )

    return agent, shared_state


async def run_conversation_turn(
    messages: list[dict],
    *,
    user_language: str | None = None,
    user_timezone: str | None = None,
    existing_config: ExistingSubscriptionContext | None = None,
) -> tuple[AgentTurnOutput, list[dict]]:
    """Run a single conversation turn with the subscription parser.

    Returns the agent output and a list of new messages to append to the conversation
    history (the final assistant message).
    """
    previous_messages = messages[:-1] if len(messages) > 1 else []
    current_message = messages[-1]["content"] if messages else ""

    agent, shared_state = _create_parser_agent(
        user_language=user_language,
        user_timezone=user_timezone,
        conversation_history=previous_messages,
        existing_config=existing_config,
    )

    agent_text = await run_agent(
        agent=agent,
        message=current_message,
    )

    output = AgentTurnOutput(
        message=agent_text,
        status=shared_state["status"],
        finalized_config=shared_state["finalized_config"],
    )
    new_messages = [{"role": "assistant", "content": agent_text}]
    return output, new_messages


async def run_conversation_turn_streaming(
    messages: list[dict],
    *,
    user_language: str | None = None,
    user_timezone: str | None = None,
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

    agent, shared_state = _create_parser_agent(
        user_language=user_language,
        user_timezone=user_timezone,
        conversation_history=previous_messages,
        existing_config=existing_config,
    )

    agent_text = ""
    try:
        async for event in await run_agent(
            agent=agent,
            message=current_message,
            streaming=True,
        ):
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
    except Exception as exc:
        logger.exception("Subscription parser agent failed")
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
