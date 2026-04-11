"""Conversational subscription parser using Chat Completions API."""

import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator
from typing import Any

import litellm

from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry
from news_service.schemas.conversation import AgentTurnOutput, ExistingSubscriptionContext

logger = logging.getLogger(__name__)
settings = get_settings()

_MAX_TOOL_ROUNDS = 3

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "validate_source_url",
            "description": (
                "Check if a source URL is reachable and has content. "
                "Use this to verify sources the user provides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The source URL to validate.",
                    },
                    "source_kind": {
                        "type": "string",
                        "enum": [
                            "rss",
                            "telegram_channel",
                            "reddit_subreddit",
                            "twitter_account",
                        ],
                        "description": "Source type.",
                    },
                },
                "required": ["url", "source_kind"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

SUBSCRIPTION_PARSER_PROMPT = """\
You are a friendly assistant helping the user set up a news subscription through a casual \
chat. Talk like a helpful friend, not a configuration wizard. Never mention technical \
terms like cron, RSS, feeds, webhooks, or API to the user.

Gather the following (internally, without exposing the technical names):
1. **Topic** — always present in the initial message.
2. **Delivery mode** — does the user want a periodic summary ("digest") or instant alerts \
about specific events like releases, premieres, or concerts ("event")? Default to digest \
unless they clearly want event alerts.
3. **Schedule** (digest only) — ask when they'd like to receive updates (e.g. "every \
morning", "twice a day"). Offer the option of only getting updates on demand. Internally \
convert to a 5-field cron expression:
   - "every morning" → "0 8 * * *"
   - "every evening at 9pm" → "0 21 * * *"
   - "every Saturday morning" → "0 8 * * 6"
   - "every third day" → "0 8 */3 * *"
   - "every hour" → "0 * * * *"
   - "every weekday at 9" → "0 9 * * 1-5"
   - "twice a day at 8 and 18" → "0 8,18 * * *"
4. **Language** — detect from the user's message language. Use the context preference if \
provided. Only ask if ambiguous (e.g. user writes in English about Russian-language content).
5. **Sources** — ask if they follow any specific Telegram channels, Reddit communities, or \
X/Twitter accounts on this topic. Extract source identifiers:
   - Telegram channels: @channel or t.me/channel → store as "channel" (no @ prefix)
   - Reddit subreddits: r/sub or reddit.com/r/sub → store as "sub" (no r/ prefix)
   - Twitter/X accounts: x.com/handle → store as "handle" (no @ prefix)
   When asking the user about sources, tell them to use @channel for Telegram and x.com/handle \
for X/Twitter to avoid ambiguity.
   Use the validate_source_url tool to verify sources are reachable when the user provides them. \
Build the full URL for validation: https://t.me/s/channel, https://www.reddit.com/r/sub/new/, \
https://x.com/handle.
6. **Source discovery** — if user provided sources, ask whether to also find additional \
sources automatically, or stick with only theirs.
7. **Format** — default to "brief summary" unless the user specifies preferences.

Behavior rules:
- Be friendly and concise. Ask at most ONE question per turn.
- Keep the conversation fully text-based — no buttons, no structured choices.
- Respond in the same language as the user.
- Never show cron expressions, technical field names, or internal config details to the user.
- If the user provides enough information in a single message to finalize, do so immediately.
- When all information is gathered, set status to "ready" and populate finalized_config.
- Accommodate mid-conversation changes (e.g. "actually make it weekly").
- For short_label: ultra-short 2-3 word category name in the user's language.
- For prompt_summary: concise 3-8 word description in the user's language.
- For canonical_prompt: copy the user's topic/request text with orthographical mistakes \
corrected. Do not rephrase or change wording — only fix spelling and grammar errors.

{context_section}\
"""


SUBSCRIPTION_EDIT_CONTEXT = """\
You are editing an EXISTING subscription, not creating a new one. The user wants to change \
something about their current subscription.

Current subscription state:
- Topic/prompt: {canonical_prompt}
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
- When the edit is clear, set status to "ready" with the complete updated config.
- If the user's request is ambiguous, ask ONE clarifying question.
- For canonical_prompt: update to reflect the new intent. If user only changes schedule/sources, \
keep the current canonical_prompt unchanged.
- short_label and prompt_summary: update only if the topic/intent changed.
"""


async def _execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "validate_source_url":
        args = json.loads(arguments)
        url: str = args["url"]
        source_kind: str = args["source_kind"]
        try:
            is_valid = await _validate_source_url(url, source_kind=source_kind)  # type: ignore[arg-type]
            if is_valid:
                return f"Source {url} ({source_kind}): valid and has content."
            return f"Source {url} ({source_kind}): could not fetch content or is empty."
        except Exception as exc:
            return f"Source {url} ({source_kind}): validation error: {exc}"
    return f"Unknown tool: {name}"


def _build_system_prompt(
    user_language: str | None,
    user_timezone: str | None,
    existing_config: ExistingSubscriptionContext | None = None,
) -> str:
    parts: list[str] = []
    if user_language:
        parts.append(f"User's preferred language: {user_language}.")
    if user_timezone:
        parts.append(f"User's timezone: {user_timezone}.")
    if existing_config is not None:
        parts.append(
            SUBSCRIPTION_EDIT_CONTEXT.format(
                canonical_prompt=existing_config.canonical_prompt,
                delivery_mode=existing_config.delivery_mode,
                schedule_cron=existing_config.schedule_cron or "none (manual only)",
                digest_language=existing_config.digest_language,
                format_instructions=existing_config.format_instructions,
                telegram_channels=", ".join(existing_config.fixed_telegram_channels) or "none",
                reddit_subreddits=", ".join(existing_config.fixed_reddit_subreddits) or "none",
                twitter_accounts=", ".join(existing_config.fixed_twitter_accounts) or "none",
            )
        )
    context_section = ""
    if parts:
        context_section = "Context:\n" + "\n".join(parts) + "\n"
    return SUBSCRIPTION_PARSER_PROMPT.format(context_section=context_section)


@with_llm_retry()
async def run_conversation_turn(
    messages: list[dict],
    *,
    user_language: str | None = None,
    user_timezone: str | None = None,
    existing_config: ExistingSubscriptionContext | None = None,
) -> tuple[AgentTurnOutput, list[dict]]:
    """Run a single conversation turn with the subscription parser.

    Returns the agent output and a list of new messages to append to the conversation
    history (tool calls, tool results, and the final assistant message).
    """
    system_msg: dict = {
        "role": "system",
        "content": _build_system_prompt(user_language, user_timezone, existing_config),
    }
    new_messages: list[dict] = []

    for _ in range(_MAX_TOOL_ROUNDS + 1):
        response = await chat_completion(
            messages=[system_msg, *messages, *new_messages],
            tools=TOOL_DEFINITIONS,
            response_format=AgentTurnOutput,
            temperature=0.2,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            output = msg.parsed
            if output is None:
                raise ValueError("LLM returned empty response for conversation turn")
            new_messages.append({"role": "assistant", "content": output.message})
            return output, new_messages

        # Store assistant message with tool calls
        tool_calls_data = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        new_messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_data,
            }
        )

        for tc in msg.tool_calls:
            result = await _execute_tool(tc.function.name, tc.function.arguments)
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    raise ValueError("Conversation turn exceeded maximum tool rounds")


_RETRYABLE = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)
_MAX_RETRY = 3


async def _parse_with_retry(system_msg: dict, all_messages: list[dict]) -> object:
    """Single LLM parse call with retry logic (for use inside async generators)."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRY + 1):
        try:
            return await chat_completion(
                messages=[system_msg, *all_messages],
                tools=TOOL_DEFINITIONS,
                response_format=AgentTurnOutput,
                temperature=0.2,
            )
        except _RETRYABLE as exc:
            last_error = exc
            if attempt == _MAX_RETRY:
                break
            delay = min(1.0 * (2 ** (attempt - 1)) + random.uniform(0, 0.5), 30.0)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt,
                _MAX_RETRY,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


def _source_display_name(url: str, source_kind: str) -> str:
    """Extract a user-friendly name from a source URL."""
    if source_kind == "telegram_channel":
        # https://t.me/s/channel → @channel
        name = url.rstrip("/").split("/")[-1]
        return f"@{name}"
    if source_kind == "reddit_subreddit":
        # https://www.reddit.com/r/sub/new/ → r/sub
        parts = url.rstrip("/").split("/")
        for i, p in enumerate(parts):
            if p == "r" and i + 1 < len(parts):
                return f"r/{parts[i + 1]}"
        return url
    if source_kind == "twitter_account":
        # https://x.com/handle → x.com/handle
        name = url.rstrip("/").split("/")[-1]
        return f"x.com/{name}"
    return url


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
    system_msg: dict = {
        "role": "system",
        "content": _build_system_prompt(user_language, user_timezone, existing_config),
    }
    new_messages: list[dict] = []

    for _ in range(_MAX_TOOL_ROUNDS + 1):
        response = await _parse_with_retry(system_msg, [*messages, *new_messages])
        msg = response.choices[0].message

        if not msg.tool_calls:
            output = msg.parsed
            if output is None:
                yield {"event": "error", "detail": "LLM returned empty response"}
                return
            new_messages.append({"role": "assistant", "content": output.message})
            yield {
                "event": "done",
                "output": output.model_dump(),
                "new_messages": new_messages,
            }
            return

        # Store assistant message with tool calls
        tool_calls_data = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        new_messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_data,
            }
        )

        for tc in msg.tool_calls:
            # Emit status before tool execution
            if tc.function.name == "validate_source_url":
                args = json.loads(tc.function.arguments)
                display = _source_display_name(args.get("url", ""), args.get("source_kind", ""))
                yield {"event": "status", "status_key": "status_checking_source", "source": display}

            result = await _execute_tool(tc.function.name, tc.function.arguments)
            new_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    yield {"event": "error", "detail": "Conversation turn exceeded maximum tool rounds"}
