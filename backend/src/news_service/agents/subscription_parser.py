"""Conversational subscription parser using Chat Completions API."""

import json
import logging

from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry
from news_service.core.openai_client import openai_client
from news_service.schemas.conversation import AgentTurnOutput

logger = logging.getLogger(__name__)
settings = get_settings()
_client = openai_client

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
You are a subscription setup assistant. Your job is to help the user configure a news \
subscription by gathering all required information through a brief conversation.

You must determine the following:
1. **Topic/interest** — always present in the initial message.
2. **Delivery mode** — "digest" (periodic summary) or "event" (instant notification about \
upcoming events like releases, premieres, concerts). Default to "digest" unless the user \
clearly wants event notifications.
3. **Schedule** (digest only) — if not specified, ask. Offer "manual only" as an option. \
Write the cron expression yourself using standard 5-field syntax (minute hour day-of-month \
month day-of-week). Examples:
   - "every morning" → "0 8 * * *"
   - "every evening at 9pm" → "0 21 * * *"
   - "every Saturday morning" → "0 8 * * 6"
   - "every third day" → "0 8 */3 * *"
   - "every hour" → "0 * * * *"
   - "every weekday at 9" → "0 9 * * 1-5"
   - "twice a day at 8 and 18" → "0 8,18 * * *"
4. **Language** — detect from the user's message language. Use the context preference if \
provided. Only ask if ambiguous (e.g. user writes in English about Russian-language content).
5. **Sources** — ask if the user knows specific channels, subreddits, or accounts to follow. \
Extract source identifiers from the user's text yourself:
   - Telegram channels: @channel or t.me/channel → store as "channel" (no @ prefix)
   - Reddit subreddits: r/sub or reddit.com/r/sub → store as "sub" (no r/ prefix)
   - Twitter/X accounts: @handle or x.com/handle → store as "handle" (no @ prefix)
   Use the validate_source_url tool to verify sources are reachable when the user provides them. \
Build the full URL for validation: https://t.me/s/channel, https://www.reddit.com/r/sub/new/, \
https://x.com/handle.
6. **Source discovery** — if user provided sources, ask whether to also discover additional \
sources automatically, or use only theirs.
7. **Format** — default to "brief summary" unless the user specifies preferences.
8. **Event matching mode** — use "strict_with_prefilter" if the user includes exclusions \
or exact requirements ("only", "not", "except"). Otherwise "basic".

Behavior rules:
- Be concise. Ask at most ONE question per turn.
- Provide choices (in the `choices` field) when there are clear options (e.g. yes/no, \
digest/event). This helps frontends render buttons.
- Respond in the same language as the user.
- If the user provides enough information in a single message to finalize, do so immediately.
- When all information is gathered, set status to "ready" and populate finalized_config.
- Accommodate mid-conversation changes (e.g. "actually make it weekly").
- For short_label: ultra-short 2-3 word category name in the user's language.
- For prompt_summary: concise 3-8 word description in the user's language.

{context_section}\
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
) -> str:
    parts: list[str] = []
    if user_language:
        parts.append(f"User's preferred language: {user_language}.")
    if user_timezone:
        parts.append(f"User's timezone: {user_timezone}.")
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
) -> tuple[AgentTurnOutput, list[dict]]:
    """Run a single conversation turn with the subscription parser.

    Returns the agent output and a list of new messages to append to the conversation
    history (tool calls, tool results, and the final assistant message).
    """
    system_msg: dict = {
        "role": "system",
        "content": _build_system_prompt(user_language, user_timezone),
    }
    new_messages: list[dict] = []

    for _ in range(_MAX_TOOL_ROUNDS + 1):
        response = await _client.beta.chat.completions.parse(
            model=settings.llm_model,
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
