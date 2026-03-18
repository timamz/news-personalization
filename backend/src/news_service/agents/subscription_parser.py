"""Conversational subscription parser agent using OpenAI Agents SDK."""

import logging

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from pydantic import BaseModel, Field

from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.agents.parser import parse_schedule_preference
from news_service.core.openai_client import agents_model
from news_service.schemas.conversation import AgentTurnOutput
from news_service.services.reddit import extract_reddit_subreddits as _extract_reddit
from news_service.services.telegram import extract_telegram_channels as _extract_tg
from news_service.services.twitter import extract_twitter_accounts as _extract_twitter

logger = logging.getLogger(__name__)


class ValidateCronResult(BaseModel):
    schedule_cron: str = Field(..., description="Parsed 5-field cron expression")
    human_readable: str = Field(..., description="Human-readable schedule description")


class ParsedSources(BaseModel):
    telegram_channels: list[str] = Field(default_factory=list)
    reddit_subreddits: list[str] = Field(default_factory=list)
    twitter_accounts: list[str] = Field(default_factory=list)


class SourceValidationResult(BaseModel):
    url: str = Field(...)
    source_kind: str = Field(...)
    is_valid: bool = Field(...)


@function_tool
async def validate_cron(schedule_text: str) -> str:
    """Parse a natural-language schedule preference into a cron expression.

    Args:
        schedule_text: Natural language schedule like "every morning at 8am" or "twice a day".
    """
    try:
        cron = await parse_schedule_preference(schedule_text)
        return f"Parsed schedule: cron={cron}"
    except Exception as exc:
        return f"Failed to parse schedule: {exc}"


@function_tool
async def parse_sources_from_text(text: str) -> str:
    """Extract Telegram channels, Reddit subreddits, and Twitter/X accounts from user text.

    Args:
        text: User text that may contain @channels, r/subreddits, or x.com/accounts.
    """
    channels = _extract_tg(text)
    subreddits = _extract_reddit(text)
    accounts = _extract_twitter(text)

    parts: list[str] = []
    if channels:
        parts.append(f"Telegram channels: {', '.join('@' + c for c in channels)}")
    if subreddits:
        parts.append(f"Reddit subreddits: {', '.join('r/' + s for s in subreddits)}")
    if accounts:
        parts.append(f"Twitter/X accounts: {', '.join('@' + a for a in accounts)}")

    if not parts:
        return "No sources found in the text."
    return "\n".join(parts)


@function_tool
async def validate_source_url(url: str, source_kind: str) -> str:
    """Check if a source URL is reachable and has content.

    Args:
        url: The source URL to validate.
        source_kind: One of: rss, telegram_channel, reddit_subreddit, twitter_account.
    """
    try:
        is_valid = await _validate_source_url(url, source_kind=source_kind)  # type: ignore[arg-type]
        if is_valid:
            return f"Source {url} ({source_kind}): valid and has content."
        return f"Source {url} ({source_kind}): could not fetch content or is empty."
    except Exception as exc:
        return f"Source {url} ({source_kind}): validation error: {exc}"


SUBSCRIPTION_PARSER_PROMPT = """\
You are a subscription setup assistant. Your job is to help the user configure a news \
subscription by gathering all required information through a brief conversation.

You must determine the following:
1. **Topic/interest** — always present in the initial message.
2. **Delivery mode** — "digest" (periodic summary) or "event" (instant notification about \
upcoming events like releases, premieres, concerts). Default to "digest" unless the user \
clearly wants event notifications.
3. **Schedule** (digest only) — if not specified, ask. Offer "manual only" as an option. \
Use the validate_cron tool to parse schedule text.
4. **Language** — detect from the user's message language. Use the context preference if \
provided. Only ask if ambiguous (e.g. user writes in English about Russian-language content).
5. **Sources** — ask if the user knows specific channels, subreddits, or accounts to follow. \
Use parse_sources_from_text to extract them from the user's answer. \
Use validate_source_url to verify sources the user provides.
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


def _build_context_section(
    user_language: str | None,
    user_timezone: str | None,
) -> str:
    parts: list[str] = []
    if user_language:
        parts.append(f"User's preferred language: {user_language}.")
    if user_timezone:
        parts.append(f"User's timezone: {user_timezone}.")
    if parts:
        return "Context:\n" + "\n".join(parts) + "\n"
    return ""


def _create_subscription_parser_agent(
    user_language: str | None = None,
    user_timezone: str | None = None,
) -> Agent[None]:
    context_section = _build_context_section(user_language, user_timezone)
    instructions = SUBSCRIPTION_PARSER_PROMPT.format(context_section=context_section)

    return Agent(
        name="subscription_parser",
        instructions=instructions,
        tools=[validate_cron, parse_sources_from_text, validate_source_url],
        model=agents_model,
        output_type=AgentTurnOutput,
        model_settings=ModelSettings(temperature=0.2),
    )


async def run_conversation_turn(
    messages: list[dict[str, str]],
    *,
    user_language: str | None = None,
    user_timezone: str | None = None,
) -> AgentTurnOutput:
    """Run a single conversation turn with the subscription parser agent.

    The full message history is passed so the agent sees the entire conversation.
    """
    agent = _create_subscription_parser_agent(
        user_language=user_language,
        user_timezone=user_timezone,
    )

    # Convert messages to the format expected by the Agent SDK:
    # The input is either a string or list of dicts with role/content
    input_messages: list[dict[str, str]] = []
    for msg in messages:
        input_messages.append({"role": msg["role"], "content": msg["content"]})

    result = await Runner.run(
        agent,
        input=input_messages,
        run_config=RunConfig(tracing_disabled=True),
        max_turns=5,
    )
    return result.final_output  # type: ignore[return-value]
