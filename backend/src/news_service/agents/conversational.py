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
from datetime import UTC, datetime
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from news_service.agents.adk_runner import run_agent, run_agent_text
from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.db.session import async_session_factory
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.models.user_spec import extract_topic, validate_user_spec
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ExistingSubscriptionContext,
    FinalizedSubscriptionConfig,
)
from news_service.services.coverage import ensure_source_coverage
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.timezones import resolve_timezone
from news_service.services.twitter import build_twitter_account_url

logger = logging.getLogger(__name__)
settings = get_settings()

CONVERSATIONAL_AGENT_PROMPT = """\
You are a friendly personal news assistant. You are the user's ONLY interface -- \
there is no menu, no buttons, no other UI. Every interaction flows through this chat.

You help the user with three jobs:
1. Explain the service and answer questions about how it works.
2. Create, edit, and manage news subscriptions.
3. Take direct actions on their behalf: add/remove sources, trigger deliveries, \
delete subscriptions, set language and timezone.

Language policy:
- ALWAYS respond in the same language as the user's most recent message.
- If no language is persisted yet, detect it from the first message and immediately \
call set_user_language(code) with the ISO code (e.g. 'en', 'ru', 'de', 'es'). Do NOT \
ask which language they want -- detect silently.
- If the user switches language mid-conversation, follow them and update with \
set_user_language.

Greeting new users (no subscriptions yet and no language set):
- Start with ONE short message: a friendly greeting in the detected language, one \
sentence about what you do, and one concrete example (e.g. "I can send you a daily \
digest about AI research, or a notification the moment your favorite artist announces \
a concert"). End with a single question: "What would you like to follow?"
- Do not dump a feature list. Keep it to ~3 sentences total.

Returning users:
- Skip the intro. Answer the request directly.
- If the user just says hi or "what can you do?", respond briefly with 1-2 concrete \
examples tailored to what they already have subscribed to, and ask what they want to \
do next.

Subscription creation -- gather:
1. **Topic** -- what they want to follow.
2. **Delivery mode** -- periodic summary ("digest") or instant alerts about specific \
events like releases, premieres, concerts ("event"). Default to digest unless they \
clearly want event alerts.
3. **Schedule** (digest only) -- ask when they want updates ("every morning", \
"twice a day"). Offer on-demand-only as an option. Convert internally to a 5-field \
cron. Never show cron to the user.
   - "every morning" -> "0 8 * * *"
   - "every evening at 9pm" -> "0 21 * * *"
   - "every Saturday morning" -> "0 8 * * 6"
   - "every third day" -> "0 8 */3 * *"
   - "every hour" -> "0 * * * *"
   - "every weekday at 9" -> "0 9 * * 1-5"
   - "twice a day at 8 and 18" -> "0 8,18 * * *"
4. **Language** -- use the persisted user language by default. Only ask if the user \
clearly wants content in a different language than they're writing in.
5. **Sources** -- ask if they already follow specific Telegram channels, Reddit \
communities, or X accounts. Identifier formats:
   - Telegram: @channel or t.me/channel -> "channel" (no @)
   - Reddit: r/sub or reddit.com/r/sub -> "sub" (no r/)
   - Twitter/X: x.com/handle -> "handle" (no @)
   Use validate_source to check reachability. Build full URLs for validation: \
https://t.me/s/channel, https://www.reddit.com/r/sub/new/, https://x.com/handle.
6. **Auto-discovery** -- if the user gave sources, ask whether to also find more \
automatically, or stick with only theirs.
7. **Format** -- default to "brief summary" unless they specify otherwise.

Timezone handling:
- When the user requests a scheduled digest and has no timezone set, ask "what city \
are you in?" (in their language).
- Pass whatever they say to set_user_timezone. Inspect the returned status:
  - "resolved" -- confirm briefly (e.g. "Got it -- using Moscow time").
  - "ambiguous" -- list the candidates (city, country) and ask which one.
  - "not_found" -- ask for a larger nearby city.
- If the user says a raw offset like "UTC+3", still pass it; the resolver handles it.

Managing existing subscriptions:
- Use list_subscriptions to see what the user has. When the user refers to one fuzzily \
("the AI one", "my tech digest"), match by topic keywords; if two could match, ask ONE \
disambiguating question.
- add_source / remove_source attach or detach sources on a specific subscription. \
Both take identifier + source_kind (no URL, no prefix). Multiple sources in one \
message? Emit the calls in parallel in a single turn; each is independent.
- trigger_digest_now for "send me the AI digest now"-style requests.
- delete_subscription for explicit deletion; confirm once in plain language before \
calling.

Help and questions:
- "How does this work?" "What kinds of sources?" "Digest vs event?" -- answer inline \
in 2-4 sentences. Do NOT call tools. Use concrete examples. Avoid feature lists.
- If they ask why a digest was empty / late, explain briefly and offer to tune the \
subscription (topic too narrow, sources stale, etc.).

Behavior rules:
- Be friendly and concise. Ask at most ONE question per turn.
- No buttons, no structured choices -- everything is text.
- Never show cron expressions, UUIDs, or internal field names.
- If the user provides enough info in a single message, act immediately.
- Accommodate mid-conversation changes ("actually make it weekly").
- When you modify subscription preferences, always call update_user_spec with the \
full updated spec.
- When subscription setup info is complete, call finalize_subscription, then \
confirm in a single friendly sentence.
- When the user gives feedback about digest quality, update user_spec.

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


async def _load_subscription_summaries(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> list[str]:
    """Fetch compact one-line descriptions of the user's active subscriptions.

    Feeds the agent's instruction so it knows what the user already has without
    needing to call list_subscriptions on trivial questions. An empty list
    signals a first-time interaction.
    """
    from sqlalchemy import select

    result = await session.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.is_active.is_(True),
        )
    )
    subs = list(result.scalars().all())
    lines: list[str] = []
    for sub in subs:
        topic = extract_topic(sub.user_spec or sub.raw_prompt or "") or (
            (sub.user_spec or sub.raw_prompt or "").strip().splitlines()[0][:80]
            if (sub.user_spec or sub.raw_prompt)
            else "(no topic)"
        )
        schedule = sub.schedule_cron or (
            "event mode" if sub.delivery_mode == "event" else "on demand"
        )
        lines.append(f"[{sub.id}] {sub.delivery_mode} | {schedule} | {topic}")
    return lines


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
    subscription_summaries: list[str] | None = None,
) -> str:
    parts: list[str] = []
    persisted_bits: list[str] = []
    if user_language:
        persisted_bits.append(f"language={user_language}")
    if user_timezone:
        persisted_bits.append(f"timezone={user_timezone}")
    parts.append(
        "Persisted user preferences: "
        + (", ".join(persisted_bits) if persisted_bits else "none yet")
        + "."
    )
    if subscription_summaries is not None:
        if subscription_summaries:
            parts.append(
                "Active subscriptions for this user:\n"
                + "\n".join(f"- {line}" for line in subscription_summaries)
            )
        else:
            parts.append(
                "Active subscriptions: none. Treat this as a first-time interaction "
                "and follow the greeting rules above."
            )
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
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    subscription_summaries: list[str] | None = None,
) -> tuple[Agent, dict[str, Any]]:
    """Create a conversational agent with tools bound to the current DB session.

    Returns the agent and a shared state dict that tools write side effects to.

    ``session_factory`` provides a dedicated session per tool call that mutates
    persistent state (add_source, remove_source, set_user_language,
    set_user_timezone). AsyncSession is not safe for concurrent operations, so
    the request-scoped ``db_session`` is reserved for read-only or sequential
    tool work. Defaults to the app session factory; tests override it.

    ``subscription_summaries`` are compact one-line strings describing each of
    the user's active subscriptions. When passed as an empty list, the agent
    treats the interaction as first-time and applies the greeting rules. Pass
    ``None`` to skip the section entirely (e.g. for legacy flows).
    """
    source_factory = session_factory or async_session_factory
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

    async def add_source(
        subscription_id: str,
        identifier: str,
        source_kind: str,
    ) -> str:
        """Attach a source to an existing subscription.

        Validates the source is reachable, upserts it into the sources table,
        and links it to the subscription as user-specified. Use one call per
        source; multiple calls in the same turn run in parallel safely.

        Args:
            subscription_id: UUID of the subscription to modify.
            identifier: Source identifier with no prefix
                (channel name for Telegram, subreddit for Reddit, handle for X).
            source_kind: One of: telegram_channel, reddit_subreddit, twitter_account.

        Returns:
            Short confirmation or an error message per source.
        """
        url_builders = {
            "telegram_channel": build_telegram_channel_url,
            "reddit_subreddit": build_reddit_subreddit_url,
            "twitter_account": build_twitter_account_url,
        }
        if source_kind not in url_builders:
            return f"{identifier}: unsupported source_kind '{source_kind}'."

        cleaned = identifier.strip().lstrip("@").lstrip("#")
        if cleaned.startswith("r/"):
            cleaned = cleaned[2:]
        if not cleaned:
            return f"{identifier}: empty identifier."

        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"{cleaned}: invalid subscription_id."

        url = url_builders[source_kind](cleaned)

        try:
            is_valid = await _validate_source_url(url, source_kind=source_kind)
        except Exception as exc:
            return f"{cleaned}: validation error: {exc}"
        if not is_valid:
            return f"{cleaned}: unreachable or empty."

        from sqlalchemy import select

        async with source_factory() as scoped:
            sub_result = await scoped.execute(
                select(Subscription).where(
                    Subscription.id == sub_uuid,
                    Subscription.user_id == user.id,
                )
            )
            subscription = sub_result.scalar_one_or_none()
            if subscription is None:
                return f"{cleaned}: subscription not found."
            if not subscription.is_active:
                return f"{cleaned}: subscription is inactive."

            existing = await scoped.execute(
                select(SubscriptionSource)
                .join(Source, Source.id == SubscriptionSource.source_id)
                .where(
                    SubscriptionSource.subscription_id == sub_uuid,
                    Source.url == url,
                )
            )
            if existing.scalar_one_or_none() is not None:
                return f"{cleaned}: already attached to this subscription."

            try:
                sources = await ensure_source_coverage(scoped, [cleaned], source_kind)
            except Exception as exc:
                logger.exception("add_source: coverage upsert failed for %s", url)
                return f"{cleaned}: could not register source ({exc})."
            if not sources:
                return f"{cleaned}: could not register source."

            scoped.add(
                SubscriptionSource(
                    subscription_id=sub_uuid,
                    source_id=sources[0].id,
                    is_user_specified=True,
                )
            )
            try:
                await scoped.commit()
            except Exception as exc:
                logger.exception("add_source: commit failed for %s", url)
                await scoped.rollback()
                return f"{cleaned}: could not attach source ({exc})."

        return f"{cleaned} ({source_kind}): added."

    async def remove_source(
        subscription_id: str,
        identifier: str,
        source_kind: str,
    ) -> str:
        """Detach a source from an existing subscription.

        Removes the link between the subscription and the source, decrements
        the source's subscriber count (and deactivates it if it reaches zero),
        and records the removal in the source_removal_log. Each call runs in
        its own DB session so parallel removals are safe.

        Args:
            subscription_id: UUID of the subscription to modify.
            identifier: Source identifier with no prefix
                (channel for Telegram, subreddit for Reddit, handle for X).
            source_kind: One of: telegram_channel, reddit_subreddit, twitter_account.

        Returns:
            Short confirmation or an error message.
        """
        url_builders = {
            "telegram_channel": build_telegram_channel_url,
            "reddit_subreddit": build_reddit_subreddit_url,
            "twitter_account": build_twitter_account_url,
        }
        if source_kind not in url_builders:
            return f"{identifier}: unsupported source_kind '{source_kind}'."

        cleaned = identifier.strip().lstrip("@").lstrip("#")
        if cleaned.startswith("r/"):
            cleaned = cleaned[2:]
        if not cleaned:
            return f"{identifier}: empty identifier."

        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"{cleaned}: invalid subscription_id."

        url = url_builders[source_kind](cleaned)

        from sqlalchemy import select

        async with source_factory() as scoped:
            sub_result = await scoped.execute(
                select(Subscription).where(
                    Subscription.id == sub_uuid,
                    Subscription.user_id == user.id,
                )
            )
            subscription = sub_result.scalar_one_or_none()
            if subscription is None:
                return f"{cleaned}: subscription not found."

            link_result = await scoped.execute(
                select(SubscriptionSource)
                .join(Source, Source.id == SubscriptionSource.source_id)
                .where(
                    SubscriptionSource.subscription_id == sub_uuid,
                    Source.url == url,
                )
            )
            link = link_result.scalar_one_or_none()
            if link is None:
                return f"{cleaned}: not attached to this subscription."

            source_result = await scoped.execute(select(Source).where(Source.id == link.source_id))
            source = source_result.scalar_one()

            await scoped.delete(link)
            source.subscriber_count = max(source.subscriber_count - 1, 0)
            if source.subscriber_count == 0:
                source.is_active = False

            scoped.add(
                SourceRemovalLog(
                    subscription_id=sub_uuid,
                    source_url=url,
                    removed_at=datetime.now(UTC),
                    removal_reason="user request",
                )
            )
            try:
                await scoped.commit()
            except Exception as exc:
                logger.exception("remove_source: commit failed for %s", url)
                await scoped.rollback()
                return f"{cleaned}: could not detach source ({exc})."

        return f"{cleaned} ({source_kind}): removed."

    async def set_user_language(code: str) -> str:
        """Persist the user's preferred language.

        Call this immediately after detecting the language of the user's first
        message, without asking. Also call it when the user switches language
        mid-conversation.

        Args:
            code: ISO 639-1 or BCP-47 short code (e.g. 'en', 'ru', 'de', 'es').

        Returns:
            Confirmation that the language was persisted.
        """
        normalized = code.strip().lower().split("-", maxsplit=1)[0]
        if len(normalized) < 2 or len(normalized) > 16:
            return f"Invalid language code '{code}'."

        from sqlalchemy import select

        async with source_factory() as scoped:
            result = await scoped.execute(select(User).where(User.id == user.id))
            persisted_user = result.scalar_one_or_none()
            if persisted_user is None:
                return "User not found."
            persisted_user.language = normalized
            await scoped.commit()

        user.language = normalized
        return f"Language set to {normalized}."

    async def set_user_timezone(query: str) -> str:
        """Resolve a free-text location to an IANA timezone and persist it.

        Pass whatever the user said (a city, country + city, or raw zone like
        'UTC+3' or 'Europe/Moscow'). Inspect the returned status:
        - 'resolved' -- timezone was set automatically.
        - 'ambiguous' -- list the candidates to the user and ask which one.
        - 'not_found' -- ask for a larger nearby city.

        Args:
            query: Free-text location, e.g. 'Berlin', 'Paris France', 'Europe/Moscow'.

        Returns:
            A line of the form 'status: details' for the agent to read.
        """
        cleaned = query.strip()
        if not cleaned:
            return "not_found: empty query."

        resolution = resolve_timezone(cleaned)
        if resolution.status == "resolved" and resolution.candidates:
            candidate = resolution.candidates[0]
            from sqlalchemy import select

            async with source_factory() as scoped:
                result = await scoped.execute(select(User).where(User.id == user.id))
                persisted_user = result.scalar_one_or_none()
                if persisted_user is None:
                    return "not_found: user missing."
                persisted_user.timezone = candidate.timezone
                await scoped.commit()
            user.timezone = candidate.timezone
            local = candidate.local_time().strftime("%H:%M")
            return f"resolved: {candidate.label} -> {candidate.timezone} (local time {local})."

        if resolution.status == "ambiguous":
            listing = " | ".join(f"{c.label} [{c.timezone}]" for c in resolution.candidates)
            return f"ambiguous: {listing}. Ask the user which one."

        return f"not_found: no match for '{cleaned}'. Ask the user for a larger city."

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

    effective_language = user_language or user.language

    instruction = _build_instruction(
        user_spec=user_spec,
        conversation_summary=conversation_summary,
        user_language=effective_language,
        user_timezone=user.timezone,
        conversation_history=conversation_history,
        existing_config=existing_config,
        subscription_summaries=subscription_summaries,
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
            add_source,
            remove_source,
            set_user_language,
            set_user_timezone,
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

    try:
        subscription_summaries = await _load_subscription_summaries(db_session, user.id)
    except Exception:
        logger.exception("Failed to load subscription summaries; continuing without context")
        subscription_summaries = None

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
        subscription_summaries=subscription_summaries,
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
                elif tool_name == "add_source":
                    args = event.get("args", {})
                    identifier = args.get("identifier", "")
                    source_kind = args.get("source_kind", "")
                    yield {
                        "event": "status",
                        "status_key": "status_adding_source",
                        "source": identifier,
                        "source_kind": source_kind,
                    }
                elif tool_name == "remove_source":
                    args = event.get("args", {})
                    identifier = args.get("identifier", "")
                    source_kind = args.get("source_kind", "")
                    yield {
                        "event": "status",
                        "status_key": "status_removing_source",
                        "source": identifier,
                        "source_kind": source_kind,
                    }
                elif tool_name == "set_user_timezone":
                    args = event.get("args", {})
                    yield {
                        "event": "status",
                        "status_key": "status_resolving_timezone",
                        "query": args.get("query", ""),
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
