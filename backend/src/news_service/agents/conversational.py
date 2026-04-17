"""Conversational agent -- the single chat surface for the user.

One ADK agent handles everything: greeting, help, subscription create / edit,
source management, language + timezone setup, digest triggering, deletion.
The agent is respawned fresh each turn; durable state lives in Postgres
(``User``, ``Subscription``) and Redis (conversation transcript, persistent
per user, no conversation ids).

Memory model:

- ``messages`` in Redis is the hot transcript the agent sees verbatim.
- ``compacted_log`` is an append-only list of scenario summaries written by
  ``close_scenario`` when a logical task (onboarding, create subscription,
  edit, add/remove sources, delete, timezone setup) finishes. Rendered
  into the system prompt so the agent keeps continuity after the hot
  messages are trimmed.
- A byte-size guardrail on the hot transcript trims the oldest entries
  if the agent forgets to close a scenario.

Tools come in five groups:

- Subscription lifecycle: ``create_subscription``,
  ``update_subscription``, ``delete_subscription``,
  ``trigger_digest_now``.
- Source attach/detach: ``add_source``, ``remove_source``.
- User state: ``set_user_language``, ``set_user_timezone``.
- Memory / awareness: ``get_subscriptions``, ``remember``.
- Conversation flow: ``close_scenario`` (mark a task finished so prior
  messages can be compacted).

Every mutation tool opens its own DB session via ``session_factory`` so the
model may emit parallel tool calls safely.
"""

import asyncio
import hashlib
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
from news_service.db.vector_store import embed_text
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.models.user_spec import (
    UserSpecSections,
    extract_topic,
    parse_user_spec,
    render_user_spec,
)
from news_service.schemas.conversation import AgentTurnOutput
from news_service.services.coverage import ensure_source_coverage
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.timezones import resolve_timezone
from news_service.services.twitter import build_twitter_account_url

logger = logging.getLogger(__name__)
settings = get_settings()

_CONVERSATION_SUMMARY_BYTE_LIMIT = 2048


CONVERSATIONAL_AGENT_PROMPT = """\
You are a friendly personal news assistant. You are the user's ONLY interface -- \
there is no menu, no buttons, no other UI. Every interaction flows through this chat.

You do three things:
1. Explain the service and answer questions about how it works.
2. Create, edit, and manage news subscriptions.
3. Take direct actions: add/remove sources, trigger deliveries, delete subscriptions, \
set language and timezone.

Language policy:
- Respond in the same language as the user's most recent message.
- On the very first turn, detect the language and immediately call set_user_language \
with the ISO code. Never ask which language they want.
- If the user switches language mid-chat, follow them and update via set_user_language.

Greeting new users (no subscriptions yet):
- One short message: friendly greeting + one sentence about what you do + one concrete \
example, ending with a single question ("What would you like to follow?"). ~3 sentences.
- Do not dump features.

Returning users:
- Skip the intro. Answer the request directly.
- "Hi" / "what can you do" -> 1-2 examples tailored to what they already have, then \
one forward-looking question.

Subscription creation via create_subscription:
- Gather topic, delivery mode (digest vs event -- default digest), schedule, sources, \
and any presentation preferences (length, format, exclusions, tone). When you have \
enough, call create_subscription.
- Pass format and exclusion guidance via the 'preferences' argument, not a separate \
field. Example: preferences="three short bullets, skip stock prices, include quotes".
- Convert schedule text to a 5-field cron internally. Never show cron to the user.
  "every morning" -> "0 8 * * *", "every evening at 9pm" -> "0 21 * * *",
  "every Saturday morning" -> "0 8 * * 6", "every third day" -> "0 8 */3 * *",
  "every hour" -> "0 * * * *", "every weekday at 9" -> "0 9 * * 1-5",
  "twice a day at 8 and 18" -> "0 8,18 * * *". Empty schedule = manual / event mode.
- Source identifiers (no prefix): Telegram "channel" (not @channel), Reddit "sub" \
(not r/sub), X "handle" (not @handle).
- If the user provided sources, ask whether to also auto-discover more.

Editing existing subscriptions via update_subscription:
- Use get_subscriptions when you need the full user_spec of a sub. The pre-loaded \
one-line summaries in context are enough for disambiguation ("the AI one") but not \
for editing details.
- To change scalar fields (schedule, language, delivery mode) or rewrite the \
topic/preferences, call update_subscription with the subscription_id and only the \
fields you want to change. Empty parameters preserve existing values. Changing the \
topic re-embeds it automatically so retrieval follows.
- For sources on an existing subscription, use add_source / remove_source (not \
update_subscription).

Parallel tool calls:
- If the user mentions multiple sources to add or remove in one message, emit the \
add_source / remove_source calls in parallel in the same turn. Each is independent \
and safe to run concurrently.

Timezone handling:
- When a scheduled digest is requested and no timezone is set, ask "what city are \
you in?" (in the user's language).
- Pass the reply to set_user_timezone. On "resolved" confirm briefly; on "ambiguous" \
list candidates and ask which one; on "not_found" ask for a larger nearby city.
- Raw offsets like "UTC+3" work too.

Memory:
- When the user tells you a durable fact about themselves or a preference that should \
outlive this conversation (they travel often, they prefer short digests, they mute \
weekends, they speak only Russian with family, etc.), call remember with one short \
sentence. Do not remember transient things ("I'm tired today").

Conversation flow (close_scenario):
- This is ONE persistent chat with the user. There is no session reset: old messages \
stay in context until you actively compact them.
- You know the shape of every task the user can complete. Call close_scenario when \
one clearly finishes so the transcript stays small and focused on what is active now.
- Scenarios and their terminal signal:
  - onboarding: user_language + set_user_timezone set + first create_subscription.
  - create subscription: create_subscription succeeded and user acknowledged.
  - edit subscription: update_subscription succeeded.
  - add sources: add_source calls returned successfully for everything the user asked for.
  - remove sources: remove_source calls returned successfully.
  - delete subscription: delete_subscription succeeded.
  - trigger digest: trigger_digest_now succeeded.
  - one-off Q&A: after you answered an informational question with no pending follow-up.
  - user cancelled mid-flow ("never mind", "forget it"): close with an abort summary.
- Do not close a scenario if something is still pending (you are still gathering info, \
the user has not confirmed, a follow-up question is on the table).
- The summary you pass to close_scenario must be ONE short sentence, factual, past tense: \
"created AI digest daily 8am", "updated schedule on AI sub to 9am", "added @bbcworld to \
news sub", "user cancelled football digest setup".

Help / questions:
- "How does this work?", "digest vs event?", "what sources?" -- answer inline in \
2-4 sentences using concrete examples. Do not call tools.

General behavior:
- Be friendly and concise. At most ONE question per turn.
- No buttons, no structured choices. Everything is text.
- Never show cron expressions, UUIDs, or internal field names to the user.
- If the user provides enough info in one message, act immediately.
- Accommodate mid-conversation changes.
- When the user gives feedback about digest quality, call update_subscription with \
an updated 'preferences' string that captures what they want.

{context_section}\
"""


def _build_instruction(
    conversation_summary: str,
    user_language: str | None,
    user_timezone: str | None,
    conversation_history: list[dict] | None = None,
    subscription_summaries: list[str] | None = None,
    compacted_log: list[str] | None = None,
    has_onboarded: bool = False,
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
    if not has_onboarded:
        parts.append(
            "This user has never completed onboarding. Treat this as a first-time "
            "interaction and follow the greeting rules above."
        )
    if subscription_summaries:
        parts.append(
            "Active subscriptions for this user:\n"
            + "\n".join(f"- {line}" for line in subscription_summaries)
        )
    elif has_onboarded:
        parts.append(
            "Active subscriptions: none right now. This is a returning user "
            "(already onboarded) who has no active subscriptions at the moment -- "
            "do NOT show the first-time greeting. Answer directly."
        )
    if conversation_summary:
        parts.append(f"What you already know about this user:\n{conversation_summary}")
    if compacted_log:
        parts.append(
            "Earlier in this chat (already-closed scenarios, compacted):\n"
            + "\n".join(f"- {line}" for line in compacted_log)
        )
    if conversation_history:
        history_lines: list[str] = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                history_lines.append(f"{role.capitalize()}: {content}")
        if history_lines:
            parts.append("Recent messages (hot transcript):\n" + "\n".join(history_lines))
    context_section = ""
    if parts:
        context_section = "Context:\n" + "\n\n".join(parts) + "\n"
    return CONVERSATIONAL_AGENT_PROMPT.format(context_section=context_section)


async def _load_subscription_summaries(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> list[str]:
    """Fetch compact one-line descriptions of the user's active subscriptions.

    An empty list signals a first-time interaction. The agent calls
    get_subscriptions when it needs the full spec of a specific one.
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
        topic = extract_topic(sub.user_spec or "") or "(no topic)"
        schedule = sub.schedule_cron or (
            "event mode" if sub.delivery_mode == "event" else "on demand"
        )
        lines.append(f"[{sub.id}] {sub.delivery_mode} | {schedule} | {topic}")
    return lines


def _parse_csv_identifiers(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _clean_identifier(identifier: str) -> str:
    cleaned = identifier.strip().lstrip("@").lstrip("#")
    if cleaned.startswith("r/"):
        cleaned = cleaned[2:]
    return cleaned


def _source_display_name(url: str, source_kind: str) -> str:
    """Extract a user-friendly name from a source URL (for status messages)."""
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


def _append_conversation_summary(existing: str, fact: str) -> str:
    """Dedup by content hash and cap at ~2KB. Evicts oldest lines when over.

    Entries are prefixed with ISO timestamp so boundary flush + remember can
    coexist peacefully in one text field.
    """
    fact = fact.strip()
    if not fact:
        return existing
    lines = [line for line in existing.split("\n") if line.strip()]
    fact_hash = hashlib.sha1(fact.lower().encode("utf-8")).hexdigest()[:8]
    tagged = f"{datetime.now(UTC).strftime('%Y-%m-%d')} [{fact_hash}] {fact}"
    lines = [line for line in lines if f"[{fact_hash}]" not in line]
    lines.append(tagged)
    serialized = "\n".join(lines)
    while len(serialized.encode("utf-8")) > _CONVERSATION_SUMMARY_BYTE_LIMIT and len(lines) > 1:
        lines.pop(0)
        serialized = "\n".join(lines)
    return serialized


def create_conversational_agent(
    *,
    db_session: AsyncSession,
    user: User,
    conversation_summary: str,
    user_language: str | None = None,
    conversation_history: list[dict] | None = None,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    subscription_summaries: list[str] | None = None,
    compacted_log: list[str] | None = None,
) -> tuple[Agent, dict[str, Any]]:
    """Build a fresh ADK agent bound to this turn's DB session and user.

    Returns the agent and a shared_state dict. All mutation tools open their
    own DB session via session_factory so the model can emit parallel tool
    calls safely. ``shared_state['scenario_close_summary']`` is set when
    the agent calls close_scenario; the caller uses that signal to compact
    the hot transcript after the turn.
    """
    scoped_factory = session_factory or async_session_factory
    shared_state: dict[str, Any] = {
        "status": "in_progress",
        "created_subscription_id": None,
        "discovery_triggered": False,
        "scenario_close_summary": None,
    }

    async def create_subscription(
        topic: str,
        preferences: str = "",
        delivery_mode: str = "digest",
        schedule_cron: str = "",
        digest_language: str = "",
        fixed_telegram_channels: str = "",
        fixed_reddit_subreddits: str = "",
        fixed_twitter_accounts: str = "",
        include_discovered_sources: bool = True,
    ) -> str:
        """Create a brand-new subscription.

        Renders ``user_spec`` from ``topic`` + ``preferences`` (single
        source of truth for LLM-facing intent), embeds the topic for
        retrieval, attaches fixed sources, optionally queues auto-
        discovery, and flips the user's ``has_onboarded`` on first
        success. For edits to an existing subscription call
        ``update_subscription`` instead.

        Args:
            topic: Subscription topic. Required.
            preferences: Freeform guidance ('brief summary', 'skip stock
                prices', 'three bullets max', 'include quotes'). Goes
                into user_spec's Preferences section.
            delivery_mode: 'digest' (periodic summary) or 'event' (instant alerts).
            schedule_cron: 5-field cron. Empty = manual / event-only delivery.
            digest_language: ISO code (en, ru, ...). Empty = use the user's language.
            fixed_telegram_channels: Comma-separated handles (no @).
            fixed_reddit_subreddits: Comma-separated sub names (no r/).
            fixed_twitter_accounts: Comma-separated X handles (no @).
            include_discovered_sources: Run auto-discovery for more sources.

        Returns:
            Confirmation with the subscription id, or an error message.
        """
        clean_topic = topic.strip()
        if not clean_topic:
            return "topic is required to create a subscription."

        resolved_language = (digest_language or user.language or "en").strip().lower()
        normalized_cron = schedule_cron.strip() or None
        preferences_text = preferences.strip()

        telegram = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_telegram_channels)]
        reddit = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_reddit_subreddits)]
        twitter = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_twitter_accounts)]

        async with scoped_factory() as scoped:
            try:
                topic_embedding = await embed_text(clean_topic)
            except Exception as exc:
                logger.exception("create_subscription: topic embedding failed")
                return f"could not embed topic: {exc}."

            subscription = Subscription(
                user_id=user.id,
                topic_embedding=topic_embedding,
                user_spec=render_user_spec(
                    UserSpecSections(topic=clean_topic, preferences=preferences_text)
                ),
                delivery_mode=delivery_mode,
                schedule_cron=normalized_cron,
                digest_language=resolved_language,
            )
            scoped.add(subscription)
            await scoped.flush()

            selected: dict[uuid.UUID, Source] = {}
            user_specified_ids: set[uuid.UUID] = set()
            for identifiers, kind in [
                (telegram, "telegram_channel"),
                (reddit, "reddit_subreddit"),
                (twitter, "twitter_account"),
            ]:
                if not identifiers:
                    continue
                try:
                    coverage = await ensure_source_coverage(scoped, identifiers, kind)
                except Exception as exc:
                    logger.exception("create_subscription: coverage failed for %s", kind)
                    await scoped.rollback()
                    return f"could not register {kind} sources: {exc}."
                for source in coverage:
                    selected[source.id] = source
                    user_specified_ids.add(source.id)

            if include_discovered_sources:
                shared_state["discovery_triggered"] = True
                shared_state["discovery_subscription_id"] = str(subscription.id)

            for source_id in selected:
                scoped.add(
                    SubscriptionSource(
                        subscription_id=subscription.id,
                        source_id=source_id,
                        is_user_specified=source_id in user_specified_ids,
                    )
                )
            try:
                await scoped.commit()
            except Exception as exc:
                logger.exception("create_subscription: commit failed")
                await scoped.rollback()
                return f"could not save subscription: {exc}."
            shared_state["created_subscription_id"] = str(subscription.id)

            if not user.has_onboarded:
                persisted_user = await scoped.get(User, user.id)
                if persisted_user is not None and not persisted_user.has_onboarded:
                    persisted_user.has_onboarded = True
                    await scoped.commit()
                user.has_onboarded = True

            return (
                f"subscription {subscription.id}: created (auto-discovery "
                f"{'queued' if include_discovered_sources else 'skipped'})."
            )

    async def update_subscription(
        subscription_id: str,
        topic: str = "",
        preferences: str = "",
        delivery_mode: str = "",
        schedule_cron: str = "",
        digest_language: str = "",
    ) -> str:
        """Edit an existing subscription's scalar fields and/or user_spec.

        Any parameter left empty is preserved. Changing ``topic``
        re-embeds the vector used for retrieval so digest candidates
        follow the new focus. Source changes go through
        ``add_source`` / ``remove_source`` -- this tool does not touch
        the source list.

        Args:
            subscription_id: UUID of the subscription to update.
            topic: New topic. Empty preserves the existing Topic section.
            preferences: New preferences. Empty preserves the existing section.
            delivery_mode: 'digest' or 'event'. Empty preserves.
            schedule_cron: 5-field cron or empty to clear the schedule.
                Omit entirely (default) to preserve the existing schedule.
            digest_language: ISO code. Empty preserves.

        Returns:
            Confirmation or an error message.
        """
        from sqlalchemy import select

        try:
            sub_uuid = uuid.UUID(subscription_id.strip())
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        clean_topic = topic.strip()
        preferences_text = preferences.strip()

        async with scoped_factory() as scoped:
            result = await scoped.execute(
                select(Subscription).where(
                    Subscription.id == sub_uuid,
                    Subscription.user_id == user.id,
                )
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return f"subscription {subscription_id}: not found."

            if delivery_mode.strip():
                existing.delivery_mode = delivery_mode.strip()
            if schedule_cron.strip():
                existing.schedule_cron = schedule_cron.strip()
            if digest_language.strip():
                existing.digest_language = digest_language.strip().lower()

            if clean_topic or preferences_text:
                current = parse_user_spec(existing.user_spec) if existing.user_spec else None
                new_topic = clean_topic or (current.topic if current else "")
                new_preferences = preferences_text or (current.preferences if current else "")
                existing.user_spec = render_user_spec(
                    UserSpecSections(topic=new_topic, preferences=new_preferences)
                )
                if clean_topic and (current is None or current.topic != clean_topic):
                    try:
                        existing.topic_embedding = await embed_text(clean_topic)
                    except Exception as exc:
                        logger.exception("update_subscription: topic embedding failed")
                        await scoped.rollback()
                        return f"could not re-embed topic: {exc}."

            await scoped.commit()
        return f"subscription {subscription_id}: updated."

    async def get_subscriptions() -> str:
        """Return every active subscription the user has, with full user_spec.

        Use this when you need the details (schedule, format, sources,
        preferences) of a specific subscription to respond accurately to an
        edit request. The one-line summaries in the turn context are enough
        to identify which sub the user means; call this to see everything.

        Returns:
            A formatted listing with topic, mode, schedule, language, format,
            sources, and the full user_spec markdown per subscription.
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
        blocks: list[str] = []
        for sub in subs:
            link_result = await db_session.execute(
                select(Source.url, SubscriptionSource.is_user_specified)
                .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
                .where(SubscriptionSource.subscription_id == sub.id)
            )
            source_lines = [
                f"  - {row.url} {'[user]' if row.is_user_specified else '[auto]'}"
                for row in link_result.all()
            ]
            schedule = sub.schedule_cron or (
                "event mode" if sub.delivery_mode == "event" else "on demand"
            )
            block = (
                f"[{sub.id}] {sub.delivery_mode} | schedule={schedule} | "
                f"language={sub.digest_language}\n"
                f"sources:\n{chr(10).join(source_lines) if source_lines else '  (none)'}\n"
                f"user_spec:\n{sub.user_spec}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    async def remember(fact: str) -> str:
        """Persist a durable fact about the user across conversations.

        Use sparingly for things worth surviving Redis TTL: travel habits,
        strong preferences, constraints ("I work nights"), language quirks,
        relationships. Skip transient moods ("I'm tired today") or anything
        already in a subscription's user_spec.

        Args:
            fact: One short sentence describing the durable fact.

        Returns:
            Confirmation.
        """
        cleaned = fact.strip()
        if not cleaned:
            return "empty fact; nothing remembered."
        from sqlalchemy import select

        async with scoped_factory() as scoped:
            result = await scoped.execute(select(User).where(User.id == user.id))
            persisted = result.scalar_one_or_none()
            if persisted is None:
                return "user not found."
            updated = _append_conversation_summary(persisted.conversation_summary or "", cleaned)
            persisted.conversation_summary = updated
            await scoped.commit()
        return "remembered."

    async def add_source(
        subscription_id: str,
        identifier: str,
        source_kind: str,
    ) -> str:
        """Attach a source to an existing subscription.

        Validates reachability, upserts the source, links it as user-specified.
        Safe to call in parallel for multiple sources in one turn.

        Args:
            subscription_id: UUID of the subscription to modify.
            identifier: Handle / name with no prefix (channel, sub, handle).
            source_kind: telegram_channel | reddit_subreddit | twitter_account.

        Returns:
            Short confirmation or per-source error.
        """
        from sqlalchemy import select

        url_builders = {
            "telegram_channel": build_telegram_channel_url,
            "reddit_subreddit": build_reddit_subreddit_url,
            "twitter_account": build_twitter_account_url,
        }
        if source_kind not in url_builders:
            return f"{identifier}: unsupported source_kind '{source_kind}'."

        cleaned = _clean_identifier(identifier)
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

        async with scoped_factory() as scoped:
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
        """Detach a source from a subscription and log the removal.

        Safe to call in parallel.

        Args:
            subscription_id: UUID of the subscription to modify.
            identifier: Handle / name with no prefix.
            source_kind: telegram_channel | reddit_subreddit | twitter_account.

        Returns:
            Short confirmation or error.
        """
        from sqlalchemy import select

        url_builders = {
            "telegram_channel": build_telegram_channel_url,
            "reddit_subreddit": build_reddit_subreddit_url,
            "twitter_account": build_twitter_account_url,
        }
        if source_kind not in url_builders:
            return f"{identifier}: unsupported source_kind '{source_kind}'."

        cleaned = _clean_identifier(identifier)
        if not cleaned:
            return f"{identifier}: empty identifier."

        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"{cleaned}: invalid subscription_id."

        url = url_builders[source_kind](cleaned)

        async with scoped_factory() as scoped:
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
        """Persist the user's preferred language (ISO code).

        Call immediately after detecting the language of the first message,
        without asking. Also call it when the user switches language.

        Args:
            code: ISO 639-1 or short BCP-47 code (en, ru, de, es, ...).

        Returns:
            Confirmation.
        """
        from sqlalchemy import select

        normalized = code.strip().lower().split("-", maxsplit=1)[0]
        if len(normalized) < 2 or len(normalized) > 16:
            return f"Invalid language code '{code}'."

        async with scoped_factory() as scoped:
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

        Inspect the returned status:
          - 'resolved' -- auto-set, confirm briefly.
          - 'ambiguous' -- list candidates to the user, ask which one.
          - 'not_found' -- ask for a larger nearby city.

        Args:
            query: Free-text location ('Berlin', 'Paris France', 'UTC+3', ...).

        Returns:
            'status: details' for the agent to parse.
        """
        from sqlalchemy import select

        cleaned = query.strip()
        if not cleaned:
            return "not_found: empty query."

        resolution = resolve_timezone(cleaned)
        if resolution.status == "resolved" and resolution.candidates:
            candidate = resolution.candidates[0]
            async with scoped_factory() as scoped:
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

    async def trigger_digest_now(subscription_id: str) -> str:
        """Queue an immediate digest delivery for one subscription.

        Args:
            subscription_id: UUID of the subscription to deliver.

        Returns:
            Confirmation that the digest is queued.
        """
        from news_service.tasks.deliver_digest import deliver_digest

        deliver_digest.delay(subscription_id, notify_if_empty=True)
        return f"Digest queued for delivery (subscription {subscription_id})."

    async def delete_subscription(subscription_id: str) -> str:
        """Soft-delete (deactivate) a subscription by id.

        Safe to call concurrently with other mutation tools -- opens
        its own scoped session like every other mutation tool.
        Confirm in plain language with the user before calling.

        Args:
            subscription_id: UUID of the subscription to deactivate.

        Returns:
            Confirmation or not-found message.
        """
        from sqlalchemy import select

        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        async with scoped_factory() as scoped:
            result = await scoped.execute(
                select(Subscription).where(
                    Subscription.id == sub_uuid,
                    Subscription.user_id == user.id,
                )
            )
            sub = result.scalar_one_or_none()
            if sub is None:
                return f"Subscription {subscription_id} not found."
            sub.is_active = False
            await scoped.commit()
        return f"Subscription {subscription_id} deleted."

    async def close_scenario(summary: str) -> str:
        """Mark the current logical task as finished so prior messages compact.

        Call this once per turn when a scenario terminates cleanly (see the
        scenario list in the system instruction). The backend then moves the
        hot transcript up to this turn into compacted_log, keeping only the
        very latest exchange live. Do not call if something is still pending.

        Args:
            summary: One short, factual, past-tense sentence describing the
                outcome. Examples: "created AI digest daily 8am", "updated
                schedule on football sub to weekends only", "user cancelled
                onboarding".

        Returns:
            Confirmation (or a note that the summary was empty).
        """
        cleaned = summary.strip()
        if not cleaned:
            return "empty summary; nothing closed."
        shared_state["scenario_close_summary"] = cleaned[:200]
        return "scenario closed."

    effective_language = user_language or user.language

    instruction = _build_instruction(
        conversation_summary=conversation_summary,
        user_language=effective_language,
        user_timezone=user.timezone,
        conversation_history=conversation_history,
        subscription_summaries=subscription_summaries,
        compacted_log=compacted_log,
        has_onboarded=bool(getattr(user, "has_onboarded", False)),
    )

    agent = Agent(
        name="conversational_agent",
        model=LiteLlm(model=settings.litellm_model),
        instruction=instruction,
        tools=[
            create_subscription,
            update_subscription,
            get_subscriptions,
            remember,
            add_source,
            remove_source,
            set_user_language,
            set_user_timezone,
            trigger_digest_now,
            delete_subscription,
            close_scenario,
        ],
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    _ = status_queue  # reserved for future per-tool UI events; not used by tools.
    return agent, shared_state


async def run_conversational_turn(
    *,
    db_session: AsyncSession,
    user: User,
    user_message: str,
    conversation_summary: str,
    user_language: str | None = None,
) -> dict[str, Any]:
    """Run a single non-streaming turn and return a simple result dict.

    Used by tests and non-streaming callers.
    """
    agent, shared_state = create_conversational_agent(
        db_session=db_session,
        user=user,
        conversation_summary=conversation_summary,
        user_language=user_language,
    )
    agent_message = await run_agent_text(
        agent=agent,
        message=user_message,
        user_id=str(user.id),
    )
    return {
        "agent_message": agent_message,
        "created_subscription_id": shared_state["created_subscription_id"],
        "discovery_triggered": shared_state["discovery_triggered"],
    }


async def run_conversation_turn_streaming(
    messages: list[dict],
    *,
    db_session: AsyncSession,
    user: User,
    conversation_summary: str,
    user_language: str | None = None,
    compacted_log: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming variant: yields status events, then one final done event.

    The done event carries the agent's shared_state so the caller can
    react to ``scenario_close_summary`` and compact the hot transcript.

    Events:
      {"event": "status", "status_key": ..., ...kwargs}
      {"event": "done", "output": {...}, "new_messages": [...], "shared_state": {...}}
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
        conversation_summary=conversation_summary,
        user_language=user_language,
        conversation_history=previous_messages,
        status_queue=status_queue,
        subscription_summaries=subscription_summaries,
        compacted_log=compacted_log,
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
                emitted = _status_for_tool_call(event)
                if emitted is not None:
                    yield emitted
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
    )
    yield {
        "event": "done",
        "output": output.model_dump(),
        "new_messages": [{"role": "assistant", "content": agent_text}],
        "shared_state": {
            "scenario_close_summary": shared_state.get("scenario_close_summary"),
        },
    }


def _status_for_tool_call(event: dict[str, Any]) -> dict[str, Any] | None:
    """Map an ADK tool_call event to a status message for the UI, or None."""
    tool_name = event.get("name", "")
    args = event.get("args", {})
    if tool_name == "add_source":
        return {
            "event": "status",
            "status_key": "status_adding_source",
            "source": args.get("identifier", ""),
            "source_kind": args.get("source_kind", ""),
        }
    if tool_name == "remove_source":
        return {
            "event": "status",
            "status_key": "status_removing_source",
            "source": args.get("identifier", ""),
            "source_kind": args.get("source_kind", ""),
        }
    if tool_name == "set_user_timezone":
        return {
            "event": "status",
            "status_key": "status_resolving_timezone",
            "query": args.get("query", ""),
        }
    if tool_name in {"create_subscription", "update_subscription"}:
        return {
            "event": "status",
            "status_key": "status_saving_subscription",
            "subscription_id": args.get("subscription_id", ""),
        }
    if tool_name == "trigger_digest_now":
        return {
            "event": "status",
            "status_key": "status_queuing_digest",
            "subscription_id": args.get("subscription_id", ""),
        }
    return None
