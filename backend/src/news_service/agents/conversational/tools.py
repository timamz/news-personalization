"""Tool closures for the conversational agent.

Each tool is built by ``build_tools`` as a closure over the current turn's
user, shared_state dict, DB session, and session factory. Tools that mutate
open their own scoped sessions via the factory so the model can emit
parallel tool calls safely.

The module imports external dependencies (embed_text, ensure_source_coverage,
resolve_timezone, _validate_source_url) directly so tests can patch them at
``news_service.agents.conversational.tools.<name>``.
"""

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from news_service.agents.conversational.helpers import (
    _append_conversation_summary,
    _clean_identifier,
    _parse_csv_identifiers,
)
from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.db.vector_store import embed_text
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.services.coverage import ensure_source_coverage
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.timezones import resolve_timezone
from news_service.services.twitter import build_twitter_account_url
from news_service.tasks.celery_app import celery_app
from news_service.tasks.discover_sources import DISCOVER_SOURCES_TASK

logger = logging.getLogger(__name__)


def _enqueue_discovery(subscription_id: uuid.UUID, reason: str) -> None:
    """Fire-and-forget enqueue of the discovery Celery task."""
    celery_app.send_task(DISCOVER_SOURCES_TASK, args=[str(subscription_id), reason])
    logger.info(
        "Queued source discovery for subscription %s (reason=%r)",
        subscription_id,
        reason[:100],
    )


MAX_USER_SPEC_LENGTH = 10_000

_URL_BUILDERS: dict[str, Callable[[str], str]] = {
    "telegram_channel": build_telegram_channel_url,
    "reddit_subreddit": build_reddit_subreddit_url,
    "twitter_account": build_twitter_account_url,
}


def build_tools(
    *,
    user: User,
    db_session: AsyncSession,
    scoped_factory: async_sessionmaker[AsyncSession],
    shared_state: dict[str, Any],
) -> list[Callable[..., Any]]:
    """Return the ordered list of ADK tool callables for this turn."""

    async def create_subscription(
        user_spec: str,
        retrieval_query: str,
        delivery_mode: str = "digest",
        schedule_cron: str = "",
        digest_language: str = "",
        fixed_telegram_channels: str = "",
        fixed_reddit_subreddits: str = "",
        fixed_twitter_accounts: str = "",
        include_discovered_sources: bool = True,
    ) -> str:
        """Create a brand-new subscription.

        ``user_spec`` is a freeform markdown document you author that
        captures everything LLM-facing about this subscription: what
        the user wants to follow, how they want it presented, what to
        avoid, tone, length, any useful context. Downstream agents
        (digest writer, source discovery, event assessor) read it
        verbatim as the single source of truth. You decide the
        structure; use headings, bullets, prose -- whatever best
        conveys the intent.

        ``retrieval_query`` is a SEPARATE short string used only to
        find relevant news via embedding similarity. Write it as a
        dense description of WHAT news to look for: topic, named
        entities, angles, regions, adjacent terms that would appear
        in matching headlines. DO NOT include formatting instructions
        (length, bullets, tone), exclusions ("skip X"), or delivery
        preferences -- those shape presentation, not retrieval, and
        will only pollute the vector. Aim for one sentence or a
        comma-separated phrase list.

        Args:
            user_spec: Freeform markdown describing what the user wants.
                Required and non-empty.
            retrieval_query: Short dense query for embedding-based news
                retrieval. Required and non-empty. Topic and entities
                only -- no format or tone guidance.
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
        spec = user_spec.strip()
        if not spec:
            return "user_spec is required to create a subscription."
        if len(spec) > MAX_USER_SPEC_LENGTH:
            spec = spec[:MAX_USER_SPEC_LENGTH]

        query = retrieval_query.strip()
        if not query:
            return "retrieval_query is required to create a subscription."

        resolved_language = (digest_language or user.language or "en").strip().lower()
        normalized_cron = schedule_cron.strip() or None

        telegram = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_telegram_channels)]
        reddit = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_reddit_subreddits)]
        twitter = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_twitter_accounts)]

        async with scoped_factory() as scoped:
            try:
                query_embedding = await embed_text(query)
            except Exception as exc:
                logger.exception("create_subscription: retrieval_query embedding failed")
                return f"could not embed retrieval_query: {exc}."

            subscription = Subscription(
                user_id=user.id,
                topic_embedding=query_embedding,
                user_spec=spec,
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

            if include_discovered_sources:
                _enqueue_discovery(
                    subscription.id,
                    f"Initial discovery on subscription creation. Retrieval query: {query}.",
                )

            return (
                f"subscription {subscription.id}: created (auto-discovery "
                f"{'queued' if include_discovered_sources else 'skipped'})."
            )

    async def update_subscription(
        subscription_id: str,
        user_spec: str = "",
        retrieval_query: str = "",
        delivery_mode: str = "",
        schedule_cron: str = "",
        digest_language: str = "",
    ) -> str:
        """Edit an existing subscription's scalar fields and/or user_spec.

        Any parameter left empty is preserved. To change how the
        subscription is interpreted, pass a new full ``user_spec`` --
        it overwrites the existing one. Read the current spec via
        ``get_subscriptions`` first so your rewrite only changes what
        the user actually asked to change.

        Pass ``retrieval_query`` only when the set of news worth
        surfacing actually shifts (new topic, new entities, different
        angle). A pure format change ("make digests shorter") does
        NOT need a new retrieval_query -- retrieval intent is
        unchanged. When you do pass it, the vector is re-embedded.

        Source changes go through ``add_source`` / ``remove_source`` --
        this tool does not touch the source list.

        Args:
            subscription_id: UUID of the subscription to update.
            user_spec: New full markdown spec. Empty preserves.
            retrieval_query: New retrieval anchor (topic + entities,
                no formatting). Empty preserves the existing embedding.
            delivery_mode: 'digest' or 'event'. Empty preserves.
            schedule_cron: 5-field cron. Empty preserves (cannot clear via this tool).
            digest_language: ISO code. Empty preserves.

        Returns:
            Confirmation or an error message.
        """
        try:
            sub_uuid = uuid.UUID(subscription_id.strip())
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        new_spec = user_spec.strip()
        if len(new_spec) > MAX_USER_SPEC_LENGTH:
            new_spec = new_spec[:MAX_USER_SPEC_LENGTH]
        new_query = retrieval_query.strip()

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

            if new_spec and new_spec != (existing.user_spec or ""):
                existing.user_spec = new_spec

            if new_query:
                try:
                    existing.topic_embedding = await embed_text(new_query)
                except Exception as exc:
                    logger.exception("update_subscription: retrieval_query embedding failed")
                    await scoped.rollback()
                    return f"could not re-embed retrieval_query: {exc}."

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
        if source_kind not in _URL_BUILDERS:
            return f"{identifier}: unsupported source_kind '{source_kind}'."

        cleaned = _clean_identifier(identifier)
        if not cleaned:
            return f"{identifier}: empty identifier."

        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"{cleaned}: invalid subscription_id."

        try:
            url = _URL_BUILDERS[source_kind](cleaned)
        except ValueError as exc:
            return f"{cleaned}: invalid identifier ({exc})."
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
        if source_kind not in _URL_BUILDERS:
            return f"{identifier}: unsupported source_kind '{source_kind}'."

        cleaned = _clean_identifier(identifier)
        if not cleaned:
            return f"{identifier}: empty identifier."

        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"{cleaned}: invalid subscription_id."

        try:
            url = _URL_BUILDERS[source_kind](cleaned)
        except ValueError as exc:
            return f"{cleaned}: invalid identifier ({exc})."

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

    async def trigger_source_discovery(subscription_id: str, reason: str) -> str:
        """Queue an asynchronous source-discovery run for a subscription.

        Use this when the set of news worth surfacing has meaningfully
        shifted (user just rewrote the spec toward a new topic, existing
        sources are stale, user explicitly asked for more sources) or on
        initial creation. Discovery runs in the background against the
        subscription's current user_spec + retrieval embedding; accepted
        sources are persisted as auto-discovered.

        Removing stale sources is a separate concern -- call
        ``remove_source`` first (after confirming with the user) and then
        trigger discovery so the replacement search happens against an
        honest inventory.

        Args:
            subscription_id: UUID of the subscription to discover for.
            reason: Short freeform paragraph (1-3 sentences) explaining why
                discovery is needed now. Be specific: what the user
                changed, what the old focus was, what the new focus is,
                any preferences to honour (language, paywall, academic
                vs consumer). The discovery agent reads this verbatim to
                shape its strategies.

        Returns:
            Confirmation or error message.
        """
        try:
            sub_uuid = uuid.UUID(subscription_id.strip())
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        cleaned_reason = reason.strip()
        if not cleaned_reason:
            return "reason is required to trigger source discovery."

        async with scoped_factory() as scoped:
            result = await scoped.execute(
                select(Subscription).where(
                    Subscription.id == sub_uuid,
                    Subscription.user_id == user.id,
                )
            )
            sub = result.scalar_one_or_none()
            if sub is None:
                return f"subscription {subscription_id}: not found."
            if not sub.is_active:
                return f"subscription {subscription_id}: inactive."

        _enqueue_discovery(sub_uuid, cleaned_reason)
        return f"Source discovery queued for subscription {subscription_id}."

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

    return [
        create_subscription,
        update_subscription,
        get_subscriptions,
        remember,
        add_source,
        remove_source,
        set_user_language,
        set_user_timezone,
        trigger_digest_now,
        trigger_source_discovery,
        delete_subscription,
        close_scenario,
    ]
