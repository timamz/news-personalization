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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from news_service.agents.conversational.helpers import (
    _append_conversation_summary,
    _clean_identifier,
    _parse_csv_identifiers,
)
from news_service.agents.discovery import validate_source_url as _validate_source_url
from news_service.core.config import get_settings
from news_service.core.confirmations import consume as consume_pending
from news_service.core.confirmations import create as create_pending
from news_service.core.rate_limit import RateLimitExceeded, check_rate_limit
from news_service.core.subscription_shares import consume as consume_share
from news_service.core.subscription_shares import create as create_share
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
from news_service.tasks.celery_app import celery_app
from news_service.tasks.discover_sources import DISCOVER_SOURCES_TASK

logger = logging.getLogger(__name__)


def _dispatch_discovery(subscription_id: uuid.UUID, reason: str) -> str:
    """Fire-and-forget enqueue of source discovery on a Celery worker.

    The conversational turn returns immediately after the subscription
    row is saved; the Celery task picks up the work, runs the full
    Discovery pipeline in its own process, and posts a follow-up
    notification to the user's webhook when it finishes. This keeps
    the HTTP conversation turn short (seconds, not minutes) and keeps
    the FastAPI DB connection pool free of the discovery run's
    session, which used to pin a connection for the entire multi-
    minute Discovery loop.

    Returns the short string the tool reports back to the LLM. The
    LLM surfaces this to the user verbatim inside its reply.
    """
    celery_app.send_task(
        DISCOVER_SOURCES_TASK,
        args=[str(subscription_id), reason],
    )
    logger.info("Queued source discovery for subscription %s", subscription_id)
    return "discovery_queued"


MAX_USER_SPEC_LENGTH = 10_000

_URL_BUILDERS: dict[str, Callable[[str], str]] = {
    "telegram_channel": build_telegram_channel_url,
    "reddit_subreddit": build_reddit_subreddit_url,
}


async def _gate_with_confirmation(
    *,
    user: User,
    shared_state: dict[str, Any],
    confirmation_token: str,
    tool_name: str,
    args: dict[str, Any],
    description: str,
    yes_label: str,
    no_label: str,
) -> tuple[bool, str]:
    """Server-side confirmation gate for destructive / expensive tools.

    First call (empty ``confirmation_token``): mint a nonce in Redis,
    push a ``requires_confirmation`` event onto the conversation stream,
    return the REQUIRES_CONFIRMATION marker string for the LLM. The
    frontend renders the event as inline yes/no buttons; the user's
    tap travels back via the ``/conversations/confirm`` endpoint, which
    re-invokes the same tool with the nonce.

    Second call (token supplied): atomically consume the nonce from
    Redis. If it is missing, expired, owned by a different user, or
    points at a different tool/args, refuse. The LLM cannot fabricate
    a valid nonce because nonces are crypto-random and never enter
    the LLM context.

    Returns ``(proceed, message_for_caller)``. When ``proceed`` is False
    the tool must return ``message_for_caller`` verbatim.
    """
    if not confirmation_token:
        nonce = await create_pending(
            user_id=str(user.id),
            tool_name=tool_name,
            args=args,
            description=description,
        )
        queue = shared_state.get("status_queue")
        if queue is not None:
            await queue.put(
                {
                    "event": "requires_confirmation",
                    "nonce": nonce,
                    "action": tool_name,
                    "description": description,
                    "yes_label": yes_label,
                    "no_label": no_label,
                }
            )
        return False, (
            f"REQUIRES_CONFIRMATION: about to {description}. The system has "
            "rendered yes/no buttons to the user. In your reply, restate "
            "what is about to happen in one short sentence in the user's "
            "language and tell them to use the buttons below. Do NOT call "
            "this tool again from text input -- the system will invoke it "
            "via the button callback once the user taps Yes."
        )

    pending = await consume_pending(confirmation_token, str(user.id))
    if pending is None:
        return False, (
            "confirmation_invalid: the confirmation token is expired, unknown, "
            "or owned by a different user. Ask the user to retry the action."
        )
    if pending.tool_name != tool_name or pending.args != args:
        return False, (
            "confirmation_mismatch: the confirmation token does not match "
            "this action's arguments. Ask the user to retry."
        )
    return True, ""


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
        title: str = "",
        delivery_mode: str = "digest",
        schedule_cron: str = "",
        digest_language: str = "",
        fixed_telegram_channels: str = "",
        fixed_reddit_subreddits: str = "",
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
            title: Short human-readable label for this subscription,
                1-5 words, in the user's language. Used in the
                subscription list so the user can identify it at a
                glance. Example: "Бадминтон", "AI дайджест", "Аниме".
            delivery_mode: 'digest' (periodic summary) or 'event' (instant alerts).
            schedule_cron: 5-field cron. Empty = manual / event-only delivery.
            digest_language: ISO code (en, ru, ...). Empty = use the user's language.
            fixed_telegram_channels: Comma-separated handles (no @).
            fixed_reddit_subreddits: Comma-separated sub names (no r/).
            include_discovered_sources: Leave True (the default) in almost
                every case. A subscription with zero attached sources has
                NO content to read, so auto-discovery is how the
                subscription gets populated. Set False ONLY when the user
                has explicitly listed every single source they want and
                said something like "use only these, nothing else". If
                the user did not enumerate specific sources, True. If the
                user said "build it" / "create it" / "set it up" with no
                source list, True. If the user provided a partial list
                and hasn't said "only these", True. Default True; flipping
                to False is a narrow opt-out for power users.

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

        active_limit = get_settings().max_active_subscriptions_per_user
        active_count_result = await db_session.execute(
            select(func.count())
            .select_from(Subscription)
            .where(
                Subscription.user_id == user.id,
                Subscription.is_active.is_(True),
                Subscription.paused_at.is_(None),
            )
        )
        active_count = int(active_count_result.scalar_one() or 0)
        if active_count >= active_limit:
            return (
                f"subscription limit reached: this user already has {active_count} "
                f"running subscriptions, and the maximum allowed is {active_limit}. "
                "Stopped subscriptions do NOT count toward this cap, only running "
                "ones. Do NOT create another one. Tell the user plainly in their "
                f"language that they have hit the limit of {active_limit} running "
                "subscriptions, that they need to either stop one (via "
                "stop_subscription) or delete one (via delete_subscription) before "
                "a new one can be created, and offer to list their current "
                "subscriptions so they can pick which to remove."
            )

        resolved_language = (digest_language or user.language or "en").strip().lower()
        normalized_cron = schedule_cron.strip() or None

        telegram = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_telegram_channels)]
        reddit = [_clean_identifier(s) for s in _parse_csv_identifiers(fixed_reddit_subreddits)]

        async with scoped_factory() as scoped:
            try:
                query_embedding = await embed_text(query, timeout=30.0)
            except Exception as exc:
                logger.exception("create_subscription: retrieval_query embedding failed")
                return f"could not embed retrieval_query: {exc}."

            subscription = Subscription(
                user_id=user.id,
                topic_embedding=query_embedding,
                user_spec=spec,
                title=title.strip()[:120] or None,
                delivery_mode=delivery_mode,
                schedule_cron=normalized_cron,
                digest_language=resolved_language,
                delivery_webhook_url=user.delivery_webhook_url,
            )
            scoped.add(subscription)
            await scoped.flush()

            selected: dict[uuid.UUID, Source] = {}
            user_specified_ids: set[uuid.UUID] = set()
            for identifiers, kind in [
                (telegram, "telegram_channel"),
                (reddit, "reddit_subreddit"),
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

            subscription_id = subscription.id

        if not include_discovered_sources:
            return f"subscription {subscription_id}: created (auto-discovery skipped)."

        return f"subscription {subscription_id}: created.\n" + _dispatch_discovery(
            subscription_id,
            f"Initial discovery on subscription creation. Retrieval query: {query}.",
        )

    async def update_subscription(
        subscription_id: str,
        user_spec: str = "",
        retrieval_query: str = "",
        title: str = "",
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

            if title.strip():
                existing.title = title.strip()[:120]
            if new_spec and new_spec != (existing.user_spec or ""):
                existing.user_spec = new_spec

            if new_query:
                try:
                    existing.topic_embedding = await embed_text(new_query, timeout=30.0)
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
            state = "STOPPED" if sub.paused_at is not None else "RUNNING"
            block = (
                f"[{sub.id}] state={state} | {sub.delivery_mode} | schedule={schedule} | "
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
            source_kind: telegram_channel | reddit_subreddit.

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
        confirmation_token: str = "",
    ) -> str:
        """Detach a source from a subscription and log the removal.

        Destructive: gated by server-side confirmation. First call mints
        a nonce and emits a ``requires_confirmation`` event; the frontend
        renders inline yes/no buttons; the actual detach fires only when
        the confirm endpoint re-invokes this tool with the nonce.

        Args:
            subscription_id: UUID of the subscription to modify.
            identifier: Handle / name with no prefix.
            source_kind: telegram_channel | reddit_subreddit.
            confirmation_token: Set by the confirm endpoint; leave empty
                in normal LLM calls.

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

        proceed, message = await _gate_with_confirmation(
            user=user,
            shared_state=shared_state,
            confirmation_token=confirmation_token,
            tool_name="remove_source",
            args={
                "subscription_id": subscription_id,
                "identifier": identifier,
                "source_kind": source_kind,
            },
            description=(
                f"detach the {source_kind} source '{cleaned}' from subscription {subscription_id}"
            ),
            yes_label="Remove",
            no_label="Keep",
        )
        if not proceed:
            return message

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

    async def trigger_digest_now(subscription_id: str, confirmation_token: str = "") -> str:
        """Queue an immediate digest delivery for one subscription.

        Gated by server-side confirmation. First call mints a nonce
        and pushes a ``requires_confirmation`` event onto the stream;
        the frontend renders inline yes/no buttons. The actual send
        only fires when the confirm endpoint re-invokes this tool
        with the nonce as ``confirmation_token``.

        Args:
            subscription_id: UUID of the subscription to deliver.
            confirmation_token: Set by the confirm endpoint; leave
                empty in normal LLM calls. The LLM cannot fabricate a
                valid value.

        Returns:
            Confirmation that the digest is queued or an error.
        """
        proceed, message = await _gate_with_confirmation(
            user=user,
            shared_state=shared_state,
            confirmation_token=confirmation_token,
            tool_name="trigger_digest_now",
            args={"subscription_id": subscription_id},
            description=f"trigger an immediate digest send for subscription {subscription_id}",
            yes_label="Send digest",
            no_label="Cancel",
        )
        if not proceed:
            return message

        try:
            await check_rate_limit(
                scope="trigger_digest_now",
                subject_id=str(user.id),
                limit=get_settings().rate_limit_digest_now_per_day,
                window_seconds=86400,
            )
        except RateLimitExceeded as exc:
            return (
                f"rate_limit_exceeded: you've hit the daily cap of "
                f"{exc.limit} immediate-digest requests. Try again in "
                f"{exc.retry_after_seconds // 60} minutes."
            )

        from news_service.tasks.deliver_digest import deliver_digest

        deliver_digest.delay(subscription_id, notify_if_empty=True)
        return f"Digest queued for delivery (subscription {subscription_id})."

    async def trigger_source_discovery(
        subscription_id: str, reason: str, confirmation_token: str = ""
    ) -> str:
        """Run source discovery inline for a subscription and return the findings.

        Discovery spends real money (LLM rounds, web searches), so this
        tool is gated by server-side confirmation. First call mints a
        nonce and emits a ``requires_confirmation`` event; the frontend
        renders inline yes/no buttons; the actual discovery runs only
        when the confirm endpoint re-invokes this tool with the nonce.

        Use this when the set of news worth surfacing has meaningfully
        shifted (user just rewrote the spec toward a new topic, existing
        sources are stale, user explicitly asked for more sources).
        Discovery runs inside the current turn (live progress is streamed
        to the user); when this tool returns, the new sources are already
        saved and the return string lists them so the reply can be
        specific about what was added.

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
            confirmation_token: Set by the confirm endpoint; leave empty
                in normal LLM calls.

        Returns:
            A summary of what discovery found, or an error message.
        """
        try:
            sub_uuid = uuid.UUID(subscription_id.strip())
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        cleaned_reason = reason.strip()
        if not cleaned_reason:
            return "reason is required to trigger source discovery."

        proceed, message = await _gate_with_confirmation(
            user=user,
            shared_state=shared_state,
            confirmation_token=confirmation_token,
            tool_name="trigger_source_discovery",
            args={"subscription_id": subscription_id, "reason": cleaned_reason},
            description=(
                f"run source discovery for subscription {subscription_id} "
                f"(spends LLM + search credits); reason: {cleaned_reason[:120]}"
            ),
            yes_label="Run discovery",
            no_label="Cancel",
        )
        if not proceed:
            return message

        try:
            await check_rate_limit(
                scope="trigger_source_discovery",
                subject_id=str(user.id),
                limit=get_settings().rate_limit_discovery_per_day,
                window_seconds=86400,
            )
        except RateLimitExceeded as exc:
            return (
                f"rate_limit_exceeded: you've hit the daily cap of "
                f"{exc.limit} source-discovery runs. Try again in "
                f"{exc.retry_after_seconds // 60} minutes."
            )

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

        return _dispatch_discovery(sub_uuid, cleaned_reason)

    async def delete_subscription(subscription_id: str, confirmation_token: str = "") -> str:
        """Soft-delete (deactivate) a subscription by id.

        Destructive: gated by server-side confirmation. First call mints
        a nonce and emits a ``requires_confirmation`` event; the frontend
        renders inline yes/no buttons; the actual delete fires only when
        the confirm endpoint re-invokes this tool with the nonce.

        Args:
            subscription_id: UUID of the subscription to deactivate.
            confirmation_token: Set by the confirm endpoint; leave empty
                in normal LLM calls.

        Returns:
            Confirmation or not-found message.
        """
        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        proceed, message = await _gate_with_confirmation(
            user=user,
            shared_state=shared_state,
            confirmation_token=confirmation_token,
            tool_name="delete_subscription",
            args={"subscription_id": subscription_id},
            description=f"deactivate (soft-delete) subscription {subscription_id}",
            yes_label="Delete",
            no_label="Keep",
        )
        if not proceed:
            return message

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

    async def stop_subscription(subscription_id: str, confirmation_token: str = "") -> str:
        """Stop (pause) a subscription without deleting it.

        A stopped subscription keeps all its metadata (sources, user_spec,
        schedule, cron) but is skipped by every polling, scheduling, and
        delivery pipeline until it is resumed. Stopped subscriptions do
        NOT count toward the active-subscription cap, so the user can
        free a slot without losing the configuration. Use this when the
        user wants a temporary break (vacation, signal fatigue, "mute
        for a while"); use ``delete_subscription`` only when the user
        wants the subscription gone for good.

        Destructive enough to warrant a confirmation gate: first call
        mints a nonce and emits a ``requires_confirmation`` event; the
        frontend renders inline yes/no buttons; the actual stop fires
        only when the confirm endpoint re-invokes this tool with the
        nonce.

        Args:
            subscription_id: UUID of the subscription to stop.
            confirmation_token: Set by the confirm endpoint; leave empty
                in normal LLM calls.

        Returns:
            Confirmation or an error message (not found, already
            stopped, soft-deleted).
        """
        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        proceed, message = await _gate_with_confirmation(
            user=user,
            shared_state=shared_state,
            confirmation_token=confirmation_token,
            tool_name="stop_subscription",
            args={"subscription_id": subscription_id},
            description=f"stop (pause) subscription {subscription_id}",
            yes_label="Stop",
            no_label="Keep running",
        )
        if not proceed:
            return message

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
                return (
                    f"subscription {subscription_id}: already deleted, "
                    "cannot stop it. Tell the user the subscription has "
                    "been removed already."
                )
            if sub.paused_at is not None:
                return (
                    f"subscription {subscription_id}: already stopped. "
                    "Tell the user it is already paused and offer to "
                    "resume it instead."
                )
            sub.paused_at = datetime.now(UTC)
            await scoped.commit()
        return f"subscription {subscription_id}: stopped."

    async def resume_subscription(subscription_id: str) -> str:
        """Resume a previously stopped subscription.

        Non-destructive: no confirmation gate. Refuses when the user is
        already at the running-subscription cap; in that case the agent
        must ask the user to stop or delete one of their running
        subscriptions before retrying. Stopped subscriptions do not
        count toward the cap, so resuming one with no running siblings
        always succeeds.

        Args:
            subscription_id: UUID of the subscription to resume.

        Returns:
            Confirmation or a clear error explaining what to do next.
        """
        try:
            sub_uuid = uuid.UUID(subscription_id)
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        active_limit = get_settings().max_active_subscriptions_per_user
        running_count_result = await db_session.execute(
            select(func.count())
            .select_from(Subscription)
            .where(
                Subscription.user_id == user.id,
                Subscription.is_active.is_(True),
                Subscription.paused_at.is_(None),
            )
        )
        running_count = int(running_count_result.scalar_one() or 0)
        if running_count >= active_limit:
            return (
                f"subscription limit reached: this user already has "
                f"{running_count} running subscriptions, and the maximum "
                f"allowed is {active_limit}. Do NOT resume the subscription. "
                "Tell the user in their language that they have hit the "
                f"limit of {active_limit} running subscriptions, that they "
                "need to either stop one (via stop_subscription) or delete "
                "one (via delete_subscription) before this one can be "
                "resumed, and offer to list their current subscriptions "
                "so they can pick which to stop."
            )

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
                return (
                    f"subscription {subscription_id}: deleted, cannot "
                    "resume. Tell the user the subscription has been "
                    "removed and offer to create a new one on the same "
                    "topic."
                )
            if sub.paused_at is None:
                return (
                    f"subscription {subscription_id}: already running, "
                    "nothing to resume. Tell the user it is already active."
                )
            sub.paused_at = None
            await scoped.commit()
        return f"subscription {subscription_id}: resumed."

    async def share_subscription(subscription_id: str) -> str:
        """Mint a short, opaque share token for one of the user's subscriptions.

        The token is the bearer credential: whoever pastes it into their
        own chat can import a COPY of the subscription. The token is
        valid for 7 days and is one-shot (importing consumes it). This
        tool only validates ownership and creates the token -- it does
        not modify the original subscription, so it does not need a
        confirmation gate.

        Args:
            subscription_id: UUID of the subscription to share. Must
                belong to the current user and must not be soft-deleted.

        Returns:
            A status string carrying the token verbatim. The agent MUST
            surface the token literally; downstream frontends rely on
            it being visible in the assistant reply.
        """
        try:
            sub_uuid = uuid.UUID(subscription_id.strip())
        except ValueError:
            return f"invalid subscription_id '{subscription_id}'."

        result = await db_session.execute(
            select(Subscription).where(
                Subscription.id == sub_uuid,
                Subscription.user_id == user.id,
            )
        )
        sub = result.scalar_one_or_none()
        if sub is None:
            return f"subscription {subscription_id}: not found."
        if not sub.is_active:
            return (
                f"subscription {subscription_id}: cannot share a deleted "
                "subscription. Tell the user the subscription is not "
                "active and offer to share a different one."
            )

        ttl = get_settings().subscription_share_ttl_seconds
        token = await create_share(
            owner_user_id=str(user.id),
            subscription_id=str(sub.id),
            ttl_seconds=ttl,
        )
        return (
            f"share_token_created: SHARE_TOKEN={token}. Show this token to "
            "the user verbatim and tell them it is valid for 7 days; the "
            "recipient must paste it into their own chat with this "
            "assistant to import the subscription. Do not paraphrase, "
            "translate, or shorten the token string itself."
        )

    async def import_shared_subscription(share_token: str) -> str:
        """Redeem a share token to import a COPY of someone else's subscription.

        Atomically consumes the token from Redis (single-shot). Refuses
        when the token is unknown / expired, when the importer is the
        same user that minted it, or when the importer is already at
        the active-subscription cap. On success, creates a new
        ``Subscription`` row owned by the importer, carrying over the
        user_spec, retrieval anchor, delivery mode, schedule, language,
        and the full source list -- but with the IMPORTER'S
        ``delivery_webhook_url`` so deliveries go to the importer's
        frontend, not the owner's.

        Args:
            share_token: The opaque token the owner gave the user.

        Returns:
            A short status string. On success this is treated as a
            subscription creation, so the agent should close the
            scenario via ``close_scenario`` after surfacing the result.
        """
        token = (share_token or "").strip()
        if not token:
            return "share_token is required to import a subscription."

        pending = await consume_share(token)
        if pending is None:
            return (
                "share_invalid: this share token is unknown, expired, or "
                "has already been used. Tell the user (in their language) "
                "that the link is no longer valid and ask the original "
                "owner to send a fresh one."
            )

        if pending.owner_user_id == str(user.id):
            return (
                "share_self_import: this token belongs to your own "
                "subscription, so there is nothing to import. Tell the "
                "user they already have the subscription."
            )

        active_limit = get_settings().max_active_subscriptions_per_user
        active_count_result = await db_session.execute(
            select(func.count())
            .select_from(Subscription)
            .where(
                Subscription.user_id == user.id,
                Subscription.is_active.is_(True),
            )
        )
        active_count = int(active_count_result.scalar_one() or 0)
        if active_count >= active_limit:
            return (
                f"subscription limit reached: this user already has {active_count} "
                f"active subscriptions, and the maximum allowed is {active_limit}. "
                "Do NOT import the shared subscription. Tell the user in "
                "their language that they have hit the limit of "
                f"{active_limit} active subscriptions, that they need to "
                "delete one of the existing ones (via delete_subscription) "
                "before this share can be imported, and offer to list their "
                "current subscriptions so they can pick which to remove. "
                "Note: the share token has already been spent and the "
                "owner will need to mint a new one."
            )

        try:
            source_sub_uuid = uuid.UUID(pending.subscription_id)
        except ValueError:
            return "share_invalid: stored subscription id is malformed."

        async with scoped_factory() as scoped:
            source_result = await scoped.execute(
                select(Subscription).where(Subscription.id == source_sub_uuid)
            )
            source_sub = source_result.scalar_one_or_none()
            if source_sub is None or not source_sub.is_active:
                return (
                    "share_invalid: the original subscription is no longer "
                    "available (the owner may have deleted it). Tell the "
                    "user and offer to create a new subscription on the "
                    "same topic."
                )

            new_sub = Subscription(
                user_id=user.id,
                topic_embedding=source_sub.topic_embedding,
                user_spec=source_sub.user_spec,
                delivery_mode=source_sub.delivery_mode,
                schedule_cron=source_sub.schedule_cron,
                digest_language=source_sub.digest_language,
                delivery_webhook_url=user.delivery_webhook_url,
                is_active=True,
            )
            scoped.add(new_sub)
            await scoped.flush()

            link_result = await scoped.execute(
                select(SubscriptionSource).where(
                    SubscriptionSource.subscription_id == source_sub_uuid
                )
            )
            for original_link in link_result.scalars().all():
                scoped.add(
                    SubscriptionSource(
                        subscription_id=new_sub.id,
                        source_id=original_link.source_id,
                        is_user_specified=original_link.is_user_specified,
                    )
                )

            try:
                await scoped.commit()
            except Exception as exc:
                logger.exception("import_shared_subscription: commit failed")
                await scoped.rollback()
                return f"could not import shared subscription: {exc}."

            shared_state["created_subscription_id"] = str(new_sub.id)

            if not user.has_onboarded:
                persisted_user = await scoped.get(User, user.id)
                if persisted_user is not None and not persisted_user.has_onboarded:
                    persisted_user.has_onboarded = True
                    await scoped.commit()
                user.has_onboarded = True

            new_sub_id = new_sub.id

        return (
            f"shared_subscription_imported: subscription {new_sub_id} created "
            "from the share token. Confirm to the user (in their language) "
            "that the shared subscription has been added to their account "
            "with the same topic, schedule, and sources; deliveries will "
            "arrive through their own frontend. Then close the scenario."
        )

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
        stop_subscription,
        resume_subscription,
        share_subscription,
        import_shared_subscription,
        close_scenario,
    ]


def build_tools_by_name(
    *,
    user: User,
    db_session: AsyncSession,
    scoped_factory: async_sessionmaker[AsyncSession],
    shared_state: dict[str, Any],
) -> dict[str, Callable[..., Any]]:
    """Same as ``build_tools`` but indexed by function name.

    Used by the ``/conversations/confirm`` endpoint to dispatch a
    confirmed action straight to the right tool without going through
    the ADK loop.
    """
    return {
        tool.__name__: tool
        for tool in build_tools(
            user=user,
            db_session=db_session,
            scoped_factory=scoped_factory,
            shared_state=shared_state,
        )
    }
