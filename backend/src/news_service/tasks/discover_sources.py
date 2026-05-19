"""Source discovery: shared async helper plus the Celery task wrapper.

``run_and_persist_discovery`` loads the full context the discovery agent
needs (user_spec, retrieval embedding, currently-attached sources with
their kinds and user/auto flag, recent removal history), invokes
``run_source_discovery``, and persists accepted sources as
auto-discovered ``SubscriptionSource`` rows. It is shared between two
call sites:

1. The Celery task in this module, kept for the reflector post-digest
   path (no HTTP streaming context available there).
2. The conversational agent's ``create_subscription`` and
   ``trigger_source_discovery`` tools, which run discovery INLINE inside
   the user's streaming HTTP turn and relay progress back to the
   Telegram bot for live message edits.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from news_service.agents.source_discovery import run_source_discovery
from news_service.core.config import get_settings
from news_service.core.llm_usage import subscription_tag
from news_service.core.provider_errors import ProviderLimitError
from news_service.db.session import get_task_session
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.coverage import ensure_source_by_url
from news_service.services.delivery import deliver
from news_service.tasks.celery_app import celery_app
from news_service.tasks.retry_policy import retry_on_provider_limit

settings = get_settings()

logger = logging.getLogger(__name__)

DISCOVER_SOURCES_TASK = "news_service.tasks.discover_sources.discover_sources_for_subscription"

_REMOVAL_HISTORY_LIMIT = 50


@celery_app.task(
    bind=True,
    name=DISCOVER_SOURCES_TASK,
    soft_time_limit=1200,
    time_limit=1260,
)
def discover_sources_for_subscription(self, subscription_id: str, reason: str = "") -> dict:
    """Celery entry point. Bridges to the async impl.

    The time limits guard against an ADK finder hanging inside a blocking
    ``asyncio.to_thread`` call (e.g. Selenium-based article-body fetch
    during RSS validation) that an inner ``asyncio.wait_for`` cannot
    cancel. A soft limit raises ``SoftTimeLimitExceeded`` in the Python
    code so the task can unwind cleanly; the hard limit SIGKILLs the
    worker process if the soft unwind itself gets stuck.
    """
    try:
        return asyncio.run(_discover_in_task_session(uuid.UUID(subscription_id), reason))
    except ProviderLimitError as exc:
        raise retry_on_provider_limit(self, exc) from exc


async def _discover_in_task_session(subscription_id: uuid.UUID, reason: str) -> dict:
    async with get_task_session() as session:
        result = await run_and_persist_discovery(session, subscription_id, reason)
    await _notify_user_of_discovery_outcome(subscription_id, result)
    return result


_COMPLETION_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "subject": "Source discovery finished",
        "already_attached": (
            "Found {n} source(s) for your subscription; they were already attached, "
            "so no new sources were added. Your subscription is ready."
        ),
        "attached_header": "Attached {n} new source(s) to your subscription:",
        "more_suffix": "... and {n} more.",
        "none_with_reason": "Source discovery did not find any sources. Reason: {reason}\n\n{hint}",
        "none_no_reason": "Source discovery did not find any sources.\n\n{hint}",
        "none_hint": (
            "Try broadening the topic wording, or add specific sources you trust "
            "(Telegram channels, subreddits, RSS feeds)."
        ),
        "at_capacity": (
            "Subscription is already at capacity ({attached}/{target} sources). "
            "Remove a source first if you want a different one."
        ),
    },
    "ru": {
        "subject": "Подбор источников завершён",
        "already_attached": (
            "Нашёл {n} источник(ов) для подписки; они уже были подключены, новых "
            "не добавилось. Подписка готова."
        ),
        "attached_header": "Подключил {n} новый(х) источник(ов) к подписке:",
        "more_suffix": "... и ещё {n}.",
        "none_with_reason": "Поиск источников не дал результатов. Причина: {reason}\n\n{hint}",
        "none_no_reason": "Поиск источников не дал результатов.\n\n{hint}",
        "none_hint": (
            "Попробуйте сформулировать тему шире или добавьте конкретные источники, "
            "которым доверяете (Telegram-каналы, сабреддиты, RSS-ленты)."
        ),
        "at_capacity": (
            "Подписка уже заполнена ({attached}/{target} источников). "
            "Удалите один из источников, если хотите заменить."
        ),
    },
}


def _completion_strings(language: str) -> dict[str, str]:
    """Pick the localized string bundle, falling back to English for any unknown ISO code."""
    return _COMPLETION_TEXT.get((language or "").lower(), _COMPLETION_TEXT["en"])


def _friendly_handle(url: str, kind: str) -> str:
    """Derive a recognizable bullet label from a source URL.

    The discovery agent persists ``title`` equal to the raw URL when it has
    no better name, which used to render two duplicate hyperlinks per
    bullet in the Telegram message. Showing a short handle (``@varlamov_news``,
    ``r/economy``) instead of a second URL gives the user a recognizable
    label and lets the bot collapse the URL into a single italic link.
    """
    cleaned = url.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if "t.me/" in lowered:
        tail = cleaned.split("t.me/", 1)[1].lstrip("/")
        if tail.startswith("s/"):
            tail = tail[2:]
        handle = tail.split("/", 1)[0].strip()
        if handle:
            return f"@{handle}"
    if "reddit.com/r/" in lowered:
        tail = cleaned.split("reddit.com/r/", 1)[1]
        sub = tail.split("/", 1)[0].strip()
        if sub:
            return f"r/{sub}"
    if kind == "rss":
        host = cleaned.split("://", 1)[-1].split("/", 1)[0]
        if host:
            return host
    host = cleaned.split("://", 1)[-1].split("/", 1)[0]
    return host or cleaned


def _format_completion_body(result: dict, language: str) -> str | None:
    """Compose a short user-facing message describing a discovery outcome.

    Returns ``None`` when the outcome should NOT be surfaced to the user --
    e.g. the subscription was deleted mid-run, or the pipeline was skipped
    for an internal reason the user would not understand. ``ok`` and
    ``no_sources_found`` are always surfaced because both are outcomes the
    user has to act on (start using the subscription or refine it).

    Each bullet emits exactly one URL so the tgbot HTML renderer produces
    one labeled link per source instead of two duplicated ones.
    """
    strings = _completion_strings(language)
    status = result.get("status")
    if status == "ok":
        selected = result.get("selected_sources") or []
        persisted = int(result.get("persisted", 0))
        if persisted == 0 and selected:
            return strings["already_attached"].format(n=len(selected))
        lines = [strings["attached_header"].format(n=persisted)]
        for src in selected[:20]:
            url = (src.get("url") or "").strip()
            if not url:
                continue
            handle = _friendly_handle(url, src.get("source_kind") or "")
            lines.append(f"- {handle}: {url}" if handle else f"- {url}")
        if len(selected) > 20:
            lines.append(strings["more_suffix"].format(n=len(selected) - 20))
        return "\n".join(lines)
    if status == "no_sources_found":
        reason = result.get("abort_reason") or result.get("reason") or ""
        hint = strings["none_hint"]
        if reason:
            return strings["none_with_reason"].format(reason=reason, hint=hint)
        return strings["none_no_reason"].format(hint=hint)
    if status == "skipped" and result.get("reason") == "at_capacity":
        return strings["at_capacity"].format(
            attached=int(result.get("attached", 0)),
            target=int(result.get("target", 0)),
        )
    return None


async def _notify_user_of_discovery_outcome(subscription_id: uuid.UUID, result: dict) -> None:
    """Post a follow-up delivery to the user's webhook after discovery ends.

    Moving discovery off the HTTP conversation turn means the user no longer
    learns the outcome from the streaming reply. This function is the
    replacement: it looks up the subscription's webhook target and posts a
    short human-readable summary in the user's language. Failures here are
    swallowed -- the delivery pipeline already logs, and a missed follow-up
    should not crash the Celery task.
    """
    try:
        async with get_task_session() as session:
            row = await session.execute(
                select(Subscription)
                .options(selectinload(Subscription.user))
                .where(Subscription.id == subscription_id)
            )
            subscription = row.scalar_one_or_none()
            if subscription is None or not subscription.is_active:
                return
            if subscription.paused_at is not None:
                return
            webhook_url = subscription.delivery_webhook_url
            language = subscription.digest_language or ""
            if subscription.user is not None:
                if webhook_url is None:
                    webhook_url = subscription.user.delivery_webhook_url
                if not language:
                    language = subscription.user.language or ""
        body = _format_completion_body(result, language)
        if body is None:
            return
        subject = _completion_strings(language)["subject"]
        await deliver(webhook_url, subject, body)
    except Exception:
        logger.exception(
            "Failed to deliver discovery completion notice for subscription %s",
            subscription_id,
        )


async def run_and_persist_discovery(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    reason: str,
    *,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    display_language: str = "en",
) -> dict:
    with subscription_tag(subscription_id):
        return await _run_and_persist_discovery_tagged(
            session,
            subscription_id,
            reason,
            status_queue=status_queue,
            display_language=display_language,
        )


async def _run_and_persist_discovery_tagged(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    reason: str,
    *,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    display_language: str = "en",
) -> dict:
    """Run discovery for one subscription and persist accepted sources.

    Caller provides the session; this function does not open its own.
    Returns a dict the caller can log or feed back to an LLM::

        {
            "status": "ok" | "skipped" | "no_sources_found",
            "reason": str,                       # present when skipped/no_sources_found;
                                                 # "at_capacity" when the subscription
                                                 # already has source_hard_cap or more
                                                 # attached sources
            "attached": int,                     # present when skipped due to at_capacity
            "target": int,                       # present when skipped due to at_capacity
                                                 # (the hard cap that triggered the skip)
            "abort_reason": str,                 # present on no_sources_found if the
                                                 # orchestrator called abort()
            "discovered": int,                   # sources the agent selected
            "persisted": int,                    # new SubscriptionSource rows
            "selected_sources": [                # present on ok
                {"url": str, "title": str, "source_kind": str},
                ...
            ],
        }

    The ``no_sources_found`` status is returned when the discovery pipeline
    terminates with zero selected sources. Callers (conversational agent,
    reflector) are expected to surface this to the user so they can refine
    the subscription topic -- a subscription with no attached sources is
    dead on arrival.

    When ``status_queue`` is provided, the underlying ``run_source_discovery``
    emits progress frames (``event: "discovery_progress"``) that the caller
    is expected to relay to the client. ``display_language`` controls which
    localized strings those frames carry.
    """
    sub_result = await session.execute(
        select(Subscription).where(Subscription.id == subscription_id)
    )
    subscription = sub_result.scalar_one_or_none()
    if subscription is None or not subscription.is_active:
        logger.warning("Discovery skipped: subscription %s not found or inactive", subscription_id)
        return {"status": "skipped", "reason": "not_found_or_inactive"}
    if subscription.paused_at is not None:
        logger.warning("Discovery skipped: subscription %s is paused", subscription_id)
        return {"status": "skipped", "reason": "paused"}
    if subscription.topic_embedding is None:
        logger.warning(
            "Discovery skipped: subscription %s has no retrieval embedding",
            subscription_id,
        )
        return {"status": "skipped", "reason": "no_embedding"}

    attached = await _load_attached_sources(session, subscription_id)
    removal_history = await _load_removal_history(session, subscription_id)
    locked_out_urls = await _load_recently_removed_urls(session, subscription_id)

    soft_max_new = max(0, settings.source_soft_cap - len(attached))
    hard_max_new = max(0, settings.source_hard_cap - len(attached))
    if hard_max_new == 0:
        logger.info(
            "Discovery skipped: subscription %s already has %d attached sources (hard cap %d)",
            subscription_id,
            len(attached),
            settings.source_hard_cap,
        )
        return {
            "status": "skipped",
            "reason": "at_capacity",
            "attached": len(attached),
            "target": settings.source_hard_cap,
        }

    result = await run_source_discovery(
        session=session,
        topic_text=subscription.user_spec or "",
        prompt_embedding=list(subscription.topic_embedding),
        user_spec=subscription.user_spec or "",
        attached_sources=attached,
        reason=reason,
        removal_history=removal_history,
        locked_out_urls=locked_out_urls,
        soft_max_new=soft_max_new,
        hard_max_new=hard_max_new,
        status_queue=status_queue,
        display_language=display_language,
    )

    sub_recheck = await session.execute(
        select(Subscription.is_active).where(Subscription.id == subscription_id)
    )
    is_active_now = sub_recheck.scalar_one_or_none()
    if not is_active_now:
        logger.warning(
            "Discovery results dropped: subscription %s was removed or "
            "deactivated during the run (discovered=%d).",
            subscription_id,
            len(result.sources),
        )
        return {
            "status": "skipped",
            "reason": "subscription_gone_after_discovery",
            "discovered": len(result.sources),
            "persisted": 0,
        }

    if not result.sources:
        logger.warning(
            "Discovery returned zero sources for subscription %s (reason=%r, abort=%r)",
            subscription_id,
            reason[:100],
            result.abort_reason[:200] if result.abort_reason else "",
        )
        payload: dict[str, Any] = {
            "status": "no_sources_found",
            "subscription_id": str(subscription_id),
            "reason": "no candidates found for this topic",
            "discovered": 0,
            "persisted": 0,
            "selected_sources": [],
        }
        if result.abort_reason:
            payload["abort_reason"] = result.abort_reason
        return payload

    persisted = 0
    selected_sources: list[dict[str, str]] = []
    for scored in result.sources:
        source = await ensure_source_by_url(
            session,
            url=scored.url,
            title=scored.title or scored.url,
            source_kind=scored.source_kind,
        )
        selected_sources.append(
            {
                "url": source.url,
                "title": source.title or source.url,
                "source_kind": scored.source_kind,
            }
        )
        link_exists = await session.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == subscription_id,
                SubscriptionSource.source_id == source.id,
            )
        )
        if link_exists.scalar_one_or_none() is not None:
            continue
        session.add(
            SubscriptionSource(
                subscription_id=subscription_id,
                source_id=source.id,
                is_user_specified=False,
            )
        )
        persisted += 1

    await session.commit()
    logger.info(
        "Discovery persisted %d new auto sources for subscription %s (reason=%r)",
        persisted,
        subscription_id,
        reason[:100],
    )
    return {
        "status": "ok",
        "subscription_id": str(subscription_id),
        "discovered": len(result.sources),
        "persisted": persisted,
        "selected_sources": selected_sources,
    }


async def _load_attached_sources(
    session: AsyncSession, subscription_id: uuid.UUID
) -> list[tuple[str, str, bool]]:
    rows = await session.execute(
        select(Source.url, Source.title, SubscriptionSource.is_user_specified)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription_id)
    )
    result: list[tuple[str, str, bool]] = []
    for url, _title, is_user in rows.all():
        result.append((url, _kind_from_url(url), bool(is_user)))
    return result


async def _load_removal_history(session: AsyncSession, subscription_id: uuid.UUID) -> str:
    rows = await session.execute(
        select(SourceRemovalLog.source_url, SourceRemovalLog.removal_reason)
        .where(SourceRemovalLog.subscription_id == subscription_id)
        .order_by(SourceRemovalLog.removed_at.desc())
        .limit(_REMOVAL_HISTORY_LIMIT)
    )
    lines = [f"- {url}: {reason or '(no reason)'}" for url, reason in rows.all()]
    return "\n".join(lines)


async def _load_recently_removed_urls(
    session: AsyncSession, subscription_id: uuid.UUID
) -> list[str]:
    """URLs removed from this subscription within the lockout window.

    The Reflector removes sources it judges dead, off-topic, or unreliable
    and logs the removal here. Re-discovery must not re-attach those URLs
    within ``discovery_removal_lockout_days`` regardless of how the LLM
    scores them -- a soft prompt hint proved insufficient in practice.
    """
    cutoff = datetime.now(UTC) - timedelta(days=settings.discovery_removal_lockout_days)
    rows = await session.execute(
        select(SourceRemovalLog.source_url).where(
            SourceRemovalLog.subscription_id == subscription_id,
            SourceRemovalLog.removed_at >= cutoff,
        )
    )
    return [url for (url,) in rows.all()]


def _kind_from_url(url: str) -> str:
    """Best-effort classification from URL shape for display to the agent."""
    lowered = url.lower()
    if "t.me/" in lowered:
        return "telegram_channel"
    if "reddit.com/r/" in lowered:
        return "reddit_subreddit"
    return "rss"
