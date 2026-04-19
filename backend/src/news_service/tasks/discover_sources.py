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
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.source_discovery import run_source_discovery
from news_service.db.session import get_task_session
from news_service.models.source import Source
from news_service.models.source_removal_log import SourceRemovalLog
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.coverage import ensure_source_by_url
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

DISCOVER_SOURCES_TASK = "news_service.tasks.discover_sources.discover_sources_for_subscription"

_REMOVAL_HISTORY_LIMIT = 50


@celery_app.task(name=DISCOVER_SOURCES_TASK)
def discover_sources_for_subscription(subscription_id: str, reason: str = "") -> dict:
    """Celery entry point. Bridges to the async impl."""
    return asyncio.run(_discover_in_task_session(uuid.UUID(subscription_id), reason))


async def _discover_in_task_session(subscription_id: uuid.UUID, reason: str) -> dict:
    async with get_task_session() as session:
        return await run_and_persist_discovery(session, subscription_id, reason)


async def run_and_persist_discovery(
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
            "status": "ok" | "skipped",
            "reason": str,                       # present when skipped
            "discovered": int,                   # sources the agent selected
            "persisted": int,                    # new SubscriptionSource rows
            "selected_sources": [                # present on ok
                {"url": str, "title": str, "source_kind": str},
                ...
            ],
        }

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
    if subscription.topic_embedding is None:
        logger.warning(
            "Discovery skipped: subscription %s has no retrieval embedding",
            subscription_id,
        )
        return {"status": "skipped", "reason": "no_embedding"}

    attached = await _load_attached_sources(session, subscription_id)
    removal_history = await _load_removal_history(session, subscription_id)

    result = await run_source_discovery(
        session=session,
        topic_text=subscription.user_spec or "",
        prompt_embedding=list(subscription.topic_embedding),
        user_spec=subscription.user_spec or "",
        attached_sources=attached,
        reason=reason,
        removal_history=removal_history,
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


def _kind_from_url(url: str) -> str:
    """Best-effort classification from URL shape for display to the agent."""
    lowered = url.lower()
    if "t.me/" in lowered:
        return "telegram_channel"
    if "reddit.com/r/" in lowered:
        return "reddit_subreddit"
    if "twitter.com/" in lowered or "x.com/" in lowered:
        return "twitter_account"
    return "rss"
