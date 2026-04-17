"""Digest generation pipeline: Fetch -> Write + Judge loop -> Reflect.

Multi-stage pipeline with quality-gated revision and self-healing reflection.
Error handling follows the tiered policy:
- Critical (writer): raise DigestPipelineError
- Quality gate (judge): log warning, use unreviewed draft
- Non-blocking (reflector): log and swallow
"""

import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.exceptions import DigestPipelineError
from news_service.core.guardrails import validate_digest_text, validate_used_item_ids
from news_service.db.vector_store import embed_text
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

from .candidates import build_items_text, fetch_candidate_items
from .judge import judge_digest
from .reflector import run_reflector
from .writer import write_digest

logger = logging.getLogger(__name__)
settings = get_settings()

_MAX_REVISIONS = 2
_RECENT_DIGEST_LIMIT = 15


async def generate_digest(session: AsyncSession, subscription: Subscription) -> str | None:
    """Run the full digest pipeline for a subscription.

    Pipeline stages:
    1. Fetch candidates (DB queries, no LLM)
    2. Write + Judge loop (max 2 revisions) -- writer raises, judge degrades gracefully
    3. Reflect on pipeline health (Reflector ADK agent) -- non-blocking
    4. Mark items as sent

    Returns the digest text, or None if no usable items.
    Raises DigestPipelineError if a critical stage fails.
    """
    sent_result = await session.execute(
        select(SentItem.news_item_id, SentItem.sent_at).where(
            SentItem.subscription_id == subscription.id
        )
    )
    sent_rows = list(sent_result.all())
    sent_ids: set[uuid.UUID] = {news_item_id for news_item_id, _ in sent_rows}
    last_sent_at = max((sent_at for _, sent_at in sent_rows), default=None)

    source_ids = await _source_ids_for_digest(session, subscription)
    if not source_ids:
        logger.warning(
            "No fixed sources for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    query_embedding = subscription.topic_embedding
    if query_embedding is None:
        logger.warning(
            "Subscription %s missing topic_embedding; falling back to user_spec",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        query_embedding = await embed_text(subscription.user_spec or "")
        subscription.topic_embedding = query_embedding

    published_after = _published_after_for_digest(last_sent_at)
    user_spec = subscription.user_spec or ""

    # --- Stage 1: Fetch candidates (no LLM) ---
    candidates = await fetch_candidate_items(
        session,
        query_embedding,
        exclude_ids=sent_ids,
        allowed_source_ids=source_ids,
        published_after=published_after,
    )
    if not candidates:
        logger.info(
            "No candidates for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    items_text = build_items_text(candidates, settings.llm_max_context_chars)
    candidates_summary = f"{len(candidates)} candidates from {len(source_ids)} sources"

    recent_summaries = await _build_recent_digest_summaries(session, subscription.id)

    # --- Stage 2: Write + Judge loop ---
    feedback = ""
    composition = None
    quality = None

    for revision in range(_MAX_REVISIONS):
        try:
            composition = await write_digest(
                items_text=items_text,
                user_spec=user_spec,
                digest_language=subscription.digest_language,
                recent_digest_summaries=recent_summaries,
                feedback=feedback,
            )
        except Exception as exc:
            raise DigestPipelineError(
                f"Writer failed (revision {revision}) for subscription {subscription.id}: {exc}"
            ) from exc

        if not composition.used_item_ids:
            logger.info(
                "Writer returned no items for subscription %s",
                subscription.id,
                extra={"subscription_id": str(subscription.id)},
            )
            return None

        try:
            quality = await judge_digest(
                digest_text=composition.digest_text,
                user_spec=user_spec,
                candidates_summary=candidates_summary,
            )
        except Exception:
            logger.warning(
                "Judge failed for subscription %s; using unreviewed draft",
                subscription.id,
                extra={"subscription_id": str(subscription.id)},
            )
            quality = None
            break

        if quality.verdict == "PASS":
            break

        feedback = quality.feedback
        logger.info(
            "Digest revision %d requested for subscription %s: %s",
            revision + 1,
            subscription.id,
            feedback[:100],
            extra={"subscription_id": str(subscription.id)},
        )

    if composition is None:
        return None

    candidate_ids = {str(item.id) for item in candidates}
    composition.used_item_ids = validate_used_item_ids(composition.used_item_ids, candidate_ids)
    if not composition.used_item_ids:
        logger.warning(
            "All used_item_ids were phantom for subscription %s",
            subscription.id,
        )
        return None

    composition.digest_text = validate_digest_text(composition.digest_text)

    # --- Stage 3: Reflect (data-driven trigger, non-blocking) ---
    should_reflect = _should_reflect(
        subscription=subscription,
        quality=quality,
        candidates=candidates,
        source_ids=source_ids,
    )

    if should_reflect:
        try:
            source_info = await _build_source_info(session, subscription.id, source_ids, candidates)

            shared_state = await run_reflector(
                db_session=session,
                subscription=subscription,
                digest_text=composition.digest_text,
                user_spec=user_spec,
                quality_scores=quality.model_dump() if quality else {},
                source_info=source_info,
            )

            subscription.last_reflected_at = datetime.now(UTC)

            if shared_state.get("discovery_triggered"):
                _queue_discovery(subscription, shared_state.get("discovery_reason", ""))

        except Exception:
            logger.exception(
                "Pipeline reflector failed for subscription %s (non-blocking)",
                subscription.id,
            )

    # --- Stage 4: Mark as sent ---
    used_ids = [uuid.UUID(item_id) for item_id in composition.used_item_ids]
    await _mark_as_sent(session, subscription.id, used_ids)

    logger.info(
        "Generated digest with %d items for subscription %s (reflected: %s)",
        len(used_ids),
        subscription.id,
        should_reflect,
        extra={"subscription_id": str(subscription.id)},
    )
    return composition.digest_text


async def _build_recent_digest_summaries(
    session: AsyncSession,
    subscription_id: uuid.UUID,
) -> str:
    """Build a rolling-window summary of recently sent digests.

    Queries the last N sent items, joins to NewsItem for headlines,
    groups by date, and formats as a short summary for the writer prompt.
    Returns an empty string if no previous digests exist.
    """
    stmt = (
        select(
            func.date(SentItem.sent_at).label("sent_date"),
            NewsItem.headline,
        )
        .join(NewsItem, NewsItem.id == SentItem.news_item_id)
        .where(SentItem.subscription_id == subscription_id)
        .order_by(SentItem.sent_at.desc())
        .limit(_RECENT_DIGEST_LIMIT)
    )
    result = await session.execute(stmt)
    rows = list(result.all())

    if not rows:
        return ""

    by_date: dict[str, list[str]] = defaultdict(list)
    for sent_date, headline in rows:
        date_str = sent_date.strftime("%b %d") if hasattr(sent_date, "strftime") else str(sent_date)
        short_headline = headline[:60] if len(headline) > 60 else headline
        by_date[date_str].append(short_headline)

    lines = ["Recent digests:"]
    for date_str, headlines in by_date.items():
        lines.append(f"- {date_str}: {', '.join(headlines)}")

    return "\n".join(lines)


def _should_reflect(
    *,
    subscription: Subscription,
    quality: object | None,
    candidates: list,
    source_ids: set[uuid.UUID],
) -> bool:
    """Decide whether to run the reflector based on pipeline health signals."""
    pipeline_struggled = quality is None or getattr(quality, "verdict", None) != "PASS"
    if pipeline_struggled:
        return True

    contributing_source_ids = {item.source_id for item in candidates}
    source_coverage = len(contributing_source_ids) / len(source_ids) if source_ids else 1.0
    if source_coverage < settings.reflector_coverage_threshold:
        return True

    relevance = getattr(quality, "relevance", 5)
    format_score = getattr(quality, "format_score", 5)
    conciseness = getattr(quality, "conciseness", 5)
    avg_score = (relevance + format_score + conciseness) / 3
    if avg_score < settings.reflector_quality_threshold:
        return True

    if subscription.last_reflected_at is None:
        return True

    days_since = (datetime.now(UTC) - subscription.last_reflected_at).days
    return days_since >= settings.reflector_max_interval_days


async def _build_source_info(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    source_ids: set[uuid.UUID],
    candidates: list,
) -> str:
    """Build labeled source info string for the reflector prompt."""
    contribution_counts: dict[uuid.UUID, int] = {}
    for item in candidates:
        contribution_counts[item.source_id] = contribution_counts.get(item.source_id, 0) + 1
    for sid in source_ids:
        contribution_counts.setdefault(sid, 0)

    links_result = await session.execute(
        select(SubscriptionSource.source_id, SubscriptionSource.is_user_specified).where(
            SubscriptionSource.subscription_id == subscription_id
        )
    )
    user_specified: dict[uuid.UUID, bool] = {row[0]: row[1] for row in links_result.all()}

    source_result = await session.execute(
        select(Source.id, Source.url, Source.title).where(Source.id.in_(list(source_ids)))
    )
    source_details = {row[0]: (row[1], row[2]) for row in source_result.all()}

    lines: list[str] = []
    for sid in source_ids:
        url, title = source_details.get(sid, ("unknown", "unknown"))
        is_user = user_specified.get(sid, False)
        label = "user-specified, DO NOT remove" if is_user else "auto-discovered, removable"
        count = contribution_counts.get(sid, 0)
        display = f"{title} ({url})" if title else url
        lines.append(f"- {display} [{label}] -- {count} candidates")

    return "\n".join(lines)


def _queue_discovery(subscription: Subscription, reason: str) -> None:
    """Queue the real source-discovery task after the reflector requests it."""
    from news_service.tasks.celery_app import celery_app
    from news_service.tasks.discover_sources import DISCOVER_SOURCES_TASK

    celery_app.send_task(
        DISCOVER_SOURCES_TASK,
        args=[str(subscription.id), reason],
    )
    logger.info(
        "Reflector triggered source discovery for subscription %s: %s",
        subscription.id,
        reason[:100],
    )


def _published_after_for_digest(last_sent_at: datetime | None) -> datetime:
    if last_sent_at is not None:
        return last_sent_at
    return datetime.now(UTC) - timedelta(days=settings.news_item_max_age_days)


async def _source_ids_for_digest(
    session: AsyncSession,
    subscription: Subscription,
) -> set[uuid.UUID]:
    source_result = await session.execute(
        select(Source.id)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription.id)
    )
    return {row[0] for row in source_result.all()}


async def _mark_as_sent(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    news_item_ids: list[uuid.UUID],
) -> None:
    for item_id in news_item_ids:
        session.add(SentItem(subscription_id=subscription_id, news_item_id=item_id))
    await session.flush()
