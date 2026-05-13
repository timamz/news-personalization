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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.exceptions import DigestPipelineError
from news_service.core.guardrails import validate_digest_text, validate_used_item_ids
from news_service.core.provider_errors import ProviderLimitError
from news_service.db.vector_store import embed_text
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.relevance import cosine_similarity

from .candidates import build_items_text, fetch_candidate_items
from .judge import judge_digest
from .reflector import run_reflector
from .writer import write_digest

logger = logging.getLogger(__name__)
settings = get_settings()

_MAX_REVISIONS = 3
_RECENT_DIGEST_LIMIT = 15


async def generate_digest(session: AsyncSession, subscription: Subscription) -> str | None:
    """Run the full digest pipeline for a subscription.

    Pipeline stages:
    1. Fetch candidates (DB queries, no LLM)
    2. Write + Judge loop (max 3 revisions) -- writer raises, judge degrades gracefully
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
        fetched_after=last_sent_at,
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
        except ProviderLimitError:
            raise
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
    try:
        source_contexts = await _load_source_contexts(
            session=session,
            subscription_id=subscription.id,
            source_ids=source_ids,
            topic_embedding=list(query_embedding),
            candidates=candidates,
        )
        reasons = _compute_reflect_reasons(
            quality=quality,
            source_contexts=source_contexts,
        )
    except Exception:
        logger.exception(
            "Failed to compute reflector triggers for subscription %s",
            subscription.id,
        )
        reasons, source_contexts = [], []

    should_reflect = bool(reasons)
    if should_reflect:
        try:
            shared_state = await run_reflector(
                db_session=session,
                subscription=subscription,
                digest_text=composition.digest_text,
                user_spec=user_spec,
                quality_scores=quality.model_dump() if quality else {},
                trigger_reasons=reasons,
                source_contexts=source_contexts,
                allowed_source_ids=source_ids,
                topic_embedding=list(query_embedding),
            )

            subscription.last_reflected_at = datetime.now(UTC)

            if shared_state.get("discovery_triggered"):
                _queue_discovery(subscription, shared_state.get("discovery_reason", ""))

        except Exception:
            logger.exception(
                "Pipeline reflector failed for subscription %s (non-blocking)",
                subscription.id,
            )

    # --- Stage 4: Mark as sent, update contribution streaks ---
    used_ids = [uuid.UUID(item_id) for item_id in composition.used_item_ids]
    await _mark_as_sent(session, subscription.id, used_ids)

    used_id_set = set(used_ids)
    contributing_source_ids = {item.source_id for item in candidates if item.id in used_id_set}
    await _update_contribution_streaks(
        session=session,
        subscription_id=subscription.id,
        contributing_source_ids=contributing_source_ids,
    )

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


@dataclass(slots=True)
class ReflectorSourceContext:
    """Per-source metadata shown to the Reflector and used to compute triggers."""

    source_id: uuid.UUID
    url: str
    title: str
    is_user_specified: bool
    contribution_count: int
    cosine_to_topic: float | None
    last_published_at: datetime | None
    days_since_last_published: int | None
    contributed_last_30_digests: int
    contribution_rate: float
    digests_since_last_contribution: int
    item_cosine_p50: float | None
    item_cosine_p90: float | None
    item_cosine_std: float | None


async def _load_source_contexts(
    *,
    session: AsyncSession,
    subscription_id: uuid.UUID,
    source_ids: set[uuid.UUID],
    topic_embedding: list[float],
    candidates: list,
    now: datetime | None = None,
) -> list[ReflectorSourceContext]:
    """Load rich per-source metadata for the Reflector in a single query."""
    if not source_ids:
        return []
    effective_now = now or datetime.now(UTC)

    contribution_counts: dict[uuid.UUID, int] = {}
    for item in candidates:
        contribution_counts[item.source_id] = contribution_counts.get(item.source_id, 0) + 1

    stmt = (
        select(
            Source.id,
            Source.url,
            Source.title,
            Source.source_description_embedding,
            SubscriptionSource.is_user_specified,
            SubscriptionSource.contributed_last_30_digests,
            SubscriptionSource.contribution_rate,
            SubscriptionSource.digests_since_last_contribution,
            SubscriptionSource.item_cosine_p50,
            SubscriptionSource.item_cosine_p90,
            SubscriptionSource.item_cosine_std,
            func.max(NewsItem.published_at).label("last_published_at"),
        )
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .outerjoin(NewsItem, NewsItem.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription_id)
        .where(Source.id.in_(list(source_ids)))
        .group_by(
            Source.id,
            Source.url,
            Source.title,
            SubscriptionSource.is_user_specified,
            SubscriptionSource.contributed_last_30_digests,
            SubscriptionSource.contribution_rate,
            SubscriptionSource.digests_since_last_contribution,
            SubscriptionSource.item_cosine_p50,
            SubscriptionSource.item_cosine_p90,
            SubscriptionSource.item_cosine_std,
        )
    )
    result = await session.execute(stmt)

    contexts: list[ReflectorSourceContext] = []
    for row in result.all():
        (
            sid,
            url,
            title,
            emb,
            is_user_specified,
            contributed_30,
            contribution_rate,
            streak,
            p50,
            p90,
            std,
            last_pub,
        ) = row
        cos = None
        if emb is not None:
            try:
                cos = cosine_similarity(list(emb), topic_embedding)
            except Exception:
                cos = None

        days_since = None
        if last_pub is not None:
            delta = effective_now - last_pub
            days_since = max(delta.days, 0)

        contexts.append(
            ReflectorSourceContext(
                source_id=sid,
                url=url,
                title=title or "",
                is_user_specified=bool(is_user_specified),
                contribution_count=contribution_counts.get(sid, 0),
                cosine_to_topic=cos,
                last_published_at=last_pub,
                days_since_last_published=days_since,
                contributed_last_30_digests=int(contributed_30 or 0),
                contribution_rate=float(contribution_rate or 0.0),
                digests_since_last_contribution=int(streak or 0),
                item_cosine_p50=p50,
                item_cosine_p90=p90,
                item_cosine_std=std,
            )
        )
    return contexts


def _compute_reflect_reasons(
    *,
    quality: object | None,
    source_contexts: list[ReflectorSourceContext],
) -> list[str]:
    """Collect human-readable reasons that justify running the Reflector.

    Callers run the Reflector iff the returned list is non-empty. The list is
    also passed into the Reflector prompt so the agent knows why it was
    invoked without re-discovering the signals.
    """
    reasons: list[str] = []

    if quality is not None and getattr(quality, "verdict", None) != "PASS":
        feedback = getattr(quality, "feedback", "") or ""
        snippet = feedback[:160].strip()
        suffix = f" Feedback: {snippet}" if snippet else ""
        reasons.append(f"Final digest verdict was REVISE after max revisions.{suffix}")

    drift_threshold = settings.reflector_drift_similarity_threshold
    staleness_days = settings.reflector_source_staleness_days
    streak_threshold = settings.reflector_contribution_streak_threshold
    for ctx in source_contexts:
        if ctx.cosine_to_topic is not None and ctx.cosine_to_topic < drift_threshold:
            reasons.append(
                f"Source {ctx.url} drifted from the subscription topic "
                f"(cos sim {ctx.cosine_to_topic:.2f} < {drift_threshold:.2f})."
            )
        if (
            ctx.days_since_last_published is not None
            and ctx.days_since_last_published >= staleness_days
        ):
            reasons.append(
                f"Source {ctx.url} has not published for {ctx.days_since_last_published} days."
            )
        if ctx.digests_since_last_contribution >= streak_threshold:
            reasons.append(
                f"Source {ctx.url} has not contributed to "
                f"{ctx.digests_since_last_contribution} consecutive digests "
                f"(threshold {streak_threshold})."
            )
    return reasons


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
    # De-dupe: Writer.used_item_ids can contain repeats and the benchmark
    # also sees the same scenario content polled into the DB multiple
    # times under different institutional-alias source_ids. Without this
    # the INSERT violates the ``uq_sent_item`` (subscription_id,
    # news_item_id) unique constraint.
    seen: set[uuid.UUID] = set()
    for item_id in news_item_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        session.add(SentItem(subscription_id=subscription_id, news_item_id=item_id))
    await session.flush()


async def _update_contribution_streaks(
    *,
    session: AsyncSession,
    subscription_id: uuid.UUID,
    contributing_source_ids: set[uuid.UUID],
) -> None:
    """After a digest ships, reset streak to 0 for contributing sources and
    increment it for the rest of the pool. Keeps ``digests_since_last_contribution``
    accurate in real time so the Reflector's streak trigger does not wait for
    the nightly stats job."""
    links = (
        (
            await session.execute(
                select(SubscriptionSource).where(
                    SubscriptionSource.subscription_id == subscription_id
                )
            )
        )
        .scalars()
        .all()
    )
    for link in links:
        if link.source_id in contributing_source_ids:
            link.digests_since_last_contribution = 0
        else:
            link.digests_since_last_contribution = link.digests_since_last_contribution + 1
    await session.flush()
