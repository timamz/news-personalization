"""Digest generation pipeline: Fetch -> Plan -> LoopAgent([Compose, Judge]) -> Reflect.

This replaces the old single-shot digest_curator with a multi-stage pipeline
featuring quality-gated revision and self-healing reflection.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

from .candidates import build_items_text, fetch_candidate_items
from .composer import compose_digest
from .judge import judge_digest
from .planner import plan_digest
from .reflector import reflect_on_pipeline

logger = logging.getLogger(__name__)
settings = get_settings()

_MAX_REVISIONS = 2


def _effective_prompt(subscription: Subscription) -> str:
    """Extract the topic text used for embedding, preferring user_spec."""
    if subscription.user_spec:
        for line in subscription.user_spec.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## Topic"):
                continue
            if stripped.startswith("##"):
                break
            if stripped:
                return stripped
    return subscription.canonical_prompt or subscription.raw_prompt


async def generate_digest(session: AsyncSession, subscription: Subscription) -> str | None:
    """Run the full digest pipeline for a subscription.

    Pipeline stages:
    1. Fetch candidates (DB queries, no LLM)
    2. Plan digest outline (Planner LLM call)
    3. Compose + Judge loop (max 2 revisions)
    4. Reflect on pipeline health (Reflector LLM call)
    5. Mark items as sent

    Returns the digest text, or None if no usable items.
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

    query_embedding = subscription.topic_embedding or subscription.canonical_prompt_embedding
    if query_embedding is None:
        query_text = _effective_prompt(subscription)
        query_embedding = await embed_text(query_text)
        subscription.topic_embedding = query_embedding

    published_after = _published_after_for_digest(last_sent_at)
    user_spec = subscription.user_spec or subscription.canonical_prompt or ""

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

    # --- Stage 2: Plan ---
    try:
        plan_result = await plan_digest(
            user_spec=user_spec,
            items_text=items_text,
            digest_language=subscription.digest_language,
            format_instructions=subscription.format_instructions,
        )
    except Exception:
        logger.exception(
            "Digest planner failed for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    # --- Stage 3: Compose + Judge loop ---
    feedback = ""
    composition = None
    quality = None

    for revision in range(_MAX_REVISIONS):
        try:
            composition = await compose_digest(
                plan=plan_result.plan,
                items_text=items_text,
                user_spec=user_spec,
                digest_language=subscription.digest_language,
                format_instructions=subscription.format_instructions,
                feedback=feedback,
            )
        except Exception:
            logger.exception(
                "Digest composer failed (revision %d) for subscription %s",
                revision,
                subscription.id,
                extra={"subscription_id": str(subscription.id)},
            )
            return None

        if not composition.used_item_ids:
            logger.info(
                "Composer returned no items for subscription %s",
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
            logger.exception(
                "Digest judge failed for subscription %s",
                subscription.id,
                extra={"subscription_id": str(subscription.id)},
            )
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

    # --- Stage 4: Reflect (fire-and-forget, don't block delivery) ---
    try:
        reflection = await reflect_on_pipeline(
            digest_text=composition.digest_text,
            user_spec=user_spec,
            quality_scores=quality.model_dump() if quality else {},
            source_contributions=candidates_summary,
        )

        if reflection.user_spec_patch and subscription.user_spec:
            subscription.user_spec = (
                subscription.user_spec.rstrip() + f"\n\n{reflection.observations}"
            )
            await session.flush()

        if reflection.sources_to_remove:
            logger.info(
                "Reflector wants to remove %d sources for subscription %s",
                len(reflection.sources_to_remove),
                subscription.id,
            )

    except Exception:
        logger.exception(
            "Pipeline reflector failed for subscription %s (non-blocking)",
            subscription.id,
        )

    # --- Stage 5: Mark as sent ---
    used_ids = [uuid.UUID(item_id) for item_id in composition.used_item_ids]
    await _mark_as_sent(session, subscription.id, used_ids)

    logger.info(
        "Generated digest with %d items for subscription %s (revisions: %d)",
        len(used_ids),
        subscription.id,
        _MAX_REVISIONS - 1 if quality and quality.verdict != "PASS" else 0,
        extra={"subscription_id": str(subscription.id)},
    )
    return composition.digest_text


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
