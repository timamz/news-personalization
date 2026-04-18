"""Celery task that smooths each source's embedding toward its recent items.

Runs daily. For every active source we load the embeddings of items ingested
since the source's last embedding update, average them, and blend the mean
with the source's current embedding using the configured smoothing factor
(``old_weight * old + (1 - old_weight) * avg_new``). When a source has no
prior embedding, the average seeds it directly; when no new items arrived,
the source is left untouched.

The goal is to let each source's retrieval vector track genuine topic drift
(e.g. a feed pivoting focus over weeks) without recomputing embeddings and
without overreacting to a single noisy day of content.
"""

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select

from news_service.core.config import get_settings
from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.source import Source
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

UPDATE_SOURCE_EMBEDDINGS_TASK = (
    "news_service.tasks.update_source_embeddings.update_all_source_embeddings"
)

settings = get_settings()


@celery_app.task(name=UPDATE_SOURCE_EMBEDDINGS_TASK)
def update_all_source_embeddings() -> dict:
    """Entry point invoked by Celery Beat."""
    return asyncio.run(_update_all())


async def _update_all() -> dict:
    smoothing = settings.source_embedding_smoothing
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError(
            f"source_embedding_smoothing must be in [0, 1], got {smoothing!r}",
        )

    updated = 0
    skipped = 0
    async with get_task_session() as session:
        sources = (
            (await session.execute(select(Source).where(Source.is_active.is_(True))))
            .scalars()
            .all()
        )

        for source in sources:
            new_embeddings = await _load_new_embeddings(
                session, source.id, source.last_embedding_update_at
            )
            if not new_embeddings:
                skipped += 1
                continue

            avg = _mean_vector(new_embeddings)
            if source.source_description_embedding is None:
                blended = avg
            else:
                blended = _blend(
                    list(source.source_description_embedding), avg, old_weight=smoothing
                )

            source.source_description_embedding = blended
            source.last_embedding_update_at = datetime.now(UTC)
            updated += 1

        await session.commit()

    logger.info(
        "Source embedding refresh: updated=%d skipped=%d smoothing=%.2f",
        updated,
        skipped,
        smoothing,
    )
    return {"status": "ok", "updated": updated, "skipped": skipped}


async def _load_new_embeddings(session, source_id, since: datetime | None) -> list[list[float]]:
    stmt = select(NewsItem.embedding).where(
        NewsItem.source_id == source_id,
        NewsItem.embedding.is_not(None),
    )
    if since is not None:
        stmt = stmt.where(NewsItem.fetched_at > since)
    rows = await session.execute(stmt)
    return [list(emb) for (emb,) in rows.all() if emb is not None]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    totals = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            totals[i] += x
    return [t / n for t in totals]


def _blend(old: list[float], new: list[float], *, old_weight: float) -> list[float]:
    new_weight = 1.0 - old_weight
    return [old_weight * o + new_weight * n for o, n in zip(old, new, strict=True)]
