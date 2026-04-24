import logging
import uuid
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import SourceKind
from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text
from news_service.models.source import Source
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.source_descriptions import describe_source
from news_service.services.telegram import build_telegram_channel_url

logger = logging.getLogger(__name__)
settings = get_settings()

_SOURCE_TYPE_CONFIG: dict[SourceKind, tuple[Callable[[str], str], str]] = {
    "telegram_channel": (build_telegram_channel_url, "Telegram @{}"),
    "reddit_subreddit": (build_reddit_subreddit_url, "Reddit r/{}"),
}


async def ensure_source_coverage(
    session: AsyncSession,
    identifiers: list[str],
    source_kind: SourceKind,
) -> list[Source]:
    if not identifiers:
        return []

    url_builder, title_template = _SOURCE_TYPE_CONFIG[source_kind]

    resolved: dict[uuid.UUID, Source] = {}
    for identifier in identifiers:
        source_url = url_builder(identifier)
        title = title_template.format(identifier)
        source_obj = await _ensure_single_source(
            session, url=source_url, title=title, source_kind=source_kind
        )
        resolved[source_obj.id] = source_obj

    return list(resolved.values())


async def _build_source_profile(
    *,
    source_kind: SourceKind,
    title: str,
    url: str,
    sample_content: list[str] | None = None,
) -> tuple[str, list[float]]:
    description = await describe_source(
        source_kind=source_kind,
        title=title,
        url=url,
        sample_content=sample_content,
    )
    embedding = await embed_text(description)
    return description, embedding


async def ensure_source_by_url(
    session: AsyncSession,
    *,
    url: str,
    title: str,
    source_kind: SourceKind,
) -> Source:
    """Fetch an existing Source by URL or create one with a generated profile.

    Used by the discovery pipeline to persist sources already produced as full
    URLs by the finders, bypassing the identifier-based ``ensure_source_coverage``.
    """
    return await _ensure_single_source(session, url=url, title=title, source_kind=source_kind)


async def _ensure_single_source(
    session: AsyncSession,
    *,
    url: str,
    title: str,
    source_kind: SourceKind,
) -> Source:
    """Return a Source row for ``url``, inserting one if none exists.

    Concurrency-safe against another task inserting the same URL between
    our SELECT and our INSERT: on ``UniqueViolationError`` the nested
    savepoint is rolled back and we re-SELECT the row that the other
    task just committed. Without this, two parallel discovery tasks that
    both picked e.g. ``https://www.reddit.com/r/SpaceX/`` would race and
    one would crash with ``IntegrityError``, rolling back its whole
    persist transaction and leaving its subscription with zero sources.
    """
    result = await session.execute(select(Source).where(Source.url == url))
    existing = result.scalar_one_or_none()
    if existing is not None:
        existing.subscriber_count += 1
        existing.is_active = True
        return existing

    description, embedding = await _build_source_profile(
        source_kind=source_kind,
        title=title,
        url=url,
    )
    source = Source(
        url=url,
        title=title,
        source_description=description,
        source_description_embedding=embedding,
        is_active=True,
        subscriber_count=1,
    )
    try:
        async with session.begin_nested():
            session.add(source)
            await session.flush()
    except IntegrityError:
        logger.info(
            "Source %s was inserted concurrently; adopting the existing row",
            url,
        )
        result = await session.execute(select(Source).where(Source.url == url))
        existing = result.scalar_one_or_none()
        if existing is None:
            raise
        existing.subscriber_count += 1
        existing.is_active = True
        return existing

    logger.info("Registered %s source: %s", source_kind, url)
    return source
