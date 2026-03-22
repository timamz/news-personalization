from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import SourceKind
from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text
from news_service.models.source import Source
from news_service.services.reddit import build_reddit_subreddit_url
from news_service.services.source_descriptions import describe_source
from news_service.services.telegram import build_telegram_channel_url
from news_service.services.twitter import build_twitter_account_url

if TYPE_CHECKING:
    from news_service.agents.source_discovery import ScoredSource

logger = logging.getLogger(__name__)
settings = get_settings()

_SOURCE_TYPE_CONFIG: dict[SourceKind, tuple[Callable[[str], str], str]] = {
    "telegram_channel": (build_telegram_channel_url, "Telegram @{}"),
    "reddit_subreddit": (build_reddit_subreddit_url, "Reddit r/{}"),
    "twitter_account": (build_twitter_account_url, "X @{}"),
}


async def ensure_prompt_coverage(
    session: AsyncSession,
    raw_prompt: str,
    prompt_embedding: list[float],
    hooks: object | None = None,
) -> list[Source]:
    from news_service.agents.source_discovery import run_source_discovery

    try:
        result = await run_source_discovery(
            session=session,
            raw_prompt=raw_prompt,
            prompt_embedding=prompt_embedding,
            hooks=hooks,
        )
    except Exception:
        logger.exception("Source discovery agent failed for prompt: %s", raw_prompt)
        return []

    selected: list[Source] = []
    for scored in result.sources:
        source_obj = await _register_or_reuse_source(session, scored)
        if source_obj is not None:
            selected.append(source_obj)

    logger.info(
        "Selected %d sources for prompt (scores: %s)",
        len(selected),
        ", ".join(f"{s.relevance_score:.3f}" for s in result.sources),
    )
    return selected


async def _register_or_reuse_source(
    session: AsyncSession,
    scored: ScoredSource,
) -> Source | None:
    existing_result = await session.execute(select(Source).where(Source.url == scored.url))
    existing_source = existing_result.scalar_one_or_none()
    if existing_source is not None:
        existing_source.subscriber_count += 1
        existing_source.is_active = True
        return existing_source

    description, embedding = await _build_source_profile(
        source_kind=scored.source_kind,
        title=scored.title or scored.url,
        url=scored.url,
    )
    source_obj = Source(
        url=scored.url,
        title=scored.title or scored.url,
        source_description=description,
        source_description_embedding=embedding,
        is_active=True,
        subscriber_count=1,
    )
    session.add(source_obj)
    await session.flush()
    logger.info("Registered new source: %s (%s)", scored.url, scored.title)
    return source_obj


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
        result = await session.execute(select(Source).where(Source.url == source_url))
        existing_source = result.scalar_one_or_none()
        if existing_source is not None:
            existing_source.subscriber_count += 1
            existing_source.is_active = True
            resolved[existing_source.id] = existing_source
            logger.info("Source already exists: %s", source_url)
            continue

        title = title_template.format(identifier)
        description, embedding = await _build_source_profile(
            source_kind=source_kind,
            title=title,
            url=source_url,
        )
        source_obj = Source(
            url=source_url,
            title=title,
            source_description=description,
            source_description_embedding=embedding,
            is_active=True,
            subscriber_count=1,
        )
        session.add(source_obj)
        await session.flush()
        resolved[source_obj.id] = source_obj
        logger.info("Registered %s source: %s", source_kind, source_url)

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
