"""Unit tests for the daily source-embedding smoothing task."""

import logging
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

logging.disable(logging.CRITICAL)


def _source(*, embedding: list[float] | None, last_update=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        is_active=True,
        source_description_embedding=embedding,
        last_embedding_update_at=last_update,
    )


def _patch_session_factory(mocker, session: MagicMock) -> None:
    @asynccontextmanager
    async def _factory():
        yield session

    mocker.patch(
        "news_service.tasks.update_source_embeddings.get_task_session",
        new=_factory,
    )


def _wire_session(mocker, sources: list[SimpleNamespace], new_items_by_source: dict) -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()

    sources_result = MagicMock()
    sources_scalars = MagicMock()
    sources_scalars.all.return_value = sources
    sources_result.scalars.return_value = sources_scalars

    per_source_results: list[MagicMock] = []
    for src in sources:
        rows = MagicMock()
        rows.all.return_value = [(emb,) for emb in new_items_by_source.get(src.id, [])]
        per_source_results.append(rows)

    session.execute = AsyncMock(side_effect=[sources_result, *per_source_results])
    _patch_session_factory(mocker, session)
    return session


@pytest.mark.asyncio
async def test_update_blends_old_and_new_with_configured_smoothing(mocker) -> None:
    mocker.patch(
        "news_service.tasks.update_source_embeddings.settings",
        SimpleNamespace(source_embedding_smoothing=0.9),
    )
    src = _source(embedding=[1.0, 0.0, 0.0])
    session = _wire_session(mocker, [src], {src.id: [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]})

    from news_service.tasks.update_source_embeddings import _update_all

    result = await _update_all()

    assert (
        result["updated"] == 1
        and result["skipped"] == 0
        and src.source_description_embedding == pytest.approx([0.9, 0.1, 0.0], abs=1e-6)
        and session.commit.await_count == 1
    ), "smoothing did not compute 0.9*old + 0.1*mean(new) and commit once"


@pytest.mark.asyncio
async def test_update_seeds_embedding_from_mean_when_source_has_none(mocker) -> None:
    mocker.patch(
        "news_service.tasks.update_source_embeddings.settings",
        SimpleNamespace(source_embedding_smoothing=0.9),
    )
    src = _source(embedding=None)
    _wire_session(mocker, [src], {src.id: [[0.4, 0.6], [0.6, 0.4]]})

    from news_service.tasks.update_source_embeddings import _update_all

    await _update_all()

    assert src.source_description_embedding == pytest.approx([0.5, 0.5], abs=1e-6), (
        "cold-start source must be seeded from the mean, not blended with zeros"
    )


@pytest.mark.asyncio
async def test_update_skips_sources_with_no_new_items(mocker) -> None:
    mocker.patch(
        "news_service.tasks.update_source_embeddings.settings",
        SimpleNamespace(source_embedding_smoothing=0.9),
    )
    untouched = [0.3, 0.7]
    src = _source(embedding=list(untouched))
    _wire_session(mocker, [src], {src.id: []})

    from news_service.tasks.update_source_embeddings import _update_all

    result = await _update_all()

    assert (
        result["updated"] == 0
        and result["skipped"] == 1
        and src.source_description_embedding == untouched
        and src.last_embedding_update_at is None
    ), "source with no new items must be left completely untouched"


@pytest.mark.asyncio
async def test_update_advances_last_embedding_update_at(mocker) -> None:
    mocker.patch(
        "news_service.tasks.update_source_embeddings.settings",
        SimpleNamespace(source_embedding_smoothing=0.9),
    )
    src = _source(embedding=[1.0, 0.0])
    _wire_session(mocker, [src], {src.id: [[0.0, 1.0]]})

    from news_service.tasks.update_source_embeddings import _update_all

    await _update_all()

    assert src.last_embedding_update_at is not None, (
        "last_embedding_update_at must be set so the next run only picks up newer items"
    )


@pytest.mark.asyncio
async def test_update_rejects_smoothing_outside_unit_interval(mocker) -> None:
    mocker.patch(
        "news_service.tasks.update_source_embeddings.settings",
        SimpleNamespace(source_embedding_smoothing=1.5),
    )

    from news_service.tasks.update_source_embeddings import _update_all

    with pytest.raises(ValueError, match="source_embedding_smoothing"):
        await _update_all()
