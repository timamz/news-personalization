import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents import digest
from news_service.agents.digest_curator import DigestCurationResult

logging.disable(logging.CRITICAL)


def _make_curator_result(digest_text: str, item_ids: list[str]) -> DigestCurationResult:
    return DigestCurationResult(digest_text=digest_text, used_item_ids=item_ids)


def _make_subscription(
    prompt: str,
    embedding: list[float] | None,
    format_instructions: str,
    digest_language: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt=prompt,
        canonical_prompt=prompt,
        canonical_prompt_embedding=embedding,
        prompt_summary=f"Краткое: {prompt[:20]}",
        format_instructions=format_instructions,
        digest_language=digest_language,
    )


def _make_session_with_sources(
    source_ids: list[uuid.UUID],
    sent_rows: list[tuple] | None = None,
) -> SimpleNamespace:
    sent_result = SimpleNamespace(all=lambda: sent_rows or [])
    source_result = SimpleNamespace(all=lambda: [(sid,) for sid in source_ids])
    return SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))


@pytest.mark.asyncio
async def test_generate_digest_returns_curator_digest_text(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Лекции об ИИ каждое утро {uuid.uuid4().hex[:6]}"
    digest_text = f"Дайджест на русском {uuid.uuid4().hex[:8]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "ru")
    curator_result = _make_curator_result(digest_text, [str(uuid.uuid4())])
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(return_value=curator_result),
    )
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    result = await digest.generate_digest(session, subscription)

    assert result == digest_text, "generate_digest did not return the curator digest text"


@pytest.mark.asyncio
async def test_generate_digest_passes_embedding_to_curator(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Научные статьи {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "подробный анализ", "ru")
    run_curator = AsyncMock(
        return_value=_make_curator_result(f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
    )
    mocker.patch("news_service.agents.digest_curator.run_digest_curator", new=run_curator)
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert run_curator.await_args.kwargs["query_embedding"] == embedding, (
        "generate_digest did not pass the subscription embedding to curator"
    )


@pytest.mark.asyncio
async def test_generate_digest_passes_allowed_source_ids_to_curator(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Новости ML {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "en")
    run_curator = AsyncMock(
        return_value=_make_curator_result(f"Text {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
    )
    mocker.patch("news_service.agents.digest_curator.run_digest_curator", new=run_curator)
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert run_curator.await_args.kwargs["allowed_source_ids"] == {source_id}, (
        "generate_digest did not pass the correct allowed_source_ids to curator"
    )


@pytest.mark.asyncio
async def test_generate_digest_passes_format_instructions_to_curator(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    format_instr = f"детальный обзор {uuid.uuid4().hex[:6]}"
    prompt = f"Аналитика {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, format_instr, "ru")
    run_curator = AsyncMock(
        return_value=_make_curator_result(f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
    )
    mocker.patch("news_service.agents.digest_curator.run_digest_curator", new=run_curator)
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert run_curator.await_args.kwargs["format_instructions"] == format_instr, (
        "generate_digest did not pass format_instructions to curator"
    )


@pytest.mark.asyncio
async def test_generate_digest_passes_digest_language_to_curator(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Обзор технологий {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "ru")
    run_curator = AsyncMock(
        return_value=_make_curator_result(f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
    )
    mocker.patch("news_service.agents.digest_curator.run_digest_curator", new=run_curator)
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert run_curator.await_args.kwargs["digest_language"] == "ru", (
        "generate_digest did not pass digest_language to curator"
    )


@pytest.mark.asyncio
async def test_generate_digest_calls_mark_as_sent_after_curator(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Новости {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "en")
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(
            return_value=_make_curator_result(f"Text {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
        ),
    )
    mark_as_sent = AsyncMock()
    mocker.patch.object(digest, "_mark_as_sent", new=mark_as_sent)

    await digest.generate_digest(session, subscription)

    assert mark_as_sent.await_count == 1, (
        "generate_digest did not call _mark_as_sent after curator returned"
    )


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources(mocker) -> None:
    session = _make_session_with_sources([])
    prompt = f"Лекции ИИ {uuid.uuid4().hex[:6]}"
    subscription = _make_subscription(prompt, [0.0] * 1536, "краткая сводка", "ru")

    result = await digest.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None when no fixed sources exist"


@pytest.mark.asyncio
async def test_generate_digest_computes_embedding_when_missing(mocker) -> None:
    source_id = uuid.uuid4()
    prompt = f"Лекции по физике {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, None, "краткая сводка", "ru")
    computed_embedding = [random.random() for _ in range(1536)]
    embed_text = AsyncMock(return_value=computed_embedding)
    mocker.patch.object(digest, "embed_text", new=embed_text)
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(
            return_value=_make_curator_result(f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
        ),
    )
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert embed_text.await_count == 1, (
        "generate_digest did not call embed_text when embedding was missing"
    )


@pytest.mark.asyncio
async def test_generate_digest_stores_computed_embedding_on_subscription(mocker) -> None:
    source_id = uuid.uuid4()
    prompt = f"Квантовые вычисления {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, None, "краткая сводка", "ru")
    computed_embedding = [random.random() for _ in range(1536)]
    mocker.patch.object(digest, "embed_text", new=AsyncMock(return_value=computed_embedding))
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(
            return_value=_make_curator_result(f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
        ),
    )
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert subscription.canonical_prompt_embedding == computed_embedding, (
        "generate_digest did not store computed embedding on subscription"
    )


@pytest.mark.asyncio
async def test_generate_digest_uses_last_sent_at_as_published_after(mocker) -> None:
    source_id = uuid.uuid4()
    last_sent_at = datetime(2026, 3, 11, 9, 30, tzinfo=UTC)
    sent_news_id = uuid.uuid4()
    session = _make_session_with_sources([source_id], sent_rows=[(sent_news_id, last_sent_at)])
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Статьи по ML {uuid.uuid4().hex[:6]}"
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "en")
    run_curator = AsyncMock(
        return_value=_make_curator_result(f"Text {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
    )
    mocker.patch("news_service.agents.digest_curator.run_digest_curator", new=run_curator)
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert run_curator.await_args.kwargs["published_after"] == last_sent_at, (
        "generate_digest did not pass last_sent_at as published_after to curator"
    )


@pytest.mark.asyncio
async def test_generate_digest_excludes_previously_sent_item_ids(mocker) -> None:
    source_id = uuid.uuid4()
    sent_news_id = uuid.uuid4()
    last_sent_at = datetime(2026, 3, 11, 9, 30, tzinfo=UTC)
    session = _make_session_with_sources([source_id], sent_rows=[(sent_news_id, last_sent_at)])
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Обзоры {uuid.uuid4().hex[:6]}"
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "en")
    run_curator = AsyncMock(
        return_value=_make_curator_result(f"Text {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])
    )
    mocker.patch("news_service.agents.digest_curator.run_digest_curator", new=run_curator)
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert run_curator.await_args.kwargs["exclude_ids"] == {sent_news_id}, (
        "generate_digest did not exclude previously sent item IDs"
    )


@pytest.mark.asyncio
async def test_generate_digest_returns_none_on_curator_failure(mocker) -> None:
    source_id = uuid.uuid4()
    session = _make_session_with_sources([source_id])
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Новости ИИ {uuid.uuid4().hex[:6]}"
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "en")
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(side_effect=RuntimeError()),
    )

    result = await digest.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None on curator failure"


@pytest.mark.asyncio
async def test_generate_digest_returns_none_when_curator_finds_no_items(mocker) -> None:
    source_id = uuid.uuid4()
    session = _make_session_with_sources([source_id])
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Новости ИИ {uuid.uuid4().hex[:6]}"
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "en")
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(return_value=None),
    )

    result = await digest.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None when curator finds no items"
