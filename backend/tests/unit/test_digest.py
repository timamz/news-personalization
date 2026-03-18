import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents import digest
from news_service.agents.digest_curator import DigestCurationResult


def _make_curator_result(
    digest_text: str = "digest",
    item_ids: list[str] | None = None,
) -> DigestCurationResult:
    if item_ids is None:
        item_ids = [str(uuid.uuid4())]
    return DigestCurationResult(digest_text=digest_text, used_item_ids=item_ids)


@pytest.mark.asyncio
async def test_generate_digest_passes_params_to_curator(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id,)])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription_id = uuid.uuid4()
    subscription = SimpleNamespace(
        id=subscription_id,
        raw_prompt="AI lectures every morning",
        canonical_prompt="AI lectures every morning",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="AI lectures every morning",
        format_instructions="brief summary",
        digest_language="ru",
    )

    curator_result = _make_curator_result("ru digest")
    run_curator = AsyncMock(return_value=curator_result)
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=run_curator,
    )
    mark_as_sent = AsyncMock()
    mocker.patch.object(digest, "_mark_as_sent", new=mark_as_sent)

    result = await digest.generate_digest(session, subscription)

    assert result == "ru digest"
    call_kwargs = run_curator.await_args.kwargs
    assert call_kwargs["query_embedding"] == [0.0] * 1536
    assert call_kwargs["exclude_ids"] == set()
    assert call_kwargs["allowed_feed_ids"] == {source_feed_id}
    assert call_kwargs["format_instructions"] == "brief summary"
    assert call_kwargs["digest_language"] == "ru"
    mark_as_sent.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    empty_sources_result = SimpleNamespace(all=lambda: [])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, empty_sources_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI lectures every morning",
        canonical_prompt="AI lectures every morning",
        prompt_summary="AI lectures every morning",
        format_instructions="brief summary",
        digest_language="ru",
    )

    result = await digest.generate_digest(session, subscription)

    assert result is None


@pytest.mark.asyncio
async def test_generate_digest_falls_back_to_raw_prompt_embedding_when_missing(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id,)])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI lectures every morning",
        canonical_prompt="AI lectures every morning",
        canonical_prompt_embedding=None,
        prompt_summary="AI lectures every morning",
        format_instructions="brief summary",
        digest_language="ru",
    )

    embed_text = AsyncMock(return_value=[0.0] * 1536)
    mocker.patch.object(digest, "embed_text", new=embed_text)
    curator_result = _make_curator_result()
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(return_value=curator_result),
    )
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    result = await digest.generate_digest(session, subscription)

    assert result is not None
    embed_text.assert_awaited_once_with("AI lectures every morning")
    assert subscription.canonical_prompt_embedding == [0.0] * 1536


@pytest.mark.asyncio
async def test_generate_digest_uses_last_sent_at_as_cutoff(mocker) -> None:
    last_sent_at = datetime(2026, 3, 11, 9, 30, tzinfo=UTC)
    sent_news_id = uuid.uuid4()
    sent_result = SimpleNamespace(all=lambda: [(sent_news_id, last_sent_at)])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id,)])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="Latest ML papers",
        canonical_prompt="Latest ML papers",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="Latest ML papers",
        format_instructions="brief summary",
        digest_language="en",
    )

    curator_result = _make_curator_result()
    run_curator = AsyncMock(return_value=curator_result)
    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=run_curator,
    )
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    call_kwargs = run_curator.await_args.kwargs
    assert call_kwargs["published_after"] == last_sent_at
    assert call_kwargs["exclude_ids"] == {sent_news_id}


@pytest.mark.asyncio
async def test_generate_digest_returns_none_on_curator_failure(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id,)])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI news",
        canonical_prompt="AI news",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="AI news",
        format_instructions="brief summary",
        digest_language="en",
    )

    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(side_effect=RuntimeError("agent crashed")),
    )

    result = await digest.generate_digest(session, subscription)

    assert result is None


@pytest.mark.asyncio
async def test_generate_digest_returns_none_when_curator_finds_no_items(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id,)])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI news",
        canonical_prompt="AI news",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="AI news",
        format_instructions="brief summary",
        digest_language="en",
    )

    mocker.patch(
        "news_service.agents.digest_curator.run_digest_curator",
        new=AsyncMock(return_value=None),
    )

    result = await digest.generate_digest(session, subscription)

    assert result is None
