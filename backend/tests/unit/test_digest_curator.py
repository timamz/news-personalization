"""Tests for single-shot digest curation."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.digest_curator import (
    DigestCurationResult,
    _build_items_text,
    _cosine_similarity,
    _format_news_item,
    _is_russian_language,
    run_digest_curator,
)


def _make_news_item(
    headline: str = "Test Headline",
    body: str = "Test body content",
    url: str = "https://example.com/article",
    embedding: list[float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        headline=headline,
        body=body,
        url=url,
        published_at=None,
        fetched_at=datetime.now(UTC),
        embedding=embedding or [0.1] * 10,
    )


def test_format_news_item_includes_id_and_fields():
    item = _make_news_item()
    result = _format_news_item(item)
    assert f"[ID: {item.id}]" in result
    assert "Test Headline" in result
    assert "Test body content" in result
    assert "https://example.com/article" in result


def test_cosine_similarity_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_is_russian_language():
    assert _is_russian_language("ru") is True
    assert _is_russian_language("ru-RU") is True
    assert _is_russian_language("en") is False


def test_build_items_text_respects_budget():
    items = [_make_news_item(headline=f"Item {i}") for i in range(10)]
    single_formatted = _format_news_item(items[0])
    budget = len(single_formatted) * 3  # room for ~3 items
    result = _build_items_text(items, budget)
    assert result.count("[ID:") <= 3


def test_build_items_text_includes_all_when_budget_large():
    items = [_make_news_item(headline=f"Item {i}") for i in range(5)]
    result = _build_items_text(items, 1_000_000)
    assert result.count("[ID:") == 5


@pytest.mark.asyncio
async def test_run_digest_curator_returns_result(mocker):
    item_id = str(uuid.uuid4())
    expected = DigestCurationResult(
        digest_text="Here is your digest...",
        used_item_ids=[item_id],
    )

    mocker.patch(
        "news_service.agents.digest_curator._parse_digest",
        new=AsyncMock(return_value=expected),
    )

    items = [_make_news_item()]
    mocker.patch(
        "news_service.agents.digest_curator.find_similar_news",
        new=AsyncMock(return_value=items),
    )
    mock_session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await run_digest_curator(
        session=mock_session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_source_ids={uuid.uuid4()},
        published_after=datetime.now(UTC),
        format_instructions="brief summary",
        digest_language="en",
    )

    assert result is not None
    assert result.digest_text == "Here is your digest..."
    assert len(result.used_item_ids) == 1


@pytest.mark.asyncio
async def test_run_digest_curator_returns_none_when_no_candidates(mocker):
    mocker.patch(
        "news_service.agents.digest_curator.find_similar_news",
        new=AsyncMock(return_value=[]),
    )
    mock_session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await run_digest_curator(
        session=mock_session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_source_ids={uuid.uuid4()},
        published_after=datetime.now(UTC),
        format_instructions="brief summary",
        digest_language="en",
    )

    assert result is None
