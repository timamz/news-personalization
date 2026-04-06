import logging
import random
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
    run_digest_curator,
)

logging.disable(logging.CRITICAL)


def _make_news_item(
    headline: str,
    body: str,
    url: str,
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
        embedding=embedding or [random.random() for _ in range(10)],
    )


@pytest.mark.parametrize(
    ("label", "vec_a", "vec_b", "expected"),
    [
        (
            "identical",
            [0.3, 0.7, 0.5],
            [0.3, 0.7, 0.5],
            1.0,
        ),
        (
            "orthogonal",
            [1.0, 0.0],
            [0.0, 1.0],
            0.0,
        ),
        (
            "zero_vector",
            [0.0, 0.0, 0.0],
            [0.4, 0.8, 0.1],
            0.0,
        ),
    ],
    ids=["identical_vectors_return_one", "orthogonal_vectors_return_zero", "zero_vector_returns_zero"],
)
def test_cosine_similarity_returns_expected_value(
    label: str,
    vec_a: list[float],
    vec_b: list[float],
    expected: float,
) -> None:
    assert _cosine_similarity(vec_a, vec_b) == pytest.approx(expected), (
        f"cosine_similarity did not return {expected} for {label} vectors"
    )


def test_build_items_text_respects_context_budget() -> None:
    items = [
        _make_news_item(
            f"Заголовок-{uuid.uuid4().hex[:6]}",
            f"Содержание-{uuid.uuid4().hex[:8]}",
            f"https://news-{uuid.uuid4().hex[:8]}.test/{i}",
        )
        for i in range(10)
    ]
    single_formatted = _format_news_item(items[0])
    budget = len(single_formatted) * 3

    result = _build_items_text(items, budget)

    assert result.count("[ID:") <= 3, "build_items_text did not respect the context budget"


def test_build_items_text_includes_all_items_when_budget_is_large() -> None:
    count = random.randint(3, 7)
    items = [
        _make_news_item(
            f"Новость-{uuid.uuid4().hex[:6]}",
            f"Текст-{uuid.uuid4().hex[:8]}",
            f"https://news-{uuid.uuid4().hex[:8]}.test/{i}",
        )
        for i in range(count)
    ]

    result = _build_items_text(items, 1_000_000)

    assert result.count("[ID:") == count, (
        "build_items_text did not include all items when budget was large"
    )


@pytest.mark.asyncio
async def test_run_digest_curator_returns_digest_text(mocker) -> None:
    item_id = str(uuid.uuid4())
    digest_body = f"Ваш дайджест новостей {uuid.uuid4().hex[:8]}"
    expected = DigestCurationResult(digest_text=digest_body, used_item_ids=[item_id])
    mocker.patch(
        "news_service.agents.digest_curator._parse_digest",
        new=AsyncMock(return_value=expected),
    )
    items = [
        _make_news_item(
            f"Заголовок-{uuid.uuid4().hex[:6]}",
            f"Текст-{uuid.uuid4().hex[:8]}",
            f"https://a-{uuid.uuid4().hex[:8]}.test/1",
        )
    ]
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
        query_embedding=[random.random() for _ in range(10)],
        exclude_ids=set(),
        allowed_source_ids={uuid.uuid4()},
        published_after=datetime.now(UTC),
        format_instructions=f"формат-{uuid.uuid4().hex[:6]}",
        digest_language="ru",
    )

    assert result is not None and result.digest_text == digest_body, (
        "run_digest_curator did not return the expected digest text"
    )


@pytest.mark.asyncio
async def test_run_digest_curator_returns_used_item_ids(mocker) -> None:
    item_id = str(uuid.uuid4())
    expected = DigestCurationResult(
        digest_text=f"Дайджест {uuid.uuid4().hex[:6]}", used_item_ids=[item_id]
    )
    mocker.patch(
        "news_service.agents.digest_curator._parse_digest",
        new=AsyncMock(return_value=expected),
    )
    items = [
        _make_news_item(
            f"Заголовок-{uuid.uuid4().hex[:6]}",
            f"Текст-{uuid.uuid4().hex[:8]}",
            f"https://b-{uuid.uuid4().hex[:8]}.test/1",
        )
    ]
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
        query_embedding=[random.random() for _ in range(10)],
        exclude_ids=set(),
        allowed_source_ids={uuid.uuid4()},
        published_after=datetime.now(UTC),
        format_instructions=f"формат-{uuid.uuid4().hex[:6]}",
        digest_language="en",
    )

    assert result is not None and len(result.used_item_ids) == 1, (
        "run_digest_curator did not return exactly one used item ID"
    )


@pytest.mark.asyncio
async def test_run_digest_curator_returns_none_when_no_candidates(mocker) -> None:
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
        query_embedding=[random.random() for _ in range(10)],
        exclude_ids=set(),
        allowed_source_ids={uuid.uuid4()},
        published_after=datetime.now(UTC),
        format_instructions=f"формат-{uuid.uuid4().hex[:6]}",
        digest_language="en",
    )

    assert result is None, "run_digest_curator did not return None when no candidates exist"
