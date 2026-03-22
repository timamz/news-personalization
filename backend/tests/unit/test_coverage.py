import logging
import random
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.source_discovery import ScoredSource, SourceDiscoveryResult
from news_service.services import coverage

logging.disable(logging.CRITICAL)


def _make_db_source(url: str, title: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        url=url,
        title=title,
        subscriber_count=random.randint(0, 100),
        is_active=True,
        source_description=f"Описание источника {uuid.uuid4().hex[:6]}",
        source_description_embedding=[random.random()] * 10,
    )


def _make_scored_source(
    url: str,
    title: str,
    source_kind: str,
    score: float,
) -> ScoredSource:
    return ScoredSource(url=url, title=title, source_kind=source_kind, relevance_score=score)


def _make_embedding() -> list[float]:
    return [random.random() for _ in range(10)]


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_returns_correct_count_for_two_agent_sources(mocker) -> None:
    session = AsyncMock()
    url_a = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    url_b = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    source_a = _make_scored_source(url_a, "Источник А", "rss", 0.9)
    source_b = _make_scored_source(url_b, "Источник Б", "rss", 0.7)
    agent_result = SourceDiscoveryResult(sources=[source_a, source_b])
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(return_value=agent_result),
    )
    db_source_a = _make_db_source(url_a, "Источник А")
    db_source_b = _make_db_source(url_b, "Источник Б")
    mocker.patch.object(
        coverage, "_register_or_reuse_source", new=AsyncMock(side_effect=[db_source_a, db_source_b])
    )

    result = await coverage.ensure_prompt_coverage(
        session, f"Запрос-{uuid.uuid4().hex[:6]}", _make_embedding()
    )

    assert len(result) == 2, "coverage did not return exactly two sources for two agent results"


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_returns_first_agent_source_in_order(mocker) -> None:
    session = AsyncMock()
    url_a = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    url_b = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    source_a = _make_scored_source(url_a, "Первый", "rss", 0.9)
    source_b = _make_scored_source(url_b, "Второй", "rss", 0.7)
    agent_result = SourceDiscoveryResult(sources=[source_a, source_b])
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(return_value=agent_result),
    )
    db_source_a = _make_db_source(url_a, "Первый")
    db_source_b = _make_db_source(url_b, "Второй")
    mocker.patch.object(
        coverage, "_register_or_reuse_source", new=AsyncMock(side_effect=[db_source_a, db_source_b])
    )

    result = await coverage.ensure_prompt_coverage(
        session, f"Запрос-{uuid.uuid4().hex[:6]}", _make_embedding()
    )

    assert result[0] == db_source_a, "coverage did not return the first agent source in position 0"


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_calls_register_for_each_agent_source(mocker) -> None:
    session = AsyncMock()
    url_a = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    url_b = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    source_a = _make_scored_source(url_a, "Фид А", "rss", 0.85)
    source_b = _make_scored_source(url_b, "Фид Б", "rss", 0.65)
    agent_result = SourceDiscoveryResult(sources=[source_a, source_b])
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(return_value=agent_result),
    )
    register_mock = AsyncMock(
        side_effect=[
            _make_db_source(url_a, "Фид А"),
            _make_db_source(url_b, "Фид Б"),
        ]
    )
    mocker.patch.object(coverage, "_register_or_reuse_source", register_mock)

    await coverage.ensure_prompt_coverage(
        session, f"Запрос-{uuid.uuid4().hex[:6]}", _make_embedding()
    )

    assert register_mock.await_count == 2, "coverage did not call register for each agent source"


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_returns_empty_list_on_agent_failure(mocker) -> None:
    session = AsyncMock()
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(side_effect=RuntimeError()),
    )

    result = await coverage.ensure_prompt_coverage(
        session, f"Запрос-{uuid.uuid4().hex[:6]}", _make_embedding()
    )

    assert result == [], "coverage did not return empty list on agent failure"


@pytest.mark.asyncio
async def test_ensure_prompt_coverage_returns_empty_list_when_agent_finds_no_sources(
    mocker,
) -> None:
    session = AsyncMock()
    mocker.patch(
        "news_service.agents.source_discovery.run_source_discovery",
        new=AsyncMock(return_value=SourceDiscoveryResult(sources=[])),
    )

    result = await coverage.ensure_prompt_coverage(
        session, f"Запрос-{uuid.uuid4().hex[:6]}", _make_embedding()
    )

    assert result == [], "coverage did not return empty list when agent finds no sources"


@pytest.mark.asyncio
async def test_register_or_reuse_source_reuses_existing_source(mocker) -> None:
    session = AsyncMock()
    url = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    existing = _make_db_source(url, "Существующий фид")
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing
    session.execute = AsyncMock(return_value=mock_result)


    scored = _make_scored_source(url, "Существующий фид", "rss", 0.8)
    result = await coverage._register_or_reuse_source(session, scored)

    assert result == existing, "register did not reuse existing source from DB"


@pytest.mark.asyncio
async def test_register_or_reuse_source_increments_subscriber_count(mocker) -> None:
    session = AsyncMock()
    url = f"https://feed-{uuid.uuid4().hex[:8]}.test/rss"
    initial_count = random.randint(1, 50)
    existing = _make_db_source(url, "Счётчик фид")
    existing.subscriber_count = initial_count
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing
    session.execute = AsyncMock(return_value=mock_result)


    scored = _make_scored_source(url, "Счётчик фид", "rss", 0.75)
    await coverage._register_or_reuse_source(session, scored)

    assert existing.subscriber_count == initial_count + 1, (
        "register did not increment subscriber count for existing source"
    )


@pytest.mark.asyncio
async def test_register_or_reuse_source_creates_new_source_when_url_not_in_db(mocker) -> None:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    mocker.patch.object(
        coverage,
        "_build_source_profile",
        new=AsyncMock(return_value=(f"Описание-{uuid.uuid4().hex[:6]}", [0.1] * 10)),
    )

    url = f"https://new-{uuid.uuid4().hex[:8]}.test/feed"
    scored = _make_scored_source(url, f"Новый фид {uuid.uuid4().hex[:4]}", "rss", 0.9)
    result = await coverage._register_or_reuse_source(session, scored)

    assert result is not None, "register did not create new source when URL not in DB"


@pytest.mark.asyncio
async def test_register_or_reuse_source_calls_session_add_for_new_source(mocker) -> None:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    mocker.patch.object(
        coverage,
        "_build_source_profile",
        new=AsyncMock(return_value=(f"Описание-{uuid.uuid4().hex[:6]}", [0.1] * 10)),
    )

    url = f"https://new-{uuid.uuid4().hex[:8]}.test/feed"
    scored = _make_scored_source(url, f"Новый фид {uuid.uuid4().hex[:4]}", "rss", 0.9)
    await coverage._register_or_reuse_source(session, scored)

    assert session.add.called, "register did not call session.add for new source"


@pytest.mark.asyncio
async def test_register_or_reuse_source_flushes_session_for_new_source(mocker) -> None:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    mocker.patch.object(
        coverage,
        "_build_source_profile",
        new=AsyncMock(return_value=(f"Описание-{uuid.uuid4().hex[:6]}", [0.1] * 10)),
    )

    url = f"https://new-{uuid.uuid4().hex[:8]}.test/feed"
    scored = _make_scored_source(url, f"Новый фид {uuid.uuid4().hex[:4]}", "rss", 0.9)
    await coverage._register_or_reuse_source(session, scored)

    assert session.flush.await_count == 1, (
        "register did not flush session after creating new source"
    )
