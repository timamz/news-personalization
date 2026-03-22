import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.source_discovery import (
    ScoredSource,
    SourceDiscoveryResult,
    _create_source_discovery_agent,
    run_source_discovery,
    tool_search_web,
)

logging.disable(logging.CRITICAL)


def _random_embedding(dim: int = 10) -> list[float]:
    return [float(i * 0.1) for i in range(dim)]


@pytest.mark.asyncio
async def test_tool_search_web_returns_search_results(mocker) -> None:
    search_url = f"https://{uuid.uuid4().hex[:8]}.com/rss"
    mocker.patch(
        "news_service.agents.source_discovery.search_web",
        new=AsyncMock(return_value=f"Найдено: {search_url} - лента новостей ИИ"),
    )

    result = await tool_search_web.on_invoke_tool(
        MagicMock(), f'{{"query": "лучшие RSS ленты про ИИ {uuid.uuid4().hex[:4]}"}}'
    )
    assert search_url in result, "tool_search_web did not include search URL in result"


@pytest.mark.asyncio
async def test_create_agent_has_search_existing_sources_tool() -> None:
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, _random_embedding())
    tool_names = {t.name for t in agent.tools}
    assert "search_existing_sources" in tool_names, (
        "agent does not have search_existing_sources tool"
    )


@pytest.mark.asyncio
async def test_create_agent_has_tool_search_web_tool() -> None:
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, _random_embedding())
    tool_names = {t.name for t in agent.tools}
    assert "tool_search_web" in tool_names, "agent does not have tool_search_web tool"


@pytest.mark.asyncio
async def test_create_agent_has_validate_and_score_source_tool() -> None:
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, _random_embedding())
    tool_names = {t.name for t in agent.tools}
    assert "validate_and_score_source" in tool_names, (
        "agent does not have validate_and_score_source tool"
    )


@pytest.mark.asyncio
async def test_create_agent_has_exactly_three_tools() -> None:
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, _random_embedding())
    assert len(agent.tools) == 3, "agent does not have exactly three tools"


@pytest.mark.asyncio
async def test_create_agent_uses_structured_output() -> None:
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, _random_embedding())
    assert agent.output_type is SourceDiscoveryResult, (
        "agent does not use SourceDiscoveryResult as output type"
    )


@pytest.mark.asyncio
async def test_run_source_discovery_returns_agent_result(mocker) -> None:
    source_url = f"https://{uuid.uuid4().hex[:8]}.com"
    expected = SourceDiscoveryResult(
        sources=[
            ScoredSource(
                url=source_url,
                title="Источник А",
                source_kind="rss",
                relevance_score=0.9,
            )
        ]
    )

    mock_run_result = MagicMock()
    mock_run_result.final_output = expected

    mocker.patch(
        "news_service.agents.source_discovery.Runner.run",
        new=AsyncMock(return_value=mock_run_result),
    )

    session = AsyncMock()
    result = await run_source_discovery(
        session=session,
        raw_prompt=f"Новости ИИ {uuid.uuid4().hex[:4]}",
        prompt_embedding=_random_embedding(),
    )

    assert result == expected, "run_source_discovery did not return expected result"


@pytest.mark.asyncio
async def test_search_existing_sources_tool_includes_url(mocker) -> None:
    session = AsyncMock()
    prompt_embedding = _random_embedding()
    source_url = f"https://existing-{uuid.uuid4().hex[:6]}.com/feed"

    mock_source = SimpleNamespace(
        url=source_url,
        title="Существующий источник",
        source_description="Покрывает новости ИИ",
    )
    mocker.patch(
        "news_service.agents.source_discovery.embed_text",
        new=AsyncMock(return_value=[0.2] * 10),
    )
    mocker.patch(
        "news_service.agents.source_discovery.find_similar_sources",
        new=AsyncMock(return_value=[mock_source]),
    )

    agent = _create_source_discovery_agent(session, prompt_embedding)
    search_tool = next(t for t in agent.tools if t.name == "search_existing_sources")

    result = await search_tool.on_invoke_tool(
        MagicMock(), f'{{"query": "ИИ {uuid.uuid4().hex[:4]}"}}'
    )
    assert source_url in result, "search_existing_sources did not include source URL"


@pytest.mark.asyncio
async def test_search_existing_sources_tool_includes_title(mocker) -> None:
    session = AsyncMock()
    prompt_embedding = _random_embedding()
    title = f"Существующий-{uuid.uuid4().hex[:6]}"

    mock_source = SimpleNamespace(
        url="https://existing.com/feed",
        title=title,
        source_description="Покрывает новости ИИ",
    )
    mocker.patch(
        "news_service.agents.source_discovery.embed_text",
        new=AsyncMock(return_value=[0.2] * 10),
    )
    mocker.patch(
        "news_service.agents.source_discovery.find_similar_sources",
        new=AsyncMock(return_value=[mock_source]),
    )

    agent = _create_source_discovery_agent(session, prompt_embedding)
    search_tool = next(t for t in agent.tools if t.name == "search_existing_sources")

    result = await search_tool.on_invoke_tool(MagicMock(), '{"query": "ИИ"}')
    assert title in result, "search_existing_sources did not include source title"


@pytest.mark.asyncio
async def test_validate_and_score_source_tool_includes_score(mocker) -> None:
    session = AsyncMock()
    prompt_embedding = _random_embedding()

    mocker.patch(
        "news_service.agents.source_discovery.score_candidate",
        new=AsyncMock(return_value=(0.85, ["образец текста поста"])),
    )

    agent = _create_source_discovery_agent(session, prompt_embedding)
    score_tool = next(t for t in agent.tools if t.name == "validate_and_score_source")

    result = await score_tool.on_invoke_tool(
        MagicMock(),
        f'{{"url": "https://{uuid.uuid4().hex[:8]}.com/feed", "source_kind": "rss"}}',
    )
    assert "0.850" in result, "validate_and_score_source did not include score in result"
