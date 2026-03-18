"""Tests for the conversational subscription parser agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationChoice,
    FinalizedSubscriptionConfig,
)


@pytest.mark.asyncio
async def test_validate_cron_tool_success(mocker):
    mocker.patch(
        "news_service.agents.subscription_parser.parse_schedule_preference",
        new=AsyncMock(return_value="0 8 * * *"),
    )
    from news_service.agents.subscription_parser import validate_cron

    result = await validate_cron.on_invoke_tool(MagicMock(), '{"schedule_text": "every morning"}')
    assert "0 8 * * *" in result


@pytest.mark.asyncio
async def test_validate_cron_tool_failure(mocker):
    mocker.patch(
        "news_service.agents.subscription_parser.parse_schedule_preference",
        new=AsyncMock(side_effect=ValueError("bad input")),
    )
    from news_service.agents.subscription_parser import validate_cron

    result = await validate_cron.on_invoke_tool(MagicMock(), '{"schedule_text": "????"}')
    assert "Failed" in result


@pytest.mark.asyncio
async def test_parse_sources_from_text_tool_finds_sources():
    from news_service.agents.subscription_parser import parse_sources_from_text

    result = await parse_sources_from_text.on_invoke_tool(
        MagicMock(),
        '{"text": "Follow @durov_russia and r/technology and https://x.com/elonmusk"}',
    )
    assert "durov_russia" in result
    assert "technology" in result
    assert "elonmusk" in result


@pytest.mark.asyncio
async def test_parse_sources_from_text_tool_no_sources():
    from news_service.agents.subscription_parser import parse_sources_from_text

    result = await parse_sources_from_text.on_invoke_tool(
        MagicMock(), '{"text": "just some plain text"}'
    )
    assert "No sources found" in result


@pytest.mark.asyncio
async def test_validate_source_url_tool_valid(mocker):
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )
    from news_service.agents.subscription_parser import validate_source_url

    result = await validate_source_url.on_invoke_tool(
        MagicMock(),
        '{"url": "https://t.me/s/durov", "source_kind": "telegram_channel"}',
    )
    assert "valid" in result


@pytest.mark.asyncio
async def test_validate_source_url_tool_invalid(mocker):
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=False),
    )
    from news_service.agents.subscription_parser import validate_source_url

    result = await validate_source_url.on_invoke_tool(
        MagicMock(),
        '{"url": "https://t.me/s/nonexistent", "source_kind": "telegram_channel"}',
    )
    assert "could not fetch" in result


@pytest.mark.asyncio
async def test_create_agent_has_correct_tools():
    from news_service.agents.subscription_parser import _create_subscription_parser_agent

    agent = _create_subscription_parser_agent(user_language="en", user_timezone="UTC")
    tool_names = {t.name for t in agent.tools}
    assert "validate_cron" in tool_names
    assert "parse_sources_from_text" in tool_names
    assert "validate_source_url" in tool_names
    assert len(agent.tools) == 3


@pytest.mark.asyncio
async def test_create_agent_uses_structured_output():
    from news_service.agents.subscription_parser import _create_subscription_parser_agent

    agent = _create_subscription_parser_agent()
    assert agent.output_type is AgentTurnOutput


@pytest.mark.asyncio
async def test_create_agent_includes_context_in_instructions():
    from news_service.agents.subscription_parser import _create_subscription_parser_agent

    agent = _create_subscription_parser_agent(user_language="ru", user_timezone="Europe/Moscow")
    assert "ru" in agent.instructions
    assert "Europe/Moscow" in agent.instructions


@pytest.mark.asyncio
async def test_run_conversation_turn_returns_agent_output(mocker):
    expected = AgentTurnOutput(
        message="What kind of news are you interested in?",
        status="in_progress",
        choices=[
            ConversationChoice(label="Digest", value="digest"),
            ConversationChoice(label="Events", value="events"),
        ],
        finalized_config=None,
    )

    mock_result = MagicMock()
    mock_result.final_output = expected

    mocker.patch(
        "news_service.agents.subscription_parser.Runner.run",
        new=AsyncMock(return_value=mock_result),
    )

    from news_service.agents.subscription_parser import run_conversation_turn

    result = await run_conversation_turn(
        [{"role": "user", "content": "AI news"}],
        user_language="en",
    )
    assert result == expected
    assert result.status == "in_progress"
    assert result.finalized_config is None


@pytest.mark.asyncio
async def test_run_conversation_turn_returns_finalized(mocker):
    config = FinalizedSubscriptionConfig(
        prompt_summary="AI news daily digest",
        short_label="AI News",
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        manual_only=False,
        format_instructions="brief summary",
        digest_language="en",
        include_discovered_sources=True,
    )
    expected = AgentTurnOutput(
        message="Your subscription is ready!",
        status="ready",
        choices=None,
        finalized_config=config,
    )

    mock_result = MagicMock()
    mock_result.final_output = expected

    mocker.patch(
        "news_service.agents.subscription_parser.Runner.run",
        new=AsyncMock(return_value=mock_result),
    )

    from news_service.agents.subscription_parser import run_conversation_turn

    result = await run_conversation_turn(
        [
            {"role": "user", "content": "AI news every morning"},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": "Yes, discover sources for me"},
        ],
        user_language="en",
        user_timezone="UTC",
    )
    assert result.status == "ready"
    assert result.finalized_config is not None
    assert result.finalized_config.schedule_cron == "0 8 * * *"
