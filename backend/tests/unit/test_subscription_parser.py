"""Tests for the conversational subscription parser."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.conversation import (
    AgentTurnOutput,
    FinalizedSubscriptionConfig,
)


def _mock_parsed_response(output: AgentTurnOutput) -> MagicMock:
    """Create a mock API response where the model produces structured output (no tool calls)."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.parsed = output
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_tool_call_response(tool_name: str, arguments: str, call_id: str = "call_1") -> MagicMock:
    """Create a mock API response where the model requests a tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = arguments

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    msg.parsed = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_execute_tool_validate_source_url_valid(mocker):
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )
    from news_service.agents.subscription_parser import _execute_tool

    result = await _execute_tool(
        "validate_source_url",
        '{"url": "https://t.me/s/durov", "source_kind": "telegram_channel"}',
    )
    assert "valid" in result


@pytest.mark.asyncio
async def test_execute_tool_validate_source_url_invalid(mocker):
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=False),
    )
    from news_service.agents.subscription_parser import _execute_tool

    result = await _execute_tool(
        "validate_source_url",
        '{"url": "https://t.me/s/nonexistent", "source_kind": "telegram_channel"}',
    )
    assert "could not fetch" in result


@pytest.mark.asyncio
async def test_execute_tool_unknown():
    from news_service.agents.subscription_parser import _execute_tool

    result = await _execute_tool("unknown_tool", "{}")
    assert "Unknown tool" in result


@pytest.mark.asyncio
async def test_run_conversation_turn_returns_output(mocker):
    expected = AgentTurnOutput(
        message="What kind of news are you interested in?",
        status="in_progress",
    )

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(
        return_value=_mock_parsed_response(expected),
    )

    with patch("news_service.agents.subscription_parser._client", mock_client):
        from news_service.agents.subscription_parser import run_conversation_turn

        output, new_messages = await run_conversation_turn(
            [{"role": "user", "content": "AI news"}],
            user_language="en",
        )

    assert output == expected
    assert output.status == "in_progress"
    assert len(new_messages) == 1
    assert new_messages[0]["role"] == "assistant"
    assert new_messages[0]["content"] == expected.message


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
        finalized_config=config,
    )

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(
        return_value=_mock_parsed_response(expected),
    )

    with patch("news_service.agents.subscription_parser._client", mock_client):
        from news_service.agents.subscription_parser import run_conversation_turn

        output, new_messages = await run_conversation_turn(
            [
                {"role": "user", "content": "AI news every morning"},
                {"role": "assistant", "content": "Got it!"},
                {"role": "user", "content": "Yes, discover sources for me"},
            ],
            user_language="en",
            user_timezone="UTC",
        )

    assert output.status == "ready"
    assert output.finalized_config is not None
    assert output.finalized_config.schedule_cron == "0 8 * * *"
    assert len(new_messages) == 1


@pytest.mark.asyncio
async def test_run_conversation_turn_handles_tool_calls(mocker):
    """When the model calls validate_source_url, the tool is executed and
    the result is fed back before the model produces its final output."""
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )

    final_output = AgentTurnOutput(
        message="The channel is valid! I'll add it.",
        status="in_progress",
    )

    # First call: model requests tool call
    tool_response = _mock_tool_call_response(
        "validate_source_url",
        '{"url": "https://t.me/s/durov", "source_kind": "telegram_channel"}',
    )
    # Second call: model produces structured output
    final_response = _mock_parsed_response(final_output)

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(
        side_effect=[tool_response, final_response],
    )

    with patch("news_service.agents.subscription_parser._client", mock_client):
        from news_service.agents.subscription_parser import run_conversation_turn

        output, new_messages = await run_conversation_turn(
            [{"role": "user", "content": "Add @durov channel"}],
            user_language="en",
        )

    assert output == final_output
    # new_messages: tool call assistant msg, tool result, final assistant msg
    assert len(new_messages) == 3
    assert new_messages[0]["role"] == "assistant"
    assert "tool_calls" in new_messages[0]
    assert new_messages[1]["role"] == "tool"
    assert "valid" in new_messages[1]["content"]
    assert new_messages[2]["role"] == "assistant"
    assert new_messages[2]["content"] == "The channel is valid! I'll add it."

    # API was called twice
    assert mock_client.beta.chat.completions.parse.await_count == 2


@pytest.mark.asyncio
async def test_run_conversation_turn_raises_on_empty_response():
    msg = MagicMock()
    msg.tool_calls = None
    msg.parsed = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]

    mock_client = AsyncMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=response)

    with (
        patch("news_service.agents.subscription_parser._client", mock_client),
        pytest.raises(ValueError, match="empty response"),
    ):
        from news_service.agents.subscription_parser import run_conversation_turn

        await run_conversation_turn([{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_system_prompt_includes_context():
    from news_service.agents.subscription_parser import _build_system_prompt

    prompt = _build_system_prompt(user_language="ru", user_timezone="Europe/Moscow")
    assert "ru" in prompt
    assert "Europe/Moscow" in prompt


@pytest.mark.asyncio
async def test_system_prompt_includes_cron_examples():
    from news_service.agents.subscription_parser import _build_system_prompt

    prompt = _build_system_prompt(user_language=None, user_timezone=None)
    assert "0 8 * * *" in prompt
    assert "0 9 * * 1-5" in prompt


def test_tool_definitions_has_validate_source_url():
    from news_service.agents.subscription_parser import TOOL_DEFINITIONS

    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert names == ["validate_source_url"]
