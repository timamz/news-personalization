import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.subscription_parser import (
    _execute_tool,
    run_conversation_turn,
    run_conversation_turn_streaming,
)
from news_service.schemas.conversation import (
    AgentTurnOutput,
    FinalizedSubscriptionConfig,
)

logging.disable(logging.CRITICAL)

_CHAT_COMPLETION_PATH = "news_service.agents.subscription_parser.chat_completion"


def _mock_parsed_response(output: AgentTurnOutput) -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = None
    msg.parsed = output
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_tool_call_response(
    tool_name: str, arguments: str, call_id: str | None = None
) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id or f"call_{uuid.uuid4().hex[:8]}"
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
async def test_execute_tool_validate_source_url_valid(mocker) -> None:
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )
    result = await _execute_tool(
        "validate_source_url",
        f'{{"url": "https://t.me/s/{uuid.uuid4().hex[:8]}", "source_kind": "telegram_channel"}}',
    )
    assert "valid" in result, "_execute_tool did not return valid indicator for valid source"


@pytest.mark.asyncio
async def test_execute_tool_validate_source_url_invalid(mocker) -> None:
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=False),
    )
    result = await _execute_tool(
        "validate_source_url",
        f'{{"url": "https://t.me/s/{uuid.uuid4().hex[:8]}", "source_kind": "telegram_channel"}}',
    )
    assert "could not fetch" in result, (
        "_execute_tool did not return failure indicator for invalid source"
    )


@pytest.mark.asyncio
async def test_execute_tool_unknown_returns_error() -> None:
    result = await _execute_tool(f"unknown_tool_{uuid.uuid4().hex[:4]}", "{}")
    assert "Unknown tool" in result, "_execute_tool did not return error for unknown tool name"


@pytest.mark.asyncio
async def test_run_conversation_turn_returns_expected_output(mocker) -> None:
    message_text = f"Какие новости вас интересуют? {uuid.uuid4().hex[:4]}"
    expected = AgentTurnOutput(message=message_text, status="in_progress")

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(return_value=_mock_parsed_response(expected)),
    )

    output, _new_messages = await run_conversation_turn(
        [{"role": "user", "content": f"Новости ИИ {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    )
    assert output == expected, "run_conversation_turn did not return expected output"


@pytest.mark.asyncio
async def test_run_conversation_turn_returns_one_new_message(mocker) -> None:
    expected = AgentTurnOutput(message=f"Вопрос {uuid.uuid4().hex[:4]}", status="in_progress")

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(return_value=_mock_parsed_response(expected)),
    )

    _output, new_messages = await run_conversation_turn(
        [{"role": "user", "content": f"Новости ИИ {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    )
    assert len(new_messages) == 1, "run_conversation_turn did not return exactly one new message"


@pytest.mark.asyncio
async def test_run_conversation_turn_finalized_has_ready_status(mocker) -> None:
    config = FinalizedSubscriptionConfig(
        prompt_summary=f"Дайджест новостей ИИ {uuid.uuid4().hex[:4]}",
        short_label="ИИ Новости",
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        manual_only=False,
        format_instructions="краткое описание",
        digest_language="ru",
        include_discovered_sources=True,
    )
    expected = AgentTurnOutput(
        message="Ваша подписка готова!", status="ready", finalized_config=config
    )

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(return_value=_mock_parsed_response(expected)),
    )

    output, _new_messages = await run_conversation_turn(
        [
            {"role": "user", "content": f"Новости ИИ каждое утро {uuid.uuid4().hex[:4]}"},
            {"role": "assistant", "content": "Понял!"},
            {"role": "user", "content": "Да, найдите источники за меня"},
        ],
        user_language="ru",
        user_timezone="Europe/Moscow",
    )
    assert output.status == "ready", (
        "run_conversation_turn did not return ready status for finalized config"
    )


@pytest.mark.asyncio
async def test_run_conversation_turn_finalized_has_cron(mocker) -> None:
    config = FinalizedSubscriptionConfig(
        prompt_summary=f"Дайджест {uuid.uuid4().hex[:4]}",
        short_label="ИИ",
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        manual_only=False,
        format_instructions="краткое описание",
        digest_language="ru",
        include_discovered_sources=True,
    )
    expected = AgentTurnOutput(message="Готово!", status="ready", finalized_config=config)

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(return_value=_mock_parsed_response(expected)),
    )

    output, _new_messages = await run_conversation_turn(
        [{"role": "user", "content": f"Каждое утро в 8 {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
        user_timezone="Europe/Moscow",
    )
    assert output.finalized_config.schedule_cron == "0 8 * * *", (
        "run_conversation_turn did not return correct schedule_cron in finalized config"
    )


@pytest.mark.asyncio
async def test_run_conversation_turn_handles_tool_calls_and_returns_final_output(mocker) -> None:
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )

    final_message = f"Канал проверен! {uuid.uuid4().hex[:4]}"
    final_output = AgentTurnOutput(message=final_message, status="in_progress")

    tool_response = _mock_tool_call_response(
        "validate_source_url",
        f'{{"url": "https://t.me/s/{uuid.uuid4().hex[:8]}", "source_kind": "telegram_channel"}}',
    )
    final_response = _mock_parsed_response(final_output)

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(side_effect=[tool_response, final_response]),
    )

    output, _new_messages = await run_conversation_turn(
        [{"role": "user", "content": f"Добавьте канал @durov {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    )
    assert output == final_output, (
        "run_conversation_turn did not return final output after tool call"
    )


@pytest.mark.asyncio
async def test_run_conversation_turn_tool_call_generates_three_messages(mocker) -> None:
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )

    final_output = AgentTurnOutput(message="Канал проверен!", status="in_progress")
    tool_response = _mock_tool_call_response(
        "validate_source_url",
        f'{{"url": "https://t.me/s/{uuid.uuid4().hex[:8]}", "source_kind": "telegram_channel"}}',
    )
    final_response = _mock_parsed_response(final_output)

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(side_effect=[tool_response, final_response]),
    )

    _output, new_messages = await run_conversation_turn(
        [{"role": "user", "content": f"Добавьте канал {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    )
    assert len(new_messages) == 3, (
        "run_conversation_turn did not produce three messages for tool call flow"
    )


@pytest.mark.asyncio
async def test_run_conversation_turn_raises_on_empty_response(mocker) -> None:
    msg = MagicMock()
    msg.tool_calls = None
    msg.parsed = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]

    mocker.patch(_CHAT_COMPLETION_PATH, new=AsyncMock(return_value=response))

    with pytest.raises(ValueError):
        await run_conversation_turn([{"role": "user", "content": f"тест {uuid.uuid4().hex[:4]}"}])


@pytest.mark.asyncio
async def test_streaming_yields_done_event(mocker) -> None:
    message_text = f"Какие новости? {uuid.uuid4().hex[:4]}"
    expected = AgentTurnOutput(message=message_text, status="in_progress")

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(return_value=_mock_parsed_response(expected)),
    )

    events = []
    async for ev in run_conversation_turn_streaming(
        [{"role": "user", "content": f"Новости ИИ {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    ):
        events.append(ev)

    assert events[0]["event"] == "done", "streaming did not yield a done event"


@pytest.mark.asyncio
async def test_streaming_done_event_contains_message(mocker) -> None:
    message_text = f"Какие новости? {uuid.uuid4().hex[:4]}"
    expected = AgentTurnOutput(message=message_text, status="in_progress")

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(return_value=_mock_parsed_response(expected)),
    )

    events = []
    async for ev in run_conversation_turn_streaming(
        [{"role": "user", "content": f"Новости ИИ {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    ):
        events.append(ev)

    assert events[0]["output"]["message"] == message_text, (
        "streaming done event did not contain expected message"
    )


@pytest.mark.asyncio
async def test_streaming_yields_status_event_for_tool_call(mocker) -> None:
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )

    final_output = AgentTurnOutput(message="Канал проверен!", status="in_progress")
    tool_response = _mock_tool_call_response(
        "validate_source_url",
        '{"url": "https://t.me/s/durov", "source_kind": "telegram_channel"}',
    )
    final_response = _mock_parsed_response(final_output)

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(side_effect=[tool_response, final_response]),
    )

    events = []
    async for ev in run_conversation_turn_streaming(
        [{"role": "user", "content": f"Добавьте @durov {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    ):
        events.append(ev)

    assert events[0]["event"] == "status", "streaming did not yield status event before tool call"


@pytest.mark.asyncio
async def test_streaming_tool_call_flow_yields_done_event_last(mocker) -> None:
    mocker.patch(
        "news_service.agents.subscription_parser._validate_source_url",
        new=AsyncMock(return_value=True),
    )

    final_output = AgentTurnOutput(message="Канал проверен!", status="in_progress")
    tool_response = _mock_tool_call_response(
        "validate_source_url",
        '{"url": "https://t.me/s/durov", "source_kind": "telegram_channel"}',
    )
    final_response = _mock_parsed_response(final_output)

    mocker.patch(
        _CHAT_COMPLETION_PATH,
        new=AsyncMock(side_effect=[tool_response, final_response]),
    )

    events = []
    async for ev in run_conversation_turn_streaming(
        [{"role": "user", "content": f"Добавьте @durov {uuid.uuid4().hex[:4]}"}],
        user_language="ru",
    ):
        events.append(ev)

    assert events[-1]["event"] == "done", (
        "streaming tool call flow did not yield done event as last event"
    )
