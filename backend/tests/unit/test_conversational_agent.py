"""Tests for the conversational agent creation and tool definitions."""

import asyncio
import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.conversational import (
    _build_instruction,
    create_conversational_agent,
)

logging.disable(logging.CRITICAL)


def _fake_user() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        timezone="Europe/Moscow",
    )


def test_build_instruction_includes_user_spec() -> None:
    spec = f"## Topic\nМашинное обучение {uuid.uuid4().hex[:6]}"
    result = _build_instruction(
        user_spec=spec,
        conversation_summary="",
        user_language="ru",
        user_timezone="Europe/Moscow",
    )
    assert spec in result, "instruction did not include user_spec content"


def test_build_instruction_includes_conversation_summary() -> None:
    summary = f"Пользователь хочет дайджест ИИ {uuid.uuid4().hex[:6]}"
    result = _build_instruction(
        user_spec="",
        conversation_summary=summary,
        user_language=None,
        user_timezone=None,
    )
    assert summary in result, "instruction did not include conversation summary"


def test_build_instruction_includes_language_preference() -> None:
    result = _build_instruction(
        user_spec="",
        conversation_summary="",
        user_language="ru",
        user_timezone=None,
    )
    assert "ru" in result, "instruction did not include language preference"


def test_build_instruction_includes_timezone() -> None:
    tz = "Europe/Berlin"
    result = _build_instruction(
        user_spec="",
        conversation_summary="",
        user_language=None,
        user_timezone=tz,
    )
    assert tz in result, "instruction did not include timezone"


def test_create_agent_returns_agent_and_shared_state() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    agent, state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec=f"## Topic\nТест {uuid.uuid4().hex[:6]}",
        conversation_summary="",
        user_language="ru",
    )
    assert agent is not None, "create_conversational_agent did not return an agent"
    assert isinstance(state, dict), "create_conversational_agent did not return state dict"


def test_create_agent_has_all_tools() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec="",
        conversation_summary="",
    )
    tool_names = {t.__name__ if callable(t) else t.name for t in agent.tools}
    expected_tools = {
        "finalize_subscription",
        "create_subscription",
        "update_user_spec",
        "validate_source",
        "discover_sources",
        "list_subscriptions",
        "trigger_digest_now",
        "delete_subscription",
        "emit_status",
    }
    assert expected_tools.issubset(tool_names), (
        f"agent missing tools: {expected_tools - tool_names}"
    )


def test_create_agent_has_nine_tools() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec="",
        conversation_summary="",
    )
    assert len(agent.tools) == 9, f"agent has {len(agent.tools)} tools, expected 9"


def test_shared_state_initial_values() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    spec = f"## Topic\nТехнологии {uuid.uuid4().hex[:6]}"
    _agent, state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec=spec,
        conversation_summary="",
    )
    assert state["user_spec_updated"] is False, "initial user_spec_updated should be False"
    assert state["new_user_spec"] == spec, "initial new_user_spec should equal provided spec"
    assert state["subscription_created"] is False, "initial subscription_created should be False"
    assert state["created_subscription_id"] is None, (
        "initial created_subscription_id should be None"
    )
    assert state["discovery_triggered"] is False, "initial discovery_triggered should be False"
    assert state["status"] == "in_progress", "initial status should be in_progress"
    assert state["finalized_config"] is None, "initial finalized_config should be None"


def test_create_agent_with_status_queue_includes_emit_status_tool() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    queue: asyncio.Queue = asyncio.Queue()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec=f"## Topic\nТестирование {uuid.uuid4().hex[:6]}",
        conversation_summary="",
        status_queue=queue,
    )
    tool_names = {t.__name__ if callable(t) else t.name for t in agent.tools}
    assert "emit_status" in tool_names, "agent did not include emit_status tool"


@pytest.mark.asyncio
async def test_emit_status_tool_puts_event_into_queue() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    queue: asyncio.Queue = asyncio.Queue()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec="## Topic\nNoticias de tecnologia",
        conversation_summary="",
        status_queue=queue,
    )
    emit_tool = next(t for t in agent.tools if callable(t) and t.__name__ == "emit_status")
    progress_msg = f"Buscando fuentes sobre IA... {uuid.uuid4().hex[:6]}"
    result = await emit_tool(progress_msg)
    assert result == "Status emitted.", "emit_status did not return confirmation"
    assert not queue.empty(), "emit_status did not put any event into the queue"
    event = queue.get_nowait()
    assert event["event"] == "status", "queued event type is not 'status'"
    assert event["status_key"] == "status_agent_progress", (
        "queued event status_key is not 'status_agent_progress'"
    )
    assert event["status_text"] == progress_msg, "queued event status_text does not match message"


@pytest.mark.asyncio
async def test_emit_status_tool_without_queue_does_not_raise() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec="## Topic\nNouvelles technologiques",
        conversation_summary="",
        status_queue=None,
    )
    emit_tool = next(t for t in agent.tools if callable(t) and t.__name__ == "emit_status")
    result = await emit_tool(f"Recherche de canaux... {uuid.uuid4().hex[:6]}")
    assert result == "Status emitted.", "emit_status without queue did not return confirmation"
