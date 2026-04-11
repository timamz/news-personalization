"""Tests for the conversational agent creation and tool definitions."""

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
        "create_subscription",
        "update_user_spec",
        "validate_source",
        "discover_sources",
        "list_subscriptions",
        "trigger_digest_now",
        "delete_subscription",
    }
    assert expected_tools.issubset(tool_names), (
        f"agent missing tools: {expected_tools - tool_names}"
    )


def test_create_agent_has_seven_tools() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec="",
        conversation_summary="",
    )
    assert len(agent.tools) == 7, f"agent has {len(agent.tools)} tools, expected 7"


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
