"""Tests for the conversational agent creation and tool definitions."""

import asyncio
import logging
import uuid
from types import SimpleNamespace
from typing import Any
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
        language="en",
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
        "add_source",
        "remove_source",
        "set_user_language",
        "set_user_timezone",
        "discover_sources",
        "list_subscriptions",
        "trigger_digest_now",
        "delete_subscription",
        "emit_status",
    }
    assert expected_tools.issubset(tool_names), (
        f"agent missing tools: {expected_tools - tool_names}"
    )


def test_create_agent_has_thirteen_tools() -> None:
    db_session = AsyncMock()
    user = _fake_user()
    agent, _state = create_conversational_agent(
        db_session=db_session,
        user=user,
        user_spec="",
        conversation_summary="",
    )
    assert len(agent.tools) == 13, f"agent has {len(agent.tools)} tools, expected 13"


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


def _build_agent_with_factory(
    *,
    user: SimpleNamespace,
    factory_session: AsyncMock,
) -> tuple[Any, dict[str, Any]]:
    from news_service.agents.conversational import create_conversational_agent as _create

    class _Factory:
        def __init__(self, session: AsyncMock) -> None:
            self._session = session

        def __call__(self) -> Any:
            mgr = AsyncMock()
            mgr.__aenter__ = AsyncMock(return_value=self._session)
            mgr.__aexit__ = AsyncMock(return_value=None)
            return mgr

    return _create(
        db_session=AsyncMock(),
        user=user,
        user_spec="## Topic\nTest",
        conversation_summary="",
        session_factory=_Factory(factory_session),
    )


def _get_tool(agent: Any, name: str):
    return next(t for t in agent.tools if callable(t) and t.__name__ == name)


@pytest.mark.asyncio
async def test_add_source_rejects_unsupported_source_kind() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    agent, _state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    add_source = _get_tool(agent, "add_source")
    result = await add_source(str(uuid.uuid4()), "some-feed", "rss")
    assert "unsupported source_kind" in result, (
        f"add_source did not reject unsupported source_kind: {result!r}"
    )


@pytest.mark.asyncio
async def test_add_source_rejects_malformed_subscription_id() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    agent, _state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    add_source = _get_tool(agent, "add_source")
    result = await add_source("not-a-uuid", "bbcworld", "telegram_channel")
    assert "invalid subscription_id" in result, (
        f"add_source did not reject malformed subscription_id: {result!r}"
    )


@pytest.mark.asyncio
async def test_add_source_returns_unreachable_when_validation_fails(mocker) -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    agent, _state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    mocker.patch(
        "news_service.agents.conversational._validate_source_url",
        new=AsyncMock(return_value=False),
    )
    add_source = _get_tool(agent, "add_source")
    result = await add_source(str(uuid.uuid4()), "deadchannel", "telegram_channel")
    assert "unreachable or empty" in result, (
        f"add_source did not report unreachable source: {result!r}"
    )


@pytest.mark.asyncio
async def test_add_source_returns_not_found_when_subscription_missing(mocker) -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(return_value=scalar_result)

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    mocker.patch(
        "news_service.agents.conversational._validate_source_url",
        new=AsyncMock(return_value=True),
    )
    add_source = _get_tool(agent, "add_source")
    result = await add_source(str(uuid.uuid4()), "worldnews", "reddit_subreddit")
    assert "subscription not found" in result, (
        f"add_source did not detect missing subscription: {result!r}"
    )


@pytest.mark.asyncio
async def test_add_source_refuses_duplicate_attachment(mocker) -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.is_active = True

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub
    dup_lookup = MagicMock()
    dup_lookup.scalar_one_or_none.return_value = MagicMock()

    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(side_effect=[sub_lookup, dup_lookup])

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    mocker.patch(
        "news_service.agents.conversational._validate_source_url",
        new=AsyncMock(return_value=True),
    )
    add_source = _get_tool(agent, "add_source")
    result = await add_source(str(sub.id), "bbcworld", "telegram_channel")
    assert "already attached" in result, (
        f"add_source did not reject duplicate attachment: {result!r}"
    )


@pytest.mark.asyncio
async def test_add_source_attaches_and_commits_on_success(mocker) -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.is_active = True

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub
    dup_lookup = MagicMock()
    dup_lookup.scalar_one_or_none.return_value = None

    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(side_effect=[sub_lookup, dup_lookup])
    scoped_session.add = MagicMock()
    scoped_session.commit = AsyncMock()

    new_source = MagicMock()
    new_source.id = uuid.uuid4()

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    mocker.patch(
        "news_service.agents.conversational._validate_source_url",
        new=AsyncMock(return_value=True),
    )
    mocker.patch(
        "news_service.agents.conversational.ensure_source_coverage",
        new=AsyncMock(return_value=[new_source]),
    )
    add_source = _get_tool(agent, "add_source")
    result = await add_source(str(sub.id), "@bbcworld", "telegram_channel")
    assert result.endswith(": added."), f"add_source did not confirm attachment: {result!r}"
    scoped_session.commit.assert_awaited_once()
    scoped_session.add.assert_called_once()


@pytest.mark.asyncio
async def test_remove_source_rejects_unsupported_source_kind() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    agent, _state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    remove_source = _get_tool(agent, "remove_source")
    result = await remove_source(str(uuid.uuid4()), "some-feed", "rss")
    assert "unsupported source_kind" in result, (
        f"remove_source did not reject unsupported source_kind: {result!r}"
    )


@pytest.mark.asyncio
async def test_remove_source_returns_not_attached_when_link_missing() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub
    link_lookup = MagicMock()
    link_lookup.scalar_one_or_none.return_value = None

    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(side_effect=[sub_lookup, link_lookup])

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    remove_source = _get_tool(agent, "remove_source")
    result = await remove_source(str(sub.id), "bbcworld", "telegram_channel")
    assert "not attached" in result, f"remove_source did not report missing link: {result!r}"


@pytest.mark.asyncio
async def test_remove_source_deletes_link_and_logs_removal() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id

    link = MagicMock()
    link.source_id = uuid.uuid4()

    source = MagicMock()
    source.id = link.source_id
    source.subscriber_count = 2
    source.is_active = True

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub
    link_lookup = MagicMock()
    link_lookup.scalar_one_or_none.return_value = link
    source_lookup = MagicMock()
    source_lookup.scalar_one.return_value = source

    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(side_effect=[sub_lookup, link_lookup, source_lookup])
    scoped_session.add = MagicMock()
    scoped_session.delete = AsyncMock()
    scoped_session.commit = AsyncMock()

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    remove_source = _get_tool(agent, "remove_source")
    result = await remove_source(str(sub.id), "bbcworld", "telegram_channel")
    assert result.endswith(": removed."), f"remove_source did not confirm removal: {result!r}"
    scoped_session.delete.assert_awaited_once_with(link)
    scoped_session.commit.assert_awaited_once()
    scoped_session.add.assert_called_once()


@pytest.mark.asyncio
async def test_set_user_language_persists_normalized_code() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    persisted_user = MagicMock()
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted_user

    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(return_value=lookup)
    scoped_session.commit = AsyncMock()

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    set_user_language = _get_tool(agent, "set_user_language")
    result = await set_user_language("RU-cyrl")
    assert "ru" in result, f"set_user_language did not normalize code: {result!r}"
    assert persisted_user.language == "ru", (
        f"persisted language was not normalized: {persisted_user.language!r}"
    )
    assert user.language == "ru", "in-memory user.language was not updated"


@pytest.mark.asyncio
async def test_set_user_language_rejects_too_short_code() -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    agent, _state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    set_user_language = _get_tool(agent, "set_user_language")
    result = await set_user_language("x")
    assert "Invalid" in result, f"set_user_language accepted overly-short code: {result!r}"


@pytest.mark.asyncio
async def test_set_user_timezone_persists_on_resolved(mocker) -> None:
    from unittest.mock import MagicMock

    user = _fake_user()
    user.timezone = None

    persisted_user = MagicMock()
    persisted_user.timezone = None
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted_user

    scoped_session = AsyncMock()
    scoped_session.execute = AsyncMock(return_value=lookup)
    scoped_session.commit = AsyncMock()

    candidate = SimpleNamespace(
        label="Berlin, Germany",
        timezone="Europe/Berlin",
        local_time=lambda: SimpleNamespace(
            strftime=lambda _fmt: "15:30",
        ),
    )
    resolution = SimpleNamespace(status="resolved", candidates=(candidate,))
    mocker.patch(
        "news_service.agents.conversational.resolve_timezone",
        return_value=resolution,
    )

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    set_user_timezone = _get_tool(agent, "set_user_timezone")
    result = await set_user_timezone("Berlin")
    assert result.startswith("resolved:"), (
        f"set_user_timezone did not prefix resolved status: {result!r}"
    )
    assert persisted_user.timezone == "Europe/Berlin", (
        "persisted timezone was not set to the resolved IANA name"
    )
    assert user.timezone == "Europe/Berlin", "in-memory user.timezone was not updated"


@pytest.mark.asyncio
async def test_set_user_timezone_returns_ambiguous_without_persisting(mocker) -> None:

    user = _fake_user()
    original_timezone = user.timezone

    scoped_session = AsyncMock()
    scoped_session.commit = AsyncMock()

    portland_or = SimpleNamespace(label="Portland, United States", timezone="America/Los_Angeles")
    portland_me = SimpleNamespace(label="Portland, United States", timezone="America/New_York")
    resolution = SimpleNamespace(status="ambiguous", candidates=(portland_or, portland_me))
    mocker.patch(
        "news_service.agents.conversational.resolve_timezone",
        return_value=resolution,
    )

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    set_user_timezone = _get_tool(agent, "set_user_timezone")
    result = await set_user_timezone("Portland")
    assert result.startswith("ambiguous:"), f"set_user_timezone did not flag ambiguity: {result!r}"
    assert user.timezone == original_timezone, (
        "ambiguous resolution must not overwrite the user's timezone"
    )
    scoped_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_set_user_timezone_returns_not_found_on_unknown_query(mocker) -> None:

    user = _fake_user()
    scoped_session = AsyncMock()
    scoped_session.commit = AsyncMock()

    resolution = SimpleNamespace(status="not_found", candidates=())
    mocker.patch(
        "news_service.agents.conversational.resolve_timezone",
        return_value=resolution,
    )

    agent, _state = _build_agent_with_factory(user=user, factory_session=scoped_session)
    set_user_timezone = _get_tool(agent, "set_user_timezone")
    result = await set_user_timezone("qzzzx")
    assert result.startswith("not_found:"), (
        f"set_user_timezone did not signal not_found: {result!r}"
    )
    scoped_session.commit.assert_not_called()


def test_build_instruction_flags_first_time_user_when_no_subscriptions() -> None:
    result = _build_instruction(
        user_spec="",
        conversation_summary="",
        user_language=None,
        user_timezone=None,
        subscription_summaries=[],
    )
    assert "first-time interaction" in result, (
        "empty subscription list should trigger first-time greeting context"
    )


def test_build_instruction_lists_active_subscriptions_when_present() -> None:
    marker = f"[{uuid.uuid4()}] digest | 0 8 * * * | AI research {uuid.uuid4().hex[:6]}"
    result = _build_instruction(
        user_spec="",
        conversation_summary="",
        user_language="en",
        user_timezone="Europe/Berlin",
        subscription_summaries=[marker],
    )
    assert marker in result, "instruction did not include the subscription summary line"
