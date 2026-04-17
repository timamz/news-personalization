"""Tests for the conversational agent: prompt assembly + tool behavior."""

import asyncio
import logging
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.conversational import (
    _append_conversation_summary,
    _build_instruction,
    create_conversational_agent,
)

logging.disable(logging.CRITICAL)


_EXPECTED_TOOL_NAMES = {
    "save_subscription",
    "get_subscriptions",
    "remember",
    "add_source",
    "remove_source",
    "set_user_language",
    "set_user_timezone",
    "trigger_digest_now",
    "delete_subscription",
    "close_scenario",
}


def _fake_user(*, has_onboarded: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        timezone="Europe/Moscow",
        language="en",
        conversation_summary="",
        has_onboarded=has_onboarded,
    )


class _FakeSessionFactory:
    """async_sessionmaker stand-in that yields a supplied mock session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __call__(self) -> Any:
        mgr = AsyncMock()
        mgr.__aenter__ = AsyncMock(return_value=self._session)
        mgr.__aexit__ = AsyncMock(return_value=None)
        return mgr


def _build_agent_with_factory(
    *,
    user: SimpleNamespace,
    factory_session: Any,
) -> tuple[Any, dict[str, Any]]:
    return create_conversational_agent(
        db_session=AsyncMock(),
        user=user,
        conversation_summary="",
        session_factory=_FakeSessionFactory(factory_session),
    )


def _get_tool(agent: Any, name: str):
    return next(t for t in agent.tools if callable(t) and t.__name__ == name)


# ---------- prompt assembly ----------


def test_build_instruction_includes_conversation_summary() -> None:
    summary = f"durable facts {uuid.uuid4().hex[:6]}"
    result = _build_instruction(
        conversation_summary=summary,
        user_language=None,
        user_timezone=None,
    )
    assert summary in result, "instruction did not include conversation_summary"


def test_build_instruction_includes_language_preference() -> None:
    result = _build_instruction(
        conversation_summary="",
        user_language="ru",
        user_timezone=None,
    )
    assert "ru" in result, "instruction did not include language preference"


def test_build_instruction_includes_timezone() -> None:
    tz = "Europe/Berlin"
    result = _build_instruction(
        conversation_summary="",
        user_language=None,
        user_timezone=tz,
    )
    assert tz in result, "instruction did not include timezone"


def test_build_instruction_flags_first_time_user_when_not_onboarded() -> None:
    result = _build_instruction(
        conversation_summary="",
        user_language=None,
        user_timezone=None,
        subscription_summaries=[],
        has_onboarded=False,
    )
    assert "first-time interaction" in result, (
        "a user who has not completed onboarding should trigger the first-time cue"
    )


def test_build_instruction_does_not_flag_first_time_for_onboarded_user_without_subs() -> None:
    result = _build_instruction(
        conversation_summary="",
        user_language="en",
        user_timezone="Europe/Berlin",
        subscription_summaries=[],
        has_onboarded=True,
    )
    assert "first-time interaction" not in result, (
        "an onboarded user who has deleted all subs must not be greeted as a first-timer"
    )


def test_build_instruction_notes_returning_user_when_onboarded_without_subs() -> None:
    result = _build_instruction(
        conversation_summary="",
        user_language="en",
        user_timezone="Europe/Berlin",
        subscription_summaries=[],
        has_onboarded=True,
    )
    assert "returning user" in result, (
        "onboarded user without subs should be flagged as returning in context"
    )


def test_build_instruction_lists_active_subscriptions_when_present() -> None:
    marker = f"[{uuid.uuid4()}] digest | 0 8 * * * | AI research {uuid.uuid4().hex[:6]}"
    result = _build_instruction(
        conversation_summary="",
        user_language="en",
        user_timezone="Europe/Berlin",
        subscription_summaries=[marker],
    )
    assert marker in result, "instruction did not include the subscription summary line"


# ---------- agent construction ----------


def test_create_agent_returns_agent_and_shared_state() -> None:
    user = _fake_user()
    agent, state = create_conversational_agent(
        db_session=AsyncMock(),
        user=user,
        conversation_summary="",
        user_language="ru",
    )
    assert agent is not None, "create_conversational_agent did not return an agent"
    assert isinstance(state, dict), "create_conversational_agent did not return state dict"


def test_create_agent_registers_the_expected_tools() -> None:
    user = _fake_user()
    agent, _ = create_conversational_agent(
        db_session=AsyncMock(),
        user=user,
        conversation_summary="",
    )
    tool_names = {t.__name__ if callable(t) else t.name for t in agent.tools}
    missing = _EXPECTED_TOOL_NAMES - tool_names
    assert not missing, f"agent missing expected tools: {missing}"


def test_create_agent_registers_no_extra_tools() -> None:
    user = _fake_user()
    agent, _ = create_conversational_agent(
        db_session=AsyncMock(),
        user=user,
        conversation_summary="",
    )
    tool_names = {t.__name__ if callable(t) else t.name for t in agent.tools}
    extras = tool_names - _EXPECTED_TOOL_NAMES
    assert not extras, f"agent exposed unexpected tools: {extras}"


# ---------- remember tool ----------


@pytest.mark.asyncio
async def test_remember_appends_fact_to_user_conversation_summary() -> None:
    user = _fake_user()
    persisted = MagicMock()
    persisted.conversation_summary = ""
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted

    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    remember = _get_tool(agent, "remember")
    fact = f"user prefers short digests {uuid.uuid4().hex[:6]}"
    result = await remember(fact)

    assert result == "remembered.", f"remember did not confirm: {result!r}"
    assert fact in persisted.conversation_summary, (
        "remember did not append the fact to the persisted summary"
    )
    scoped.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_remember_ignores_empty_fact() -> None:
    user = _fake_user()
    scoped = AsyncMock()
    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    remember = _get_tool(agent, "remember")
    result = await remember("   ")
    assert "empty" in result.lower(), f"remember accepted an empty fact: {result!r}"
    scoped.execute.assert_not_called()


def test_append_conversation_summary_dedups_same_fact_hash() -> None:
    first = _append_conversation_summary("", "user is based in Berlin")
    twice = _append_conversation_summary(first, "user is based in Berlin")
    assert twice.count("Berlin") == 1, (
        "duplicate fact should be deduped, summary still contains multiple copies"
    )


def test_append_conversation_summary_caps_at_byte_limit() -> None:
    existing = "\n".join(f"2026-04-17 [{i:08d}] filler {i}" * 3 for i in range(100))
    added = _append_conversation_summary(existing, "brand new fact")
    assert len(added.encode("utf-8")) <= 2048, (
        "append_conversation_summary did not cap the summary below the byte limit"
    )
    assert "brand new fact" in added, "cap eviction removed the newly-added fact"


# ---------- save_subscription tool ----------


@pytest.mark.asyncio
async def test_save_subscription_requires_topic_on_create() -> None:
    user = _fake_user()
    scoped = AsyncMock()
    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    save = _get_tool(agent, "save_subscription")
    result = await save()
    assert "topic is required" in result, (
        f"save_subscription without topic on create should error: {result!r}"
    )
    scoped.commit.assert_not_called()


@pytest.mark.asyncio
async def test_save_subscription_rejects_malformed_id_on_update() -> None:
    user = _fake_user()
    scoped = AsyncMock()
    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    save = _get_tool(agent, "save_subscription")
    result = await save(subscription_id="not-a-uuid")
    assert "invalid subscription_id" in result, (
        f"save_subscription did not reject malformed id: {result!r}"
    )


@pytest.mark.asyncio
async def test_save_subscription_updates_scalar_fields(mocker) -> None:
    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id

    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub

    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    save = _get_tool(agent, "save_subscription")
    result = await save(
        subscription_id=str(sub.id),
        delivery_mode="event",
        schedule_cron="",
        digest_language="de",
        format_instructions="detailed",
    )
    assert result.endswith(": updated."), (
        f"save_subscription update path did not confirm: {result!r}"
    )
    assert sub.delivery_mode == "event", "update did not write delivery_mode"
    assert sub.digest_language == "de", "update did not write digest_language"


@pytest.mark.asyncio
async def test_save_subscription_creates_subscription_with_embedding(mocker) -> None:
    user = _fake_user()
    scoped = AsyncMock()
    scoped.execute = AsyncMock()
    scoped.add = MagicMock()
    scoped.flush = AsyncMock()
    scoped.commit = AsyncMock()

    mocker.patch(
        "news_service.agents.conversational.embed_text",
        new=AsyncMock(return_value=[0.0] * 8),
    )
    mocker.patch(
        "news_service.agents.conversational.ensure_source_coverage",
        new=AsyncMock(return_value=[]),
    )

    agent, shared_state = _build_agent_with_factory(user=user, factory_session=scoped)
    save = _get_tool(agent, "save_subscription")
    result = await save(
        topic=f"news about AI {uuid.uuid4().hex[:6]}",
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        include_discovered_sources=True,
    )
    assert ": created" in result, f"save_subscription did not confirm create: {result!r}"
    assert shared_state["created_subscription_id"], (
        "shared_state.created_subscription_id was not set after create"
    )


@pytest.mark.asyncio
async def test_save_subscription_flips_has_onboarded_on_first_create(mocker) -> None:
    user = _fake_user(has_onboarded=False)
    persisted = MagicMock()
    persisted.has_onboarded = False

    scoped = AsyncMock()
    scoped.execute = AsyncMock()
    scoped.add = MagicMock()
    scoped.flush = AsyncMock()
    scoped.commit = AsyncMock()
    scoped.get = AsyncMock(return_value=persisted)

    mocker.patch(
        "news_service.agents.conversational.embed_text",
        new=AsyncMock(return_value=[0.0] * 8),
    )
    mocker.patch(
        "news_service.agents.conversational.ensure_source_coverage",
        new=AsyncMock(return_value=[]),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    save = _get_tool(agent, "save_subscription")
    await save(
        topic=f"news {uuid.uuid4().hex[:6]}",
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        include_discovered_sources=False,
    )

    assert persisted.has_onboarded is True, (
        "first successful save_subscription did not flip has_onboarded on the persisted user"
    )
    assert user.has_onboarded is True, (
        "first successful save_subscription did not mirror has_onboarded onto the in-memory user"
    )


# ---------- get_subscriptions tool ----------


@pytest.mark.asyncio
async def test_get_subscriptions_returns_empty_message_when_none() -> None:
    user = _fake_user()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []

    db_session = AsyncMock()
    db_session.execute = AsyncMock(return_value=result_mock)

    agent, _ = create_conversational_agent(
        db_session=db_session,
        user=user,
        conversation_summary="",
    )
    get_subs = _get_tool(agent, "get_subscriptions")
    result = await get_subs()
    assert "No active subscriptions" in result, (
        f"get_subscriptions should say none when empty: {result!r}"
    )


# ---------- source tools (preserved behavior) ----------


@pytest.mark.asyncio
async def test_add_source_rejects_unsupported_source_kind() -> None:
    user = _fake_user()
    agent, _ = _build_agent_with_factory(user=user, factory_session=MagicMock())
    add_source = _get_tool(agent, "add_source")
    result = await add_source(str(uuid.uuid4()), "some-feed", "rss")
    assert "unsupported source_kind" in result, (
        f"add_source did not reject unsupported source_kind: {result!r}"
    )


@pytest.mark.asyncio
async def test_add_source_returns_unreachable_when_validation_fails(mocker) -> None:
    user = _fake_user()
    agent, _ = _build_agent_with_factory(user=user, factory_session=MagicMock())
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
async def test_remove_source_returns_not_attached_when_link_missing() -> None:
    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub
    link_lookup = MagicMock()
    link_lookup.scalar_one_or_none.return_value = None

    scoped = AsyncMock()
    scoped.execute = AsyncMock(side_effect=[sub_lookup, link_lookup])

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    remove_source = _get_tool(agent, "remove_source")
    result = await remove_source(str(sub.id), "bbcworld", "telegram_channel")
    assert "not attached" in result, f"remove_source did not report missing link: {result!r}"


# ---------- user state tools ----------


@pytest.mark.asyncio
async def test_set_user_language_persists_normalized_code() -> None:
    user = _fake_user()
    persisted = MagicMock()
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted

    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    set_lang = _get_tool(agent, "set_user_language")
    result = await set_lang("RU-cyrl")
    assert "ru" in result, f"set_user_language did not normalize code: {result!r}"
    assert persisted.language == "ru", (
        f"persisted language was not normalized: {persisted.language!r}"
    )


@pytest.mark.asyncio
async def test_set_user_timezone_persists_on_resolved(mocker) -> None:
    user = _fake_user()
    user.timezone = None
    persisted = MagicMock()
    persisted.timezone = None
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted

    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    candidate = SimpleNamespace(
        label="Berlin, Germany",
        timezone="Europe/Berlin",
        local_time=lambda: SimpleNamespace(strftime=lambda _: "15:30"),
    )
    mocker.patch(
        "news_service.agents.conversational.resolve_timezone",
        return_value=SimpleNamespace(status="resolved", candidates=(candidate,)),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    tool = _get_tool(agent, "set_user_timezone")
    result = await tool("Berlin")
    assert result.startswith("resolved:"), (
        f"set_user_timezone did not prefix resolved status: {result!r}"
    )
    assert persisted.timezone == "Europe/Berlin", (
        "persisted timezone was not set to the resolved IANA name"
    )


@pytest.mark.asyncio
async def test_set_user_timezone_returns_ambiguous_without_persisting(mocker) -> None:
    user = _fake_user()
    original = user.timezone
    scoped = AsyncMock()
    scoped.commit = AsyncMock()

    portland_or = SimpleNamespace(label="Portland, United States", timezone="America/Los_Angeles")
    portland_me = SimpleNamespace(label="Portland, United States", timezone="America/New_York")
    mocker.patch(
        "news_service.agents.conversational.resolve_timezone",
        return_value=SimpleNamespace(status="ambiguous", candidates=(portland_or, portland_me)),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    tool = _get_tool(agent, "set_user_timezone")
    result = await tool("Portland")
    assert result.startswith("ambiguous:"), f"set_user_timezone did not flag ambiguity: {result!r}"
    assert user.timezone == original, "ambiguous resolution must not overwrite the user's timezone"
    scoped.commit.assert_not_called()


@pytest.mark.asyncio
async def test_set_user_timezone_returns_not_found_on_unknown_query(mocker) -> None:
    user = _fake_user()
    scoped = AsyncMock()
    scoped.commit = AsyncMock()
    mocker.patch(
        "news_service.agents.conversational.resolve_timezone",
        return_value=SimpleNamespace(status="not_found", candidates=()),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    tool = _get_tool(agent, "set_user_timezone")
    result = await tool("qzzzx")
    assert result.startswith("not_found:"), (
        f"set_user_timezone did not signal not_found: {result!r}"
    )
    scoped.commit.assert_not_called()


# ---------- close_scenario tool ----------


@pytest.mark.asyncio
async def test_close_scenario_records_summary_into_shared_state() -> None:
    user = _fake_user()
    agent, shared_state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    close = _get_tool(agent, "close_scenario")
    summary = f"created AI digest daily 8am {uuid.uuid4().hex[:6]}"
    result = await close(summary)
    assert result == "scenario closed.", f"close_scenario did not confirm: {result!r}"
    assert shared_state["scenario_close_summary"] == summary, (
        "close_scenario did not expose the summary via shared_state"
    )


@pytest.mark.asyncio
async def test_close_scenario_ignores_empty_summary() -> None:
    user = _fake_user()
    agent, shared_state = _build_agent_with_factory(user=user, factory_session=MagicMock())
    close = _get_tool(agent, "close_scenario")
    result = await close("   ")
    assert "empty" in result.lower(), f"close_scenario accepted an empty summary: {result!r}"
    assert shared_state["scenario_close_summary"] is None, (
        "empty close_scenario call still wrote to shared_state"
    )


def test_build_instruction_renders_compacted_log_when_present() -> None:
    marker = f"created AI digest daily 8am {uuid.uuid4().hex[:6]}"
    result = _build_instruction(
        conversation_summary="",
        user_language="en",
        user_timezone="Europe/Berlin",
        compacted_log=[marker],
    )
    assert marker in result, (
        "instruction did not include the compacted_log entry in the context section"
    )


# ---------- status_queue wiring ----------


def test_create_agent_accepts_status_queue_without_error() -> None:
    user = _fake_user()
    queue: asyncio.Queue = asyncio.Queue()
    agent, _ = create_conversational_agent(
        db_session=AsyncMock(),
        user=user,
        conversation_summary="",
        status_queue=queue,
    )
    assert agent is not None, "agent should still build when a status_queue is passed"
