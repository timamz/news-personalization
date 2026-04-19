"""Tests for the conversational agent: prompt assembly + tool behavior."""

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


def _fake_user(*, has_onboarded: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        timezone="Europe/Moscow",
        language="en",
        conversation_summary="",
        has_onboarded=has_onboarded,
    )


class _FakeSessionFactory:
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


def test_build_instruction_renders_all_context_sections() -> None:
    summary = f"durable {uuid.uuid4().hex[:6]}"
    tz = "Europe/Berlin"
    sub_line = f"[{uuid.uuid4()}] digest | 0 8 * * * | AI {uuid.uuid4().hex[:4]}"
    compacted = f"closed scenario {uuid.uuid4().hex[:4]}"

    result = _build_instruction(
        conversation_summary=summary,
        user_language="ru",
        user_timezone=tz,
        subscription_summaries=[sub_line],
        compacted_log=[compacted],
        has_onboarded=True,
    )

    assert all(s in result for s in (summary, "ru", tz, sub_line, compacted)), (
        "instruction did not include every context section"
    )


def test_prompt_forbids_a_second_create_subscription_for_a_just_created_topic() -> None:
    from news_service.agents.conversational.prompt import CONVERSATIONAL_AGENT_PROMPT

    assert "Never call it twice for the same topic" in CONVERSATIONAL_AGENT_PROMPT, (
        "prompt is missing the guard against duplicate create_subscription calls"
    )


def test_prompt_forbids_clarifying_questions_after_create_subscription() -> None:
    from news_service.agents.conversational.prompt import CONVERSATIONAL_AGENT_PROMPT

    assert "Ask every clarifying question you need BEFORE" in CONVERSATIONAL_AGENT_PROMPT, (
        "prompt is missing the up-front-questions rule that prevents re-triggering discovery"
    )


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


def test_build_instruction_treats_onboarded_user_without_subs_as_returning() -> None:
    result = _build_instruction(
        conversation_summary="",
        user_language="en",
        user_timezone="Europe/Berlin",
        subscription_summaries=[],
        has_onboarded=True,
    )
    assert "first-time interaction" not in result and "returning user" in result


# ---------- agent construction ----------


def test_create_agent_registers_the_expected_tools() -> None:
    expected = {
        "create_subscription",
        "update_subscription",
        "get_subscriptions",
        "remember",
        "add_source",
        "remove_source",
        "set_user_language",
        "set_user_timezone",
        "trigger_digest_now",
        "trigger_source_discovery",
        "delete_subscription",
        "close_scenario",
    }
    agent, _ = create_conversational_agent(
        db_session=AsyncMock(),
        user=_fake_user(),
        conversation_summary="",
    )
    tool_names = {t.__name__ for t in agent.tools if callable(t)}
    assert tool_names == expected, (
        f"registered tool set drifted from the expected set (diff={tool_names ^ expected})"
    )


# ---------- remember + conversation summary ----------


@pytest.mark.asyncio
async def test_remember_appends_fact_to_user_conversation_summary() -> None:
    persisted = MagicMock()
    persisted.conversation_summary = ""
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted

    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent_with_factory(user=_fake_user(), factory_session=scoped)
    fact = f"user prefers short digests {uuid.uuid4().hex[:6]}"
    await _get_tool(agent, "remember")(fact)
    assert fact in persisted.conversation_summary, (
        "remember did not append the fact to the persisted summary"
    )


def test_append_conversation_summary_dedups_same_fact_and_caps_bytes() -> None:
    first = _append_conversation_summary("", "user is based in Berlin")
    twice = _append_conversation_summary(first, "user is based in Berlin")
    assert twice.count("Berlin") == 1, "duplicate fact should be deduped"

    existing = "\n".join(f"2026-04-17 [{i:08d}] filler {i}" * 3 for i in range(100))
    added = _append_conversation_summary(existing, "brand new fact")
    assert len(added.encode("utf-8")) <= 2048 and "brand new fact" in added, (
        "cap eviction did not keep the newest fact under the byte limit"
    )


# ---------- create / update subscription ----------


@pytest.mark.asyncio
async def test_create_subscription_embeds_retrieval_query_and_queues_discovery(mocker) -> None:
    user = _fake_user()
    scoped = AsyncMock()
    scoped.add = MagicMock()
    scoped.flush = AsyncMock()
    scoped.commit = AsyncMock()
    embed_mock = mocker.patch(
        "news_service.agents.conversational.tools.embed_text",
        new=AsyncMock(return_value=[0.0] * 8),
    )
    mocker.patch(
        "news_service.agents.conversational.tools.ensure_source_coverage",
        new=AsyncMock(return_value=[]),
    )
    enqueue_mock = mocker.patch(
        "news_service.agents.conversational.tools._enqueue_discovery",
    )

    agent, shared_state = _build_agent_with_factory(user=user, factory_session=scoped)
    query = f"AI safety, alignment, interpretability {uuid.uuid4().hex[:6]}"
    result = await _get_tool(agent, "create_subscription")(
        user_spec="AI safety research. Three bullets, neutral tone. Skip hype.",
        retrieval_query=query,
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        include_discovered_sources=True,
    )
    assert (
        ": created" in result
        and shared_state["created_subscription_id"]
        and embed_mock.await_args.args[0] == query
        and enqueue_mock.call_count == 1
        and query in enqueue_mock.call_args.args[1]
    ), (
        "creation must embed retrieval_query, record the id, and enqueue "
        "discovery with a reason that mentions the query"
    )


@pytest.mark.asyncio
async def test_create_subscription_skips_discovery_when_not_requested(mocker) -> None:
    user = _fake_user()
    scoped = AsyncMock()
    scoped.add = MagicMock()
    scoped.flush = AsyncMock()
    scoped.commit = AsyncMock()
    mocker.patch(
        "news_service.agents.conversational.tools.embed_text",
        new=AsyncMock(return_value=[0.0] * 8),
    )
    mocker.patch(
        "news_service.agents.conversational.tools.ensure_source_coverage",
        new=AsyncMock(return_value=[]),
    )
    enqueue_mock = mocker.patch(
        "news_service.agents.conversational.tools._enqueue_discovery",
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    await _get_tool(agent, "create_subscription")(
        user_spec="News. Brief.",
        retrieval_query="world news",
        delivery_mode="digest",
        include_discovered_sources=False,
    )
    assert enqueue_mock.call_count == 0, (
        "discovery must not fire when include_discovered_sources is False"
    )


@pytest.mark.asyncio
async def test_create_subscription_rejects_empty_retrieval_query(mocker) -> None:
    user = _fake_user()
    scoped = AsyncMock()
    embed_mock = mocker.patch(
        "news_service.agents.conversational.tools.embed_text",
        new=AsyncMock(return_value=[0.0] * 8),
    )

    agent, shared_state = _build_agent_with_factory(user=user, factory_session=scoped)
    result = await _get_tool(agent, "create_subscription")(
        user_spec="AI news. Brief bullets.",
        retrieval_query="   ",
        delivery_mode="digest",
    )
    assert "retrieval_query is required" in result, (
        "empty retrieval_query must be rejected without embedding or DB writes"
    )
    assert embed_mock.await_count == 0 and shared_state["created_subscription_id"] is None


@pytest.mark.asyncio
async def test_create_subscription_flips_has_onboarded_on_first_success(mocker) -> None:
    user = _fake_user(has_onboarded=False)
    persisted = MagicMock()
    persisted.has_onboarded = False

    scoped = AsyncMock()
    scoped.add = MagicMock()
    scoped.flush = AsyncMock()
    scoped.commit = AsyncMock()
    scoped.get = AsyncMock(return_value=persisted)
    mocker.patch(
        "news_service.agents.conversational.tools.embed_text",
        new=AsyncMock(return_value=[0.0] * 8),
    )
    mocker.patch(
        "news_service.agents.conversational.tools.ensure_source_coverage",
        new=AsyncMock(return_value=[]),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    await _get_tool(agent, "create_subscription")(
        user_spec=f"news {uuid.uuid4().hex[:6]}. Daily summary.",
        retrieval_query=f"world news, headlines, breaking {uuid.uuid4().hex[:4]}",
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        include_discovered_sources=False,
    )
    assert persisted.has_onboarded is True and user.has_onboarded is True, (
        "first create_subscription did not mark persisted+in-memory user as onboarded"
    )


@pytest.mark.asyncio
async def test_update_subscription_reembeds_only_when_retrieval_query_provided(mocker) -> None:
    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.user_spec = "old spec about biotech"
    sub.topic_embedding = [0.0] * 8

    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    new_vector = [0.42] * 8
    embed_mock = mocker.patch(
        "news_service.agents.conversational.tools.embed_text",
        new=AsyncMock(return_value=new_vector),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    update = _get_tool(agent, "update_subscription")

    await update(
        subscription_id=str(sub.id),
        user_spec="new spec about robotics. Short bullets.",
        retrieval_query="robotics, humanoid robots, Boston Dynamics, Figure AI",
    )
    assert sub.topic_embedding == new_vector and embed_mock.await_count == 1, (
        "retrieval_query change did not trigger exactly one re-embed"
    )

    embed_mock.reset_mock()
    await update(
        subscription_id=str(sub.id),
        user_spec="robotics news. Make it shorter now.",
    )
    assert embed_mock.await_count == 0, "user_spec-only edit (no retrieval_query) must not re-embed"


@pytest.mark.asyncio
async def test_trigger_source_discovery_enqueues_with_reason(mocker) -> None:
    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.is_active = True

    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)

    enqueue_mock = mocker.patch(
        "news_service.agents.conversational.tools._enqueue_discovery",
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    reason = (
        f"User shifted focus from biotech to AI safety {uuid.uuid4().hex[:6]}. "
        "Existing sources stale."
    )
    result = await _get_tool(agent, "trigger_source_discovery")(
        subscription_id=str(sub.id),
        reason=reason,
    )
    assert (
        "queued" in result
        and enqueue_mock.call_count == 1
        and enqueue_mock.call_args.args[0] == sub.id
        and enqueue_mock.call_args.args[1] == reason
    ), "trigger_source_discovery did not enqueue the task with the given reason"


@pytest.mark.asyncio
async def test_trigger_source_discovery_rejects_empty_reason(mocker) -> None:
    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.is_active = True
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    enqueue_mock = mocker.patch(
        "news_service.agents.conversational.tools._enqueue_discovery",
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    result = await _get_tool(agent, "trigger_source_discovery")(
        subscription_id=str(sub.id),
        reason="   ",
    )
    assert "reason is required" in result and enqueue_mock.call_count == 0, (
        "empty reason must be rejected before any Celery dispatch"
    )


# ---------- source tools ----------


@pytest.mark.asyncio
async def test_add_source_reports_unreachable_when_validation_fails(mocker) -> None:
    mocker.patch(
        "news_service.agents.conversational.tools._validate_source_url",
        new=AsyncMock(return_value=False),
    )
    agent, _ = _build_agent_with_factory(user=_fake_user(), factory_session=MagicMock())
    result = await _get_tool(agent, "add_source")(
        str(uuid.uuid4()), "deadchannel", "telegram_channel"
    )
    assert "unreachable" in result


@pytest.mark.asyncio
async def test_remove_source_reports_when_link_is_missing() -> None:
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
    result = await _get_tool(agent, "remove_source")(str(sub.id), "bbcworld", "telegram_channel")
    assert "not attached" in result


# ---------- user state tools ----------


@pytest.mark.asyncio
async def test_set_user_language_normalizes_and_persists_iso_code() -> None:
    user = _fake_user()
    persisted = MagicMock()
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = persisted

    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    await _get_tool(agent, "set_user_language")("RU-cyrl")
    assert persisted.language == "ru", (
        "set_user_language did not strip the region suffix before persisting"
    )


@pytest.mark.asyncio
async def test_set_user_timezone_persists_resolved_candidate(mocker) -> None:
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
        "news_service.agents.conversational.tools.resolve_timezone",
        return_value=SimpleNamespace(status="resolved", candidates=(candidate,)),
    )

    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    await _get_tool(agent, "set_user_timezone")("Berlin")
    assert persisted.timezone == "Europe/Berlin"


@pytest.mark.asyncio
async def test_set_user_timezone_does_not_persist_on_ambiguous_or_not_found(mocker) -> None:
    user = _fake_user()
    original = user.timezone
    scoped = AsyncMock()
    scoped.commit = AsyncMock()

    mocker.patch(
        "news_service.agents.conversational.tools.resolve_timezone",
        return_value=SimpleNamespace(
            status="ambiguous",
            candidates=(
                SimpleNamespace(label="Portland, US", timezone="America/Los_Angeles"),
                SimpleNamespace(label="Portland, US", timezone="America/New_York"),
            ),
        ),
    )
    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    await _get_tool(agent, "set_user_timezone")("Portland")
    assert user.timezone == original and scoped.commit.await_count == 0, (
        "ambiguous resolution should not overwrite the stored timezone"
    )

    mocker.patch(
        "news_service.agents.conversational.tools.resolve_timezone",
        return_value=SimpleNamespace(status="not_found", candidates=()),
    )
    agent, _ = _build_agent_with_factory(user=user, factory_session=scoped)
    await _get_tool(agent, "set_user_timezone")("qzzzx")
    assert user.timezone == original, "not_found resolution should not overwrite the timezone"


# ---------- close_scenario ----------


@pytest.mark.asyncio
async def test_close_scenario_records_summary_and_ignores_empty() -> None:
    agent, shared_state = _build_agent_with_factory(user=_fake_user(), factory_session=MagicMock())
    close = _get_tool(agent, "close_scenario")

    summary = f"created AI digest daily 8am {uuid.uuid4().hex[:6]}"
    await close(summary)
    assert shared_state["scenario_close_summary"] == summary

    shared_state["scenario_close_summary"] = None
    await close("   ")
    assert shared_state["scenario_close_summary"] is None, (
        "empty summary must not be written to shared_state"
    )
