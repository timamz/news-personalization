"""
S-conv close_scenario fires: positive half of the close_scenario pair.

Paired with ``test_s_conv_close_scenario_stays_silent.py``. Test A proves
the positive signal -- after a clean ``create_subscription`` followed by
a user acknowledgement ("thanks, that's all I need"), the conversational
agent calls ``close_scenario`` and the hot transcript compacts. The
observable signal is ``len(state.compacted_log) == 1``.

The two halves live in separate files because pytest-asyncio's
function-scoped event loop combined with ADK/asyncpg connection state
causes cross-loop Future errors when two ADK-driving tests run in one
process (existing conversational tests all live in their own files for
this reason).

The companion file covers the negative case (agent stays silent when a
clarifying question is pending). Both halves are required to rule out
degenerate implementations (always-close or never-close).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


async def _drive_turn(state, *, user, db_session, message: str) -> str:
    from news_service.agents.conversational import run_conversation_turn_streaming
    from news_service.schemas.conversation import AgentTurnOutput

    state.messages.append({"role": "user", "content": message})

    agent_text = ""
    async for event in run_conversation_turn_streaming(
        state.messages,
        db_session=db_session,
        user=user,
        conversation_summary=user.conversation_summary or "",
        user_language=state.user_language,
        compacted_log=list(state.compacted_log),
    ):
        if event["event"] == "done":
            output = AgentTurnOutput.model_validate(event["output"])
            agent_text = output.message
            state.messages.extend(event["new_messages"])
            shared = event.get("shared_state") or {}
            close_summary = shared.get("scenario_close_summary")
            if close_summary:
                state.compacted_log.append(close_summary.strip())
    return agent_text


@pytest.mark.asyncio
async def test_s_conv_close_scenario_fires_on_completed_subscription_creation(world):
    """close_scenario fires after a clean create_subscription plus user acknowledgement."""
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.models.failed_task import FailedTask
    from news_service.models.subscription import Subscription
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    async def _noop_discovery(*_args, **_kwargs):
        return {"status": "skipped", "reason": "S-conv close_scenario test bypass"}

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _noop_discovery  # type: ignore[assignment]

    try:
        user_id = uuid.uuid4()
        async with async_session_factory() as s:
            s.add(
                User(
                    id=user_id,
                    api_key=f"bench-{user_id.hex}",
                    language="en",
                    timezone="UTC",
                    has_onboarded=False,
                )
            )
            await s.commit()

        state = ConversationState(user_id=str(user_id), user_language="en")

        create_msg = (
            "Create a daily digest about EU energy regulation at 09:00 UTC. "
            "Don't ask questions, don't do anything else."
        )
        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None, "user disappeared before driving turn 1"
            turn1_text = await _drive_turn(
                state, user=user, db_session=s, message=create_msg
            )

        ack_msg = "Great, thanks. That's all I need."
        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None, "user disappeared before driving turn 2"
            turn2_text = await _drive_turn(
                state, user=user, db_session=s, message=ack_msg
            )

        assert len(state.compacted_log) == 1, (
            f"close_scenario did not fire after create+acknowledge: "
            f"compacted_log has {len(state.compacted_log)} entries, expected 1. "
            f"Turn 1: {turn1_text!r}. Turn 2: {turn2_text!r}"
        )
        entry = state.compacted_log[0]
        assert entry and len(entry) <= 200, (
            f"compacted_log entry should be non-empty and <=200 chars (tool truncates), "
            f"got length={len(entry)}: {entry!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(
                select(Subscription).where(
                    Subscription.user_id == user_id,
                    Subscription.is_active.is_(True),
                )
            )
            subs = list(rows.scalars().all())
        assert len(subs) == 1, (
            f"scenario claims to be closed but no completed work exists: "
            f"expected 1 active subscription, got {len(subs)}. "
            f"Turn 1: {turn1_text!r}. Turn 2: {turn2_text!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
