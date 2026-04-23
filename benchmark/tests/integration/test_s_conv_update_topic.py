"""
S-conv-update-topic: conversational-agent topic-rewrite smoke test.

A single scripted user turn that asks the Conversational Agent to
retarget an existing subscription from one topic (EU energy policy)
to a completely different one (African wildlife conservation).

The test proves that when the user's instruction is unambiguous and
ends with a "don't do anything else, don't ask more questions"
directive, the agent routes to ``update_subscription`` and rewrites
the subscription's ``user_spec`` (and, by necessity, the embedded
retrieval intent) while leaving every unrelated field untouched:
schedule_cron, delivery_mode, digest_language, and the is_active
flag must all survive the edit intact, and no duplicate active
subscription may be created as a side effect.

Failure modes caught: wrong tool selection (e.g. ``create_subscription``
instead of ``update_subscription``), field-wipe on update, silent
no-op on ``user_spec``, duplicate-subscription creation, or any
exception that leaves a ``failed_tasks`` row behind.

``conv_tools._run_inline_discovery`` is stubbed to a no-op because
``update_subscription`` may kick off inline source re-discovery when
the topic shifts, and that pipeline is out of scope for this test.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


ORIGINAL_USER_SPEC = (
    "Topic: EU energy policy and regulation.\n"
    "Focus on the European Commission's energy directives, gas and electricity "
    "market rules, carbon pricing mechanisms, and cross-border interconnector "
    "decisions. Prefer primary-source announcements from the Commission, ENTSO-E, "
    "and ACER over opinion pieces. Exclude national partisan politics that is not "
    "directly tied to EU-wide policy instruments. Tone: neutral, analytical, "
    "English."
)

ORIGINAL_SCHEDULE_CRON = "0 9 * * *"
ORIGINAL_DELIVERY_MODE = "digest"
ORIGINAL_DIGEST_LANGUAGE = "en"

TOPIC_CHANGE_TURN = (
    "Change that EU energy subscription to be about African wildlife conservation "
    "instead. Don't do anything else, don't ask more questions."
)


async def _drive_turn(state, *, user, db_session, message: str) -> str:
    """Append one user message, run the streaming agent, update state.

    Returns the agent's final text so diagnostics can include it on
    assertion failure.
    """
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
async def test_s_conv_update_topic_rewrites_user_spec_without_clobbering_delivery(world):
    """Agent rewrites user_spec on topic change and preserves all delivery fields."""
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.models.failed_task import FailedTask
    from news_service.models.subscription import Subscription
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    async def _noop_discovery(*_args, **_kwargs):
        return {"status": "skipped", "reason": "S-conv-update-topic smoke test bypass"}

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _noop_discovery  # type: ignore[assignment]

    try:
        from news_service.db.vector_store import embed_text

        user_id = uuid.uuid4()
        sub_id = uuid.uuid4()
        topic_embedding = await embed_text("EU energy policy news")
        async with async_session_factory() as s:
            s.add(
                User(
                    id=user_id,
                    api_key=f"bench-{user_id.hex}",
                    language="en",
                    timezone="UTC",
                    has_onboarded=True,
                )
            )
            s.add(
                Subscription(
                    id=sub_id,
                    user_id=user_id,
                    user_spec=ORIGINAL_USER_SPEC,
                    topic_embedding=topic_embedding,
                    delivery_mode=ORIGINAL_DELIVERY_MODE,
                    schedule_cron=ORIGINAL_SCHEDULE_CRON,
                    digest_language=ORIGINAL_DIGEST_LANGUAGE,
                    is_active=True,
                )
            )
            await s.commit()

        state = ConversationState(user_id=str(user_id), user_language="en")

        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None, "seeded user vanished before the turn ran"
            agent_text = await _drive_turn(
                state, user=user, db_session=s, message=TOPIC_CHANGE_TURN
            )

        async with async_session_factory() as s:
            active_rows = await s.execute(
                select(Subscription).where(
                    Subscription.user_id == user_id,
                    Subscription.is_active.is_(True),
                )
            )
            active_subs = list(active_rows.scalars().all())
            refreshed = await s.get(Subscription, sub_id)
            failed_rows = await s.execute(select(FailedTask))
            failed = list(failed_rows.scalars().all())

        assert refreshed is not None, (
            f"subscription {sub_id} disappeared after update. Agent said: {agent_text!r}"
        )
        assert refreshed.is_active is True, (
            f"subscription must stay active after topic rewrite, "
            f"got is_active={refreshed.is_active!r}. Agent said: {agent_text!r}"
        )

        new_spec = (refreshed.user_spec or "").strip()
        assert new_spec and new_spec != ORIGINAL_USER_SPEC.strip(), (
            f"user_spec should be rewritten; got unchanged spec {new_spec!r}. "
            f"Agent said: {agent_text!r}"
        )
        lowered = new_spec.lower()
        assert any(kw in lowered for kw in ("wildlife", "conservation", "africa")), (
            f"rewritten user_spec should mention wildlife/conservation/africa, "
            f"got {new_spec!r}. Agent said: {agent_text!r}"
        )

        assert (refreshed.schedule_cron or "").strip() == ORIGINAL_SCHEDULE_CRON, (
            f"schedule_cron must survive topic rewrite, expected "
            f"{ORIGINAL_SCHEDULE_CRON!r}, got {refreshed.schedule_cron!r}. "
            f"Agent said: {agent_text!r}"
        )
        assert refreshed.delivery_mode == ORIGINAL_DELIVERY_MODE, (
            f"delivery_mode must survive topic rewrite, expected "
            f"{ORIGINAL_DELIVERY_MODE!r}, got {refreshed.delivery_mode!r}. "
            f"Agent said: {agent_text!r}"
        )
        assert (refreshed.digest_language or "").lower() == ORIGINAL_DIGEST_LANGUAGE, (
            f"digest_language must survive topic rewrite, expected "
            f"{ORIGINAL_DIGEST_LANGUAGE!r}, got {refreshed.digest_language!r}. "
            f"Agent said: {agent_text!r}"
        )

        assert len(active_subs) == 1, (
            f"expected exactly one active subscription for user after update, "
            f"got {len(active_subs)}: {[s.id for s in active_subs]!r}. "
            f"Agent said: {agent_text!r}"
        )
        assert active_subs[0].id == sub_id, (
            f"the single active subscription should be the seeded one {sub_id}, "
            f"got {active_subs[0].id!r}. Agent said: {agent_text!r}"
        )

        assert not failed, (
            f"expected 0 failed_tasks rows after update_subscription, "
            f"got {len(failed)}: {failed!r}. Agent said: {agent_text!r}"
        )
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
