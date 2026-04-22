"""
S-conv-trigger-digest: conversational-agent ``trigger_digest_now`` smoke test.

One scripted user turn asks the agent to deliver an EU energy digest
immediately. The conversational agent should pick the
``trigger_digest_now`` tool, which calls
``deliver_digest.delay(subscription_id, notify_if_empty=True)``. The
benchmark's ``CeleryShim`` routes that ``.delay(...)`` dispatch to the
underlying async ``_deliver_digest`` coroutine inline.

The seeded subscription intentionally has zero ``NewsItem`` rows, so
``_deliver_digest`` hits its empty-queue branch. Because
``notify_if_empty=True`` was set by the tool, that branch fires a
"No new updates right now" webhook instead of silently skipping.

Assertions prove the whole chain works end to end:

  agent tool selection
    -> celery dispatch (``.delay`` captured by the shim)
      -> ``_deliver_digest`` empty-queue branch
        -> empty-notification webhook landed on the sub's webhook URL.

Out of scope: actual Digest Writer planning / composition / judge
behaviour. Those live in S-digest-happy. We only care here that the
conversational tool correctly wires into the delivery pipeline and that
the empty-queue notification path is reachable from a user utterance.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

WEBHOOK_URL = "https://bench.invalid/webhook/s-conv-trigger-digest"
SOURCE_URL = "https://brussels-energy-policy.invalid/feed.xml"


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
async def test_s_conv_trigger_digest_now_empty_notification(world):
    """One-turn transcript covering ``trigger_digest_now`` with the empty-queue branch.

    Seeds a user + subscription + one source + zero news items, drives a
    single "deliver my digest right now even if empty" user turn, and
    verifies that the empty-notification webhook landed on the sub's
    webhook URL.
    """
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    async def _noop_discovery(*_args, **_kwargs):
        return {
            "status": "skipped",
            "reason": "S-conv-trigger-digest smoke test bypass",
        }

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _noop_discovery  # type: ignore[assignment]

    try:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        sub_id = uuid.uuid4()

        topic_embedding = await embed_text("EU energy regulation news")

        async with async_session_factory() as s:
            s.add(
                User(
                    id=user_id,
                    api_key=f"bench-{user_id.hex}",
                    language="en",
                    timezone="UTC",
                    delivery_webhook_url=WEBHOOK_URL,
                    has_onboarded=True,
                )
            )
            s.add(
                Source(
                    id=source_id,
                    url=SOURCE_URL,
                    title="Brussels Energy Policy Monitor",
                    source_description=(
                        "Daily updates on EU-wide energy regulation, ETS reforms, "
                        "grid integration, and Commission proposals."
                    ),
                    is_active=True,
                )
            )
            s.add(
                Subscription(
                    id=sub_id,
                    user_id=user_id,
                    user_spec=(
                        "EU energy policy news, focused on Commission proposals, "
                        "ETS reforms, and grid integration. Daily digest at 08:00 UTC "
                        "in English. Skip national-level politics."
                    ),
                    topic_embedding=topic_embedding,
                    delivery_mode="digest",
                    digest_language="en",
                    schedule_cron="0 8 * * *",
                    delivery_webhook_url=WEBHOOK_URL,
                    is_active=True,
                )
            )
            s.add(
                SubscriptionSource(
                    subscription_id=sub_id,
                    source_id=source_id,
                    is_user_specified=True,
                )
            )
            await s.commit()

        state = ConversationState(user_id=str(user_id), user_language="en")

        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None, "seeded user disappeared before the turn"
            agent_text = await _drive_turn(
                state,
                user=user,
                db_session=s,
                message=(
                    "Deliver my EU energy digest to me right now even if the queue "
                    "is empty. Don't do anything else, don't ask more questions."
                ),
            )

        await world.celery.drain()

        captures = world.delivery.for_url(WEBHOOK_URL)
        assert len(captures) == 1, (
            f"expected exactly 1 webhook on {WEBHOOK_URL}, got {len(captures)}. "
            f"Agent said: {agent_text!r}"
        )

        body = (captures[0].body or "").lower()
        assert body, (
            f"captured webhook body must be non-empty, got {body!r}. Agent said: {agent_text!r}"
        )
        assert ("no new" in body) or ("queue" in body) or ("later" in body), (
            f"empty-queue webhook body should mention 'no new', 'queue', or 'later'; "
            f"got {body!r}. Agent said: {agent_text!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, (
            f"expected 0 failed_tasks rows after trigger_digest_now, got {len(failed)}: {failed!r}"
        )
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
