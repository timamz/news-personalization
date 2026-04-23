"""
S-conv-feedback-rewrites-spec: digest-quality feedback turns into a
``user_spec`` rewrite via ``update_subscription``.

The conversational prompt is explicit about this behaviour: "When the
user gives feedback about digest quality, call update_subscription with
a rewritten user_spec that captures what they want (reuse the existing
parts, change only what feedback addresses)." This one-turn smoke test
pins that down.

We seed a subscription whose ``user_spec`` carries clearly identifiable
tokens ("ENTSO-E", "ACER", "EUR-Lex", "EU energy") so we can tell a
rewrite apart from a wholesale replacement. The user then asks for
fewer but deeper items. The agent should pick ``update_subscription``,
write a new ``user_spec`` that folds in the feedback vocabulary
(longer / analytical / fewer / etc.) while keeping at least one of the
original focus tokens -- proving the "reuse existing parts, change only
what feedback addresses" half of the instruction.

Out of scope: the exact prose of the rewrite, any re-embedding of
``topic_embedding``, and any downstream digest behaviour. We only care
here that (a) the agent routes feedback to ``update_subscription``,
(b) the new ``user_spec`` reflects the requested change, and (c) the
non-feedback parts of the original spec survive the rewrite.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


ORIGINAL_USER_SPEC = (
    "Daily EU energy policy digest. 5-6 items max. Plain text, no markdown. "
    "Delivered at 9am UTC. Focus on ENTSO-E, ACER, EUR-Lex. English."
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
async def test_s_conv_feedback_rewrites_user_spec(world):
    """Digest-quality feedback should trigger ``update_subscription`` with a rewritten spec.

    Seeds a user + subscription with an original ``user_spec`` that has
    identifiable tokens, drives a single user turn asking for fewer but
    deeper items, and verifies both that the new spec picks up the
    feedback vocabulary and that at least one original focus token
    survives the rewrite.
    """
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.subscription import Subscription
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    async def _noop_discovery(*_args, **_kwargs):
        return {
            "status": "skipped",
            "reason": "S-conv-feedback-rewrites-spec smoke test bypass",
        }

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _noop_discovery  # type: ignore[assignment]

    try:
        user_id = uuid.uuid4()
        sub_id = uuid.uuid4()

        topic_embedding = await embed_text(
            "EU energy policy regulation ENTSO-E ACER EUR-Lex"
        )

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
                    delivery_mode="digest",
                    digest_language="en",
                    schedule_cron="0 9 * * *",
                    is_active=True,
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
                    "The last digest had too many short punchy items. I want "
                    "fewer items but each one should go deeper -- longer "
                    "analytical summaries instead of news-wire one-liners. "
                    "Don't do anything else, don't ask more questions."
                ),
            )

        async with async_session_factory() as s:
            refreshed = await s.get(Subscription, sub_id)
        assert refreshed is not None, (
            f"subscription {sub_id} disappeared after the feedback turn. "
            f"Agent said: {agent_text!r}"
        )
        assert refreshed.is_active is True, (
            f"feedback turn must not deactivate the subscription, "
            f"got is_active={refreshed.is_active!r}. Agent said: {agent_text!r}"
        )

        new_spec = refreshed.user_spec or ""
        new_spec_lower = new_spec.lower()

        assert new_spec.strip() and new_spec != ORIGINAL_USER_SPEC, (
            f"user_spec should have been rewritten, but it is unchanged "
            f"(len={len(new_spec)}). Agent said: {agent_text!r}"
        )

        feedback_tokens = ("longer", "analytical", "deeper", "in-depth", "fewer", "analysis")
        assert any(tok in new_spec_lower for tok in feedback_tokens), (
            f"rewritten user_spec should reflect feedback vocabulary "
            f"(any of {feedback_tokens!r}); got {new_spec!r}. "
            f"Agent said: {agent_text!r}"
        )

        preservation_tokens = ("entso", "acer", "eur-lex", "eu energy")
        assert any(tok in new_spec_lower for tok in preservation_tokens), (
            f"rewritten user_spec should preserve at least one original focus "
            f"token (any of {preservation_tokens!r}); got {new_spec!r}. "
            f"Agent said: {agent_text!r}"
        )

        assert (refreshed.schedule_cron or "").strip() == "0 9 * * *", (
            f"schedule_cron should be unchanged by a spec-rewrite feedback turn, "
            f"got {refreshed.schedule_cron!r}. Agent said: {agent_text!r}"
        )
        assert refreshed.delivery_mode == "digest", (
            f"delivery_mode should be unchanged by a spec-rewrite feedback turn, "
            f"got {refreshed.delivery_mode!r}. Agent said: {agent_text!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, (
            f"expected 0 failed_tasks rows after a feedback-rewrite turn, "
            f"got {len(failed)}: {failed!r}"
        )
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
