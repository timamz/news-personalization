"""
S-conv-update-digest-language: the "don't bleed into user_language" trap.

One scripted user turn asks the conversational agent to switch the
subscription's digest output language from English to Russian while
explicitly leaving the account locale untouched. The prompt
distinguishes ``set_user_language`` (User.language -- account locale)
from changing a subscription's digest language (which lives in
``Subscription.digest_language`` or inside ``user_spec``). This test
guards the trap where an agent, upon hearing "Russian", conflates the
two and rewrites ``user.language``.

Accepted mechanisms for the per-subscription switch:

  * ``Subscription.digest_language`` changes to ``"ru"``, or
  * ``Subscription.user_spec`` gains a Russian-language instruction
    (case-insensitive match on "russian" or the Cyrillic root "russk").

The trap assertion is that ``user.language`` stays ``"en"`` -- i.e.
``set_user_language`` was NOT invoked.

Out of scope: trajectory length, prose quality, retrieval_query
semantics, digest rendering. Only the tool-routing decision is
exercised here.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


WEBHOOK_URL = "https://bench.invalid/webhook/s-conv-update-digest-language"


TURN = (
    "From now on I want that subscription's digest text delivered in Russian "
    "instead of English. My account language stays English. "
    "Don't do anything else, don't ask more questions."
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


async def _get_user(user_id):
    from news_service.db.session import async_session_factory
    from news_service.models.user import User

    async with async_session_factory() as s:
        return await s.get(User, user_id)


async def _get_sub(sub_id):
    from news_service.db.session import async_session_factory
    from news_service.models.subscription import Subscription

    async with async_session_factory() as s:
        return await s.get(Subscription, sub_id)


@pytest.mark.asyncio
async def test_s_conv_update_digest_language_does_not_touch_user_language(world):
    """Switching the digest's output language must not rewrite User.language."""
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
            "reason": "S-conv-update-digest-language smoke test bypass",
        }

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _noop_discovery  # type: ignore[assignment]

    try:
        user_id = uuid.uuid4()
        sub_id = uuid.uuid4()

        topic_embedding = await embed_text("EU energy policy news")

        original_user_spec = (
            "Give me the digest in English, concise and friendly tone, "
            "focused on EU energy policy."
        )
        original_schedule_cron = "0 9 * * *"
        original_delivery_mode = "digest"

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
                Subscription(
                    id=sub_id,
                    user_id=user_id,
                    user_spec=original_user_spec,
                    topic_embedding=topic_embedding,
                    delivery_mode=original_delivery_mode,
                    digest_language="en",
                    schedule_cron=original_schedule_cron,
                    delivery_webhook_url=WEBHOOK_URL,
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
                message=TURN,
            )

        sub_after = await _get_sub(sub_id)
        assert sub_after is not None, (
            f"subscription {sub_id} should still exist after the turn. "
            f"Agent said: {agent_text!r}"
        )
        assert sub_after.is_active is True, (
            f"subscription should remain active after a language update, "
            f"got is_active={sub_after.is_active!r}. Agent said: {agent_text!r}"
        )

        digest_lang = (sub_after.digest_language or "").lower()
        spec_lower = (sub_after.user_spec or "").lower()
        spec_mentions_russian = ("russian" in spec_lower) or ("русск" in spec_lower)
        assert digest_lang == "ru" or spec_mentions_russian, (
            f"expected Russian to be encoded either in digest_language ('ru') or "
            f"in user_spec (mentioning 'russian'/'русск'); got "
            f"digest_language={sub_after.digest_language!r}, "
            f"user_spec={sub_after.user_spec!r}. Agent said: {agent_text!r}"
        )

        user_after = await _get_user(user_id)
        assert user_after is not None
        assert (user_after.language or "").lower() == "en", (
            f"user.language must stay 'en' -- set_user_language was wrongly invoked. "
            f"got {user_after.language!r}. Agent said: {agent_text!r}"
        )

        assert sub_after.schedule_cron == original_schedule_cron, (
            f"schedule_cron must not change on a language-only update; expected "
            f"{original_schedule_cron!r}, got {sub_after.schedule_cron!r}. "
            f"Agent said: {agent_text!r}"
        )
        assert sub_after.delivery_mode == original_delivery_mode, (
            f"delivery_mode must not change on a language-only update; expected "
            f"{original_delivery_mode!r}, got {sub_after.delivery_mode!r}. "
            f"Agent said: {agent_text!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, (
            f"expected 0 failed_tasks rows after language update, got {len(failed)}: "
            f"{failed!r}"
        )
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
