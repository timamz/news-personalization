"""
S-conv: conversational-agent smoke test.

Nine scripted user turns that create, configure, modify, query, trim,
and delete a subscription, with a DB assertion after every turn plus an
end-state backstop. No polling, no digest cron, no event verifier, no
inline discovery -- this exercises only the conversational ADK loop and
its DB-modifying + read tools.

The test proves that when scripted instructions are unambiguous and end
with a "don't do anything else, don't ask more questions" directive,
the agent correctly routes to:

  1. ``create_subscription``   -- valid user_spec / retrieval_query.
  2. ``add_source``            -- attaches a Telegram channel.
  3. ``remember``              -- persists a durable user fact.
  4. ``set_user_timezone``     -- updates the user's tz field.
  5. ``update_subscription``   -- rewrites schedule_cron without
                                   clobbering unrelated fields.
  6. ``get_subscriptions``     -- the reply surfaces the sub topic.
  7. ``remove_source``         -- detaches the previously attached source.
  8. ``set_user_language``     -- updates the user's locale without
                                   bleeding into ``digest_language``.
  9. ``delete_subscription``   -- soft-deletes (is_active = False).

Also implicitly exercised: ``close_scenario`` (the agent compacts each
logical task into the transcript's ``compacted_log``).

Failure modes caught: wrong tool selection, hallucinated tool args that
produce ``failed_tasks`` rows, user/sub locale conflation, silent
no-ops, field-wipe on update. Not caught: trajectory length, prose
quality, or ``retrieval_query`` semantics (those live in dedicated
tests). ``trigger_digest_now`` and ``trigger_source_discovery`` are
out of scope here because they run the digest / discovery pipelines.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from news_benchmark.fakes.adapters import FakeAdapter

EXPECTED_TELEGRAM_URL = "https://t.me/s/euenergynews"


TURNS: list[str] = [
    # 1. create_subscription
    "Create a daily digest about EU energy regulation news in English, "
    "delivered at 09:00 UTC. Don't do anything else, don't ask more questions.",
    # 2. add_source
    "Add the Telegram channel euenergynews as a source for that subscription. "
    "Don't do anything else, don't ask more questions.",
    # 3. remember
    "Remember that I'm based in Brussels and care mostly about EU-wide policy, "
    "not national politics. Don't do anything else, don't ask more questions.",
    # 4. set_user_timezone
    "Set my timezone to Europe/Brussels. "
    "Don't do anything else, don't ask more questions.",
    # 5. update_subscription
    "Change that EU energy digest to run at 07:00 UTC instead of 09:00. "
    "Don't do anything else, don't ask more questions.",
    # 6. get_subscriptions
    "What subscriptions do I have? "
    "Don't do anything else, don't ask more questions.",
    # 7. remove_source
    "Remove the Telegram channel euenergynews from that subscription. "
    "Don't do anything else, don't ask more questions.",
    # 8. set_user_language
    "Change my account language to Russian. "
    "Don't do anything else, don't ask more questions.",
    # 9. delete_subscription
    "Delete the EU energy subscription. "
    "Don't do anything else, don't ask more questions.",
]


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


async def _live_subs_for(user_id):
    from news_service.db.session import async_session_factory
    from news_service.models.subscription import Subscription

    async with async_session_factory() as s:
        rows = await s.execute(
            select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.is_active.is_(True),
            )
        )
        return list(rows.scalars().all())


async def _sources_for_sub(sub_id):
    from news_service.db.session import async_session_factory
    from news_service.models.source import Source
    from news_service.models.subscription_source import SubscriptionSource

    async with async_session_factory() as s:
        rows = await s.execute(
            select(Source)
            .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
            .where(SubscriptionSource.subscription_id == sub_id)
        )
        return list(rows.scalars().all())


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
async def test_s_conv_covers_nine_tools(world):
    """Nine-turn transcript covering nine conversational tools with per-turn assertions."""
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    world.adapters[EXPECTED_TELEGRAM_URL] = FakeAdapter(
        source_url=EXPECTED_TELEGRAM_URL, items=[]
    )

    async def _noop_discovery(*_args, **_kwargs):
        return {"status": "skipped", "reason": "S-conv smoke test bypass"}

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

        async def _load_user_and_drive(msg: str) -> str:
            async with async_session_factory() as s:
                user = await s.get(User, user_id)
                assert user is not None, "user disappeared between turns"
                return await _drive_turn(state, user=user, db_session=s, message=msg)

        # --- Turn 1: create_subscription -----------------------------------
        t1 = await _load_user_and_drive(TURNS[0])
        subs = await _live_subs_for(user_id)
        assert len(subs) == 1, (
            f"turn 1: expected 1 active subscription, got {len(subs)}. Agent said: {t1!r}"
        )
        sub = subs[0]
        assert sub.digest_language.lower() == "en", (
            f"turn 1: digest_language should be 'en', got {sub.digest_language!r}"
        )
        assert sub.delivery_mode == "digest", (
            f"turn 1: delivery_mode should be 'digest', got {sub.delivery_mode!r}"
        )
        sub_id = sub.id

        # --- Turn 2: add_source --------------------------------------------
        t2 = await _load_user_and_drive(TURNS[1])
        sources = await _sources_for_sub(sub_id)
        assert len(sources) == 1, (
            f"turn 2: expected 1 source attached to sub, got {len(sources)}. "
            f"Agent said: {t2!r}"
        )
        assert sources[0].url == EXPECTED_TELEGRAM_URL, (
            f"turn 2: attached source URL mismatch: {sources[0].url!r}"
        )

        # --- Turn 3: remember ----------------------------------------------
        t3 = await _load_user_and_drive(TURNS[2])
        u = await _get_user(user_id)
        assert u is not None
        summary = (u.conversation_summary or "").lower()
        assert summary, (
            f"turn 3: conversation_summary should be non-empty. Agent said: {t3!r}"
        )
        assert "brussels" in summary or "eu-wide" in summary or "eu wide" in summary, (
            f"turn 3: conversation_summary should mention Brussels or EU-wide, "
            f"got {summary!r}. Agent said: {t3!r}"
        )

        # --- Turn 4: set_user_timezone -------------------------------------
        t4 = await _load_user_and_drive(TURNS[3])
        u = await _get_user(user_id)
        assert u is not None
        assert u.timezone == "Europe/Brussels", (
            f"turn 4: user.timezone should be 'Europe/Brussels', got {u.timezone!r}. "
            f"Agent said: {t4!r}"
        )

        # --- Turn 5: update_subscription (cron change) --------------------
        t5 = await _load_user_and_drive(TURNS[4])
        refreshed_sub = await _get_sub(sub_id)
        assert refreshed_sub is not None
        cron = (refreshed_sub.schedule_cron or "").strip()
        cron_fields = cron.split()
        assert len(cron_fields) >= 2 and cron_fields[1] == "7", (
            f"turn 5: schedule_cron hour field should be '7' after 07:00 UTC ask, "
            f"got cron={cron!r}. Agent said: {t5!r}"
        )
        assert refreshed_sub.digest_language.lower() == "en", (
            f"turn 5: update must not wipe digest_language, got "
            f"{refreshed_sub.digest_language!r}"
        )
        assert refreshed_sub.is_active is True, (
            "turn 5: update must not deactivate the subscription"
        )

        # --- Turn 6: get_subscriptions (read-back) -------------------------
        t6 = await _load_user_and_drive(TURNS[5])
        assert "energy" in t6.lower(), (
            f"turn 6: agent reply should mention 'energy' after get_subscriptions, "
            f"got {t6!r}"
        )

        # --- Turn 7: remove_source -----------------------------------------
        t7 = await _load_user_and_drive(TURNS[6])
        remaining = await _sources_for_sub(sub_id)
        assert len(remaining) == 0, (
            f"turn 7: expected 0 sources on sub after remove_source, got {len(remaining)}. "
            f"Agent said: {t7!r}"
        )

        # --- Turn 8: set_user_language -------------------------------------
        t8 = await _load_user_and_drive(TURNS[7])
        u = await _get_user(user_id)
        assert u is not None
        assert (u.language or "").lower() == "ru", (
            f"turn 8: user.language should be 'ru', got {u.language!r}. "
            f"Agent said: {t8!r}"
        )
        sub_after_lang = await _get_sub(sub_id)
        assert sub_after_lang is not None
        assert sub_after_lang.digest_language.lower() == "en", (
            f"turn 8: digest_language should stay 'en' after user-locale switch, "
            f"got {sub_after_lang.digest_language!r}"
        )

        # --- Turn 9: delete_subscription -----------------------------------
        t9 = await _load_user_and_drive(TURNS[8])
        active = await _live_subs_for(user_id)
        assert len(active) == 0, (
            f"turn 9: expected 0 active subscriptions after delete, got {len(active)}. "
            f"Agent said: {t9!r}"
        )
        final_sub = await _get_sub(sub_id)
        assert final_sub is not None and final_sub.is_active is False, (
            f"turn 9: sub {sub_id} should be is_active=False, "
            f"got is_active={final_sub.is_active!r}"
        )

        # --- End-state backstop --------------------------------------------
        from news_service.models.failed_task import FailedTask

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"

        final_user = await _get_user(user_id)
        assert final_user is not None
        assert (final_user.language or "").lower() == "ru", (
            f"end-state: user.language should be 'ru', got {final_user.language!r}"
        )
        assert final_user.timezone == "Europe/Brussels", (
            f"end-state: user.timezone should be 'Europe/Brussels', got {final_user.timezone!r}"
        )
        assert (final_user.conversation_summary or "").strip(), (
            "end-state: user.conversation_summary should be non-empty"
        )
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
