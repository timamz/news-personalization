"""
S-conv-trigger-discovery: the conversational agent's ``trigger_source_discovery``
tool plumbing test.

Scope
-----
Covers ``news_service.agents.conversational.tools.trigger_source_discovery``
end-to-end through one scripted user turn. The user asks for more sources,
the conversational ADK agent must select ``trigger_source_discovery`` with
the correct ``subscription_id`` and a non-empty freeform ``reason`` paragraph,
and the tool must then dispatch to ``_run_inline_discovery``.

Why a spy (not real discovery)
------------------------------
``_run_inline_discovery`` would otherwise call ``run_and_persist_discovery``,
which exercises the Discovery Agent, Source Finders, Yandex Cloud Search
and the semantic-search / pgvector stack. The semantic-search fake for
the benchmark harness is a separate future project. We therefore replace
``_run_inline_discovery`` with a spy that records every invocation and
returns a benign non-ok payload. This keeps the test focused on the
plumbing: agent -> tool -> downstream dispatch.

Assertions
----------
  * spy was invoked at least once (the agent routed to the tool),
  * spy was invoked with the seeded ``subscription_id`` (right sub),
  * the ``reason`` the agent synthesised is a non-empty paragraph,
  * no ``FailedTask`` rows were produced (no tool-arg hallucinations).

Out of scope
------------
Real discovery orchestrator behaviour, Source Finder trajectories, the
selection / backfill logic, and the specific wording the agent chooses
for ``reason``. Those belong in dedicated tests with a proper
semantic-search fake.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


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
async def test_s_conv_trigger_source_discovery_invokes_discovery(world):
    """Agent routes an explicit 'find more sources' ask to trigger_source_discovery."""
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    discovery_calls: list[dict] = []

    async def _spy_discovery(*, scoped_factory, subscription_id, reason, shared_state):
        discovery_calls.append(
            {
                "subscription_id": str(subscription_id),
                "reason": reason,
            }
        )
        return {
            "status": "stubbed",
            "added_urls": [],
            "discarded": [],
            "reason": "S-conv-trigger-discovery test bypass",
        }

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _spy_discovery  # type: ignore[assignment]

    try:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        sub_id = uuid.uuid4()
        source_url = "https://brussels-energy-policy.invalid/feed.xml"
        webhook_url = "https://bench.invalid/webhook/s-conv-trigger-discovery"

        topic_vector = await embed_text("EU energy regulation news")
        user_spec = (
            "# EU energy policy digest\n\n"
            "I want news about European Union energy regulation, especially\n"
            "Council decisions, Commission proposals, and ACER / ENTSO-E\n"
            "announcements. Keep it policy-flavoured; skip purely national\n"
            "political coverage and retail-market price reporting.\n"
            "\n"
            "Present as short bulletised digest in English.\n"
        )

        async with async_session_factory() as s:
            s.add(
                User(
                    id=user_id,
                    api_key=f"bench-{user_id.hex}",
                    language="en",
                    timezone="UTC",
                    delivery_webhook_url=webhook_url,
                    has_onboarded=True,
                )
            )
            s.add(
                Source(
                    id=source_id,
                    url=source_url,
                    title="Brussels Energy Policy Tracker",
                    source_description="EU-level energy policy newswire.",
                    is_active=True,
                )
            )
            s.add(
                Subscription(
                    id=sub_id,
                    user_id=user_id,
                    topic_embedding=topic_vector,
                    user_spec=user_spec,
                    delivery_mode="digest",
                    schedule_cron="0 8 * * *",
                    digest_language="en",
                    delivery_webhook_url=webhook_url,
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
        message = (
            "Please find additional EU energy policy sources for my subscription "
            "- I want better coverage of Council and Commission announcements. "
            "Don't do anything else, don't ask more questions."
        )

        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None, "seeded user vanished before the turn"
            agent_text = await _drive_turn(state, user=user, db_session=s, message=message)

        assert len(discovery_calls) >= 1, (
            "expected trigger_source_discovery to dispatch to _run_inline_discovery "
            f"at least once; spy was never called. Agent said: {agent_text!r}"
        )

        first_call = discovery_calls[0]
        assert first_call["subscription_id"] == str(sub_id), (
            "trigger_source_discovery dispatched for the wrong subscription: "
            f"expected {sub_id}, got {first_call['subscription_id']}. "
            f"Agent said: {agent_text!r}"
        )

        assert len(first_call["reason"].strip()) > 0, (
            "trigger_source_discovery was dispatched with an empty reason paragraph; "
            f"the tool should have rejected this before dispatch. Agent said: {agent_text!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, (
            f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}. "
            f"Agent said: {agent_text!r}"
        )
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
