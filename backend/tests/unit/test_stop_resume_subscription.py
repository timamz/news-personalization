"""Tests for the stop_subscription and resume_subscription tools.

Covers the pause / resume lifecycle: stopping flips ``paused_at`` to a
timestamp, resuming clears it, and the active-subscription cap counts
only running (active and not paused) subscriptions.
"""

import logging
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.conversational import create_conversational_agent

logging.disable(logging.CRITICAL)


def _fake_user() -> SimpleNamespace:
    """Build a fully populated user namespace for tool wiring."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        timezone="Europe/Moscow",
        language="en",
        delivery_webhook_url=None,
        conversation_summary="",
        has_onboarded=True,
    )


class _FakeSessionFactory:
    """Async context manager factory yielding a single shared session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __call__(self) -> Any:
        mgr = AsyncMock()
        mgr.__aenter__ = AsyncMock(return_value=self._session)
        mgr.__aexit__ = AsyncMock(return_value=None)
        return mgr


def _db_session_with_running_count(count: int) -> AsyncMock:
    """Build a db_session whose first execute returns the given scalar count."""
    session = AsyncMock()
    count_result = MagicMock()
    count_result.scalar_one.return_value = count
    session.execute = AsyncMock(return_value=count_result)
    return session


def _build_agent(
    *,
    user: SimpleNamespace,
    factory_session: Any,
    running_count: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Wire up the conversational agent with a controlled session graph."""
    return create_conversational_agent(
        db_session=_db_session_with_running_count(running_count),
        user=user,
        conversation_summary="",
        session_factory=_FakeSessionFactory(factory_session),
    )


def _get_tool(agent: Any, name: str):
    """Locate a tool callable by name in the ADK agent."""
    return next(t for t in agent.tools if callable(t) and t.__name__ == name)


def _running_subscription(user_id: uuid.UUID) -> MagicMock:
    """Create a mock subscription row in the running state."""
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user_id
    sub.is_active = True
    sub.paused_at = None
    return sub


def _stopped_subscription(user_id: uuid.UUID) -> MagicMock:
    """Create a mock subscription row already in the paused state."""
    from datetime import UTC, datetime

    sub = _running_subscription(user_id)
    sub.paused_at = datetime.now(UTC)
    return sub


@pytest.mark.asyncio
async def test_stop_subscription_stamps_paused_at_on_a_running_subscription(mocker) -> None:
    user = _fake_user()
    sub = _running_subscription(user.id)
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    mocker.patch(
        "news_service.agents.conversational.tools._gate_with_confirmation",
        new=AsyncMock(return_value=(True, "")),
    )

    agent, _ = _build_agent(user=user, factory_session=scoped)
    result = await _get_tool(agent, "stop_subscription")(str(sub.id))
    assert sub.paused_at is not None and scoped.commit.await_count == 1 and "stopped" in result, (
        "stop_subscription must persist a non-NULL paused_at and confirm to the agent"
    )


@pytest.mark.asyncio
async def test_stop_subscription_cannot_be_invoked_without_confirmation(mocker) -> None:
    mocker.patch(
        "news_service.agents.conversational.tools.create_pending",
        new=AsyncMock(return_value="nonce-token"),
    )
    user = _fake_user()
    scoped = AsyncMock()
    scoped.commit = AsyncMock()

    agent, _ = _build_agent(user=user, factory_session=scoped)
    result = await _get_tool(agent, "stop_subscription")(str(uuid.uuid4()))
    assert result.startswith("REQUIRES_CONFIRMATION:") and scoped.commit.await_count == 0, (
        "stop_subscription must refuse to mutate without a valid confirmation_token"
    )


@pytest.mark.asyncio
async def test_stop_subscription_dont_find_unknown_subscription(mocker) -> None:
    user = _fake_user()
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = None
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    mocker.patch(
        "news_service.agents.conversational.tools._gate_with_confirmation",
        new=AsyncMock(return_value=(True, "")),
    )

    agent, _ = _build_agent(user=user, factory_session=scoped)
    unknown_id = str(uuid.uuid4())
    result = await _get_tool(agent, "stop_subscription")(unknown_id)
    assert "not found" in result and scoped.commit.await_count == 0, (
        "stop_subscription cannot mutate when the subscription does not exist"
    )


@pytest.mark.asyncio
async def test_stop_subscription_refuses_a_soft_deleted_subscription(mocker) -> None:
    user = _fake_user()
    sub = _running_subscription(user.id)
    sub.is_active = False
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    mocker.patch(
        "news_service.agents.conversational.tools._gate_with_confirmation",
        new=AsyncMock(return_value=(True, "")),
    )

    agent, _ = _build_agent(user=user, factory_session=scoped)
    result = await _get_tool(agent, "stop_subscription")(str(sub.id))
    assert (
        "already deleted" in result and sub.paused_at is None and scoped.commit.await_count == 0
    ), "stop_subscription cannot operate on a soft-deleted subscription"


@pytest.mark.asyncio
async def test_stop_subscription_refuses_a_subscription_that_is_already_paused(mocker) -> None:
    user = _fake_user()
    sub = _stopped_subscription(user.id)
    original_paused = sub.paused_at
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    mocker.patch(
        "news_service.agents.conversational.tools._gate_with_confirmation",
        new=AsyncMock(return_value=(True, "")),
    )

    agent, _ = _build_agent(user=user, factory_session=scoped)
    result = await _get_tool(agent, "stop_subscription")(str(sub.id))
    assert (
        "already stopped" in result
        and sub.paused_at == original_paused
        and scoped.commit.await_count == 0
    ), "stop_subscription must not re-stamp paused_at when it is already set"


@pytest.mark.asyncio
async def test_resume_subscription_clears_paused_at_when_under_cap() -> None:
    user = _fake_user()
    sub = _stopped_subscription(user.id)
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent(user=user, factory_session=scoped, running_count=2)
    result = await _get_tool(agent, "resume_subscription")(str(sub.id))
    assert sub.paused_at is None and scoped.commit.await_count == 1 and "resumed" in result, (
        "resume_subscription must NULL paused_at when the running cap is not reached"
    )


@pytest.mark.asyncio
async def test_resume_subscription_refuses_when_the_user_has_five_running_subs() -> None:
    user = _fake_user()
    sub = _stopped_subscription(user.id)
    original_paused = sub.paused_at
    scoped = AsyncMock()
    scoped.execute = AsyncMock()
    scoped.commit = AsyncMock()

    agent, _ = _build_agent(user=user, factory_session=scoped, running_count=5)
    result = await _get_tool(agent, "resume_subscription")(str(sub.id))
    assert (
        "subscription limit reached" in result
        and "stop_subscription" in result
        and sub.paused_at == original_paused
        and scoped.commit.await_count == 0
    ), (
        "resume_subscription must refuse when running cap is hit and instruct the "
        "agent to stop or delete one before retrying"
    )


@pytest.mark.asyncio
async def test_resume_subscription_reports_already_running_when_paused_at_is_null() -> None:
    user = _fake_user()
    sub = _running_subscription(user.id)
    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = sub
    scoped = AsyncMock()
    scoped.execute = AsyncMock(return_value=lookup)
    scoped.commit = AsyncMock()

    agent, _ = _build_agent(user=user, factory_session=scoped, running_count=1)
    result = await _get_tool(agent, "resume_subscription")(str(sub.id))
    assert "already running" in result and scoped.commit.await_count == 0, (
        "resume_subscription must not commit when the subscription is already running"
    )


@pytest.mark.asyncio
async def test_create_subscription_allows_a_new_sub_when_only_stopped_subs_exist(mocker) -> None:
    """The active-subscription cap counts only running subs, not paused ones.

    Five stopped subscriptions plus zero running ones must still allow a
    sixth creation -- otherwise the user would be forced to delete data
    that they explicitly chose to keep when stopping.
    """
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

    agent, shared_state = _build_agent(
        user=user,
        factory_session=scoped,
        running_count=0,
    )
    spec = f"news {uuid.uuid4().hex[:6]}. Brief bullets."
    query = f"world news {uuid.uuid4().hex[:6]}"
    result = await _get_tool(agent, "create_subscription")(
        user_spec=spec,
        retrieval_query=query,
        delivery_mode="digest",
        include_discovered_sources=False,
    )
    assert (
        ": created" in result
        and shared_state["created_subscription_id"] is not None
        and scoped.commit.await_count >= 1
    ), (
        "create_subscription must succeed when running subs are below the cap, "
        "even if the user has stopped subscriptions sitting in the database"
    )
