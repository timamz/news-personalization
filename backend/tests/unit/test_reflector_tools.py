"""Tests for the digest pipeline reflector tools."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _session():
    session = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def _subscription():
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = uuid.uuid4()
    sub.user_spec = "Neural network research."
    sub.last_reflected_at = None
    return sub


@pytest.mark.asyncio
async def test_reflector_does_not_remove_user_specified_source(_session, _subscription) -> None:
    link = MagicMock()
    link.is_user_specified = True
    link.source_id = uuid.uuid4()

    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = link
    _session.execute = AsyncMock(return_value=lookup)

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Reviewed pipeline. All healthy.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=_session,
            subscription=_subscription,
            digest_text="digest",
            user_spec=_subscription.user_spec,
            quality_scores={},
            source_info="- source [user-specified] — 5 candidates",
        )

    _session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_reflector_returns_shared_state_with_discovery_and_observations(
    _session, _subscription
) -> None:
    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Coverage is thin. Triggered discovery.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        shared_state = await run_reflector(
            db_session=_session,
            subscription=_subscription,
            digest_text=f"Digest {uuid.uuid4().hex[:6]}",
            user_spec=_subscription.user_spec,
            quality_scores={"relevance": 3, "format_score": 3, "conciseness": 3},
            source_info="- source [auto-discovered] — 0 candidates",
        )

    assert "discovery_triggered" in shared_state and "observations" in shared_state, (
        "reflector did not expose discovery_triggered and observations in shared state"
    )
