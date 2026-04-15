"""Tests for the reflector's remove_source and trigger_discovery tools."""

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
    sub.user_spec = "## Topic\nRecherche sur les réseaux de neurones"
    sub.last_reflected_at = None
    return sub


@pytest.mark.asyncio
async def test_remove_source_rejects_user_specified_source(_session, _subscription):
    link = MagicMock()
    link.is_user_specified = True
    link.source_id = uuid.uuid4()

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = link
    _session.execute = AsyncMock(return_value=result_mock)

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Reviewed pipeline. All healthy.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=_session,
            subscription=_subscription,
            digest_text="Les dernières nouvelles de ML",
            user_spec=_subscription.user_spec,
            quality_scores={},
            source_info="- source [user-specified] — 5 candidates",
        )

        _session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_discovery_sets_shared_state_flag(_session, _subscription):
    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Coverage is thin. Triggered discovery.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        shared_state = await run_reflector(
            db_session=_session,
            subscription=_subscription,
            digest_text="Краткий дайджест",
            user_spec=_subscription.user_spec,
            quality_scores={"relevance": 3, "format_score": 3, "conciseness": 3},
            source_info="- source [auto-discovered] — 0 candidates",
        )

        assert isinstance(shared_state, dict), "run_reflector should return a dict"
        assert "discovery_triggered" in shared_state, (
            "shared state should have discovery_triggered key"
        )
        assert "observations" in shared_state, "shared state should have observations key"
