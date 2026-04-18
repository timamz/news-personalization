"""Tests for the on-send contribution-streak maintenance on subscription_sources."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.digest.pipeline import _update_contribution_streaks


@pytest.mark.asyncio
async def test_streak_resets_for_contributing_sources_and_increments_for_the_rest() -> None:
    contributing_id = uuid.uuid4()
    silent_id = uuid.uuid4()

    contributing_link = SimpleNamespace(
        source_id=contributing_id,
        digests_since_last_contribution=9,
    )
    silent_link = SimpleNamespace(
        source_id=silent_id,
        digests_since_last_contribution=4,
    )

    scalars_result = SimpleNamespace(all=lambda: [contributing_link, silent_link])
    query_result = SimpleNamespace(scalars=lambda: scalars_result)
    session = SimpleNamespace(
        execute=AsyncMock(return_value=query_result),
        flush=AsyncMock(),
    )

    await _update_contribution_streaks(
        session=session,
        subscription_id=uuid.uuid4(),
        contributing_source_ids={contributing_id},
    )

    assert (
        contributing_link.digests_since_last_contribution == 0
        and silent_link.digests_since_last_contribution == 5
    ), "streak was not reset for contributing sources or incremented for silent ones"
