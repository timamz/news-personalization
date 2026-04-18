"""Tests for the digest pipeline reflector tools."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


def _make_subscription() -> MagicMock:
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = uuid.uuid4()
    sub.user_spec = "Neural network research."
    sub.last_reflected_at = None
    return sub


def _source_ctx(url: str = "https://example.com/feed", user_specified: bool = False) -> MagicMock:
    ctx = MagicMock()
    ctx.source_id = uuid.uuid4()
    ctx.url = url
    ctx.title = "T"
    ctx.is_user_specified = user_specified
    ctx.contribution_count = 0
    ctx.cosine_to_topic = 0.2
    ctx.last_published_at = None
    ctx.days_since_last_published = 40
    return ctx


@pytest.mark.asyncio
async def test_reflector_does_not_remove_user_specified_source() -> None:
    session = _make_session()
    subscription = _make_subscription()
    link = MagicMock()
    link.is_user_specified = True
    link.source_id = uuid.uuid4()

    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = link
    session.execute = AsyncMock(return_value=lookup)

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Reviewed pipeline. All healthy.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text="digest",
            user_spec=subscription.user_spec,
            quality_scores={},
            trigger_reasons=["Drift"],
            source_contexts=[_source_ctx(user_specified=True)],
        )

    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_reflector_returns_shared_state_with_discovery_and_observations() -> None:
    session = _make_session()
    subscription = _make_subscription()
    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Coverage is thin. Triggered discovery.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        shared_state = await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text=f"Digest {uuid.uuid4().hex[:6]}",
            user_spec=subscription.user_spec,
            quality_scores={"relevance": 3, "format_score": 3, "conciseness": 3},
            trigger_reasons=["Drift"],
            source_contexts=[_source_ctx()],
        )

    assert "discovery_triggered" in shared_state and "observations" in shared_state, (
        "reflector did not expose discovery_triggered and observations in shared state"
    )


@pytest.mark.asyncio
async def test_reflector_prompt_carries_trigger_reasons_and_source_contexts() -> None:
    session = _make_session()
    subscription = _make_subscription()
    reason_text = f"drift {uuid.uuid4().hex[:6]}"
    source_url = f"https://example.com/{uuid.uuid4().hex[:6]}"
    captured: dict[str, str] = {}

    async def _capture(*, agent, message, user_id):
        captured["message"] = message
        return "ok"

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new=AsyncMock(side_effect=_capture),
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text="digest",
            user_spec=subscription.user_spec,
            quality_scores={},
            trigger_reasons=[reason_text],
            source_contexts=[_source_ctx(url=source_url)],
        )

    assert reason_text in captured["message"] and source_url in captured["message"], (
        "reflector prompt did not carry both the trigger reason and the source URL"
    )
