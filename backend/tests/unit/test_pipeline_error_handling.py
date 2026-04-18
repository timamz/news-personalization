"""Tests for digest pipeline error tiers and the reflector trigger heuristic."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.core.exceptions import DigestPipelineError


@pytest.fixture
def _subscription():
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = uuid.uuid4()
    sub.user_spec = "ML research. Brief digest."
    sub.topic_embedding = [0.1] * 1536
    sub.digest_language = "de"
    sub.schedule_cron = "0 8 * * *"
    sub.last_reflected_at = datetime.now(UTC)
    return sub


@pytest.fixture
def _session():
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    return session


def _candidate():
    candidate = MagicMock()
    candidate.id = uuid.uuid4()
    candidate.source_id = uuid.uuid4()
    candidate.embedding = [0.2] * 1536
    return candidate


@pytest.mark.asyncio
async def test_writer_failure_raises_pipeline_error(_subscription, _session) -> None:
    candidate = _candidate()
    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch("news_service.agents.digest.pipeline.build_items_text", return_value="items"),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.write_digest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM timeout"),
        ),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        with pytest.raises(DigestPipelineError, match="Writer failed"):
            await generate_digest(_session, _subscription)


@pytest.mark.asyncio
async def test_judge_failure_falls_back_to_unreviewed_draft(_subscription, _session) -> None:
    candidate = _candidate()
    composition = MagicMock()
    composition.digest_text = "draft text"
    composition.used_item_ids = [str(candidate.id)]

    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch("news_service.agents.digest.pipeline.build_items_text", return_value="items"),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.write_digest",
            new_callable=AsyncMock,
            return_value=composition,
        ),
        patch(
            "news_service.agents.digest.pipeline.judge_digest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("judge model unavailable"),
        ),
        patch(
            "news_service.agents.digest.pipeline._load_source_contexts",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "news_service.agents.digest.pipeline.run_reflector",
            new_callable=AsyncMock,
            return_value={"discovery_triggered": False},
        ),
        patch("news_service.agents.digest.pipeline._mark_as_sent", new_callable=AsyncMock),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        assert await generate_digest(_session, _subscription) == "draft text"


@pytest.mark.asyncio
async def test_reflector_failure_does_not_block_digest(_subscription, _session) -> None:
    _subscription.last_reflected_at = None
    candidate = _candidate()
    composition = MagicMock()
    composition.digest_text = "digest"
    composition.used_item_ids = [str(candidate.id)]

    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 5
    quality.format_score = 5
    quality.conciseness = 5
    quality.model_dump.return_value = {}

    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch("news_service.agents.digest.pipeline.build_items_text", return_value="items"),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.write_digest",
            new_callable=AsyncMock,
            return_value=composition,
        ),
        patch(
            "news_service.agents.digest.pipeline.judge_digest",
            new_callable=AsyncMock,
            return_value=quality,
        ),
        patch(
            "news_service.agents.digest.pipeline._load_source_contexts",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "news_service.agents.digest.pipeline.run_reflector",
            new_callable=AsyncMock,
            side_effect=RuntimeError("reflector crashed"),
        ),
        patch("news_service.agents.digest.pipeline._mark_as_sent", new_callable=AsyncMock),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        assert await generate_digest(_session, _subscription) == "digest"


def _healthy_quality() -> MagicMock:
    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 5
    quality.format_score = 5
    quality.conciseness = 5
    quality.feedback = ""
    return quality


def _revise_quality() -> MagicMock:
    quality = MagicMock()
    quality.verdict = "REVISE"
    quality.relevance = 2
    quality.format_score = 3
    quality.conciseness = 3
    quality.feedback = f"Too verbose {uuid.uuid4().hex[:6]}"
    return quality


def _source_ctx(
    *,
    cos: float | None = 0.8,
    days_since: int | None = 1,
    streak: int = 0,
    user_specified: bool = False,
):
    from news_service.agents.digest.pipeline import ReflectorSourceContext

    return ReflectorSourceContext(
        source_id=uuid.uuid4(),
        url=f"https://example.com/{uuid.uuid4().hex[:6]}",
        title="T",
        is_user_specified=user_specified,
        contribution_count=1,
        cosine_to_topic=cos,
        last_published_at=datetime.now(UTC) - timedelta(days=days_since or 0),
        days_since_last_published=days_since,
        contributed_last_30_digests=5,
        contribution_rate=0.1,
        digests_since_last_contribution=streak,
        item_cosine_p50=0.4,
        item_cosine_p90=0.7,
        item_cosine_std=0.15,
    )


def test_compute_reflect_reasons_lists_drift_staleness_streak_and_revise_signals() -> None:
    from news_service.agents.digest.pipeline import _compute_reflect_reasons

    reasons_revise = _compute_reflect_reasons(
        quality=_revise_quality(),
        source_contexts=[_source_ctx()],
    )
    reasons_drift = _compute_reflect_reasons(
        quality=_healthy_quality(),
        source_contexts=[_source_ctx(cos=0.15)],
    )
    reasons_stale = _compute_reflect_reasons(
        quality=_healthy_quality(),
        source_contexts=[_source_ctx(days_since=90)],
    )
    reasons_streak = _compute_reflect_reasons(
        quality=_healthy_quality(),
        source_contexts=[_source_ctx(streak=15)],
    )
    reasons_none = _compute_reflect_reasons(
        quality=_healthy_quality(),
        source_contexts=[_source_ctx()],
    )

    assert (
        any("REVISE" in r for r in reasons_revise)
        and any("drifted" in r for r in reasons_drift)
        and any("not published" in r for r in reasons_stale)
        and any("consecutive digests" in r for r in reasons_streak)
        and reasons_none == []
    ), "reflect-reasons did not fire one reason per signal and stay silent when healthy"
