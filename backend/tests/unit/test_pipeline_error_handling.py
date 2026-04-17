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
            "news_service.agents.digest.pipeline._build_source_info",
            new_callable=AsyncMock,
            return_value="- source info",
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
    return quality


def _mediocre_quality() -> MagicMock:
    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 3
    quality.format_score = 3
    quality.conciseness = 3
    return quality


def test_should_reflect_fires_on_unhealthy_signals_and_otherwise_stays_quiet() -> None:
    from news_service.agents.digest.pipeline import _should_reflect

    sub_recent = MagicMock()
    sub_recent.last_reflected_at = datetime.now(UTC)
    sub_old = MagicMock()
    sub_old.last_reflected_at = datetime.now(UTC) - timedelta(days=2)
    sub_never = MagicMock()
    sub_never.last_reflected_at = None

    source_a, source_b, source_c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    candidate = MagicMock()
    candidate.source_id = source_a

    # Judge failed (quality is None)
    assert _should_reflect(
        subscription=sub_recent, quality=None, candidates=[], source_ids=set()
    ), "reflector should fire when the judge failed"

    # Low source coverage (1 of 3)
    assert _should_reflect(
        subscription=sub_recent,
        quality=_healthy_quality(),
        candidates=[candidate],
        source_ids={source_a, source_b, source_c},
    ), "reflector should fire when only a small fraction of sources contributed"

    # Mediocre quality scores
    assert _should_reflect(
        subscription=sub_recent,
        quality=_mediocre_quality(),
        candidates=[candidate],
        source_ids={source_a},
    ), "reflector should fire when quality scores fall below the threshold"

    # Never reflected before
    assert _should_reflect(
        subscription=sub_never,
        quality=_healthy_quality(),
        candidates=[candidate],
        source_ids={source_a},
    ), "reflector should fire when it has never run before"

    # Healthy + recently reflected -> skip
    assert not _should_reflect(
        subscription=sub_old,
        quality=_healthy_quality(),
        candidates=[candidate],
        source_ids={source_a},
    ), "reflector should stay quiet when all signals are healthy"
