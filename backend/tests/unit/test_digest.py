import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.digest import pipeline as digest_pipeline
from news_service.agents.digest.composer import DigestComposition
from news_service.agents.digest.judge import QualityScores
from news_service.agents.digest.planner import DigestPlan
from news_service.agents.digest.reflector import ReflectionResult

logging.disable(logging.CRITICAL)

_PIPELINE = "news_service.agents.digest.pipeline"


def _make_subscription(
    prompt: str,
    embedding: list[float] | None,
    format_instructions: str,
    digest_language: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt=prompt,
        topic_embedding=embedding,
        user_spec=f"## Topic\n{prompt}",
        format_instructions=format_instructions,
        digest_language=digest_language,
    )


def _make_session_with_sources(
    source_ids: list[uuid.UUID],
    sent_rows: list[tuple] | None = None,
) -> SimpleNamespace:
    sent_result = SimpleNamespace(all=lambda: sent_rows or [])
    source_result = SimpleNamespace(all=lambda: [(sid,) for sid in source_ids])
    return SimpleNamespace(
        execute=AsyncMock(side_effect=[sent_result, source_result]),
        flush=AsyncMock(),
        add=lambda x: None,
    )


def _mock_plan() -> DigestPlan:
    return DigestPlan(plan="Cover AI news, 3 items", target_item_count=3)


def _mock_composition(digest_text: str, item_ids: list[str]) -> DigestComposition:
    return DigestComposition(digest_text=digest_text, used_item_ids=item_ids)


def _mock_quality(verdict: str = "PASS", feedback: str = "") -> QualityScores:
    return QualityScores(
        relevance=4, coverage=4, dedup=5, quality=4, verdict=verdict, feedback=feedback
    )


def _mock_reflection() -> ReflectionResult:
    return ReflectionResult(observations="All healthy")


def _patch_pipeline_stages(mocker, digest_text: str, item_ids: list[str]) -> dict:
    """Patch all pipeline stages with default happy-path mocks. Returns mock dict."""
    fake_item = SimpleNamespace(
        id=uuid.uuid4(),
        headline="Test",
        body="Body",
        url="http://test.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )

    mocks = {
        "fetch": mocker.patch(
            f"{_PIPELINE}.fetch_candidate_items",
            new=AsyncMock(return_value=[fake_item]),
        ),
        "build_text": mocker.patch(
            f"{_PIPELINE}.build_items_text",
            return_value="[ID: ...] Headline: Test",
        ),
        "plan": mocker.patch(
            f"{_PIPELINE}.plan_digest",
            new=AsyncMock(return_value=_mock_plan()),
        ),
        "compose": mocker.patch(
            f"{_PIPELINE}.compose_digest",
            new=AsyncMock(return_value=_mock_composition(digest_text, item_ids)),
        ),
        "judge": mocker.patch(
            f"{_PIPELINE}.judge_digest",
            new=AsyncMock(return_value=_mock_quality()),
        ),
        "reflect": mocker.patch(
            f"{_PIPELINE}.reflect_on_pipeline",
            new=AsyncMock(return_value=_mock_reflection()),
        ),
    }
    return mocks


@pytest.mark.asyncio
async def test_generate_digest_returns_composed_digest_text(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Лекции об ИИ {uuid.uuid4().hex[:6]}"
    digest_text = f"Дайджест {uuid.uuid4().hex[:8]}"
    item_id = str(uuid.uuid4())
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "краткая сводка", "ru")
    _patch_pipeline_stages(mocker, digest_text, [item_id])

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result == digest_text, "generate_digest did not return the composed digest text"


@pytest.mark.asyncio
async def test_generate_digest_calls_planner_with_user_spec(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"Научные статьи {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, "детальный обзор", "ru")
    mocks = _patch_pipeline_stages(mocker, f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])

    await digest_pipeline.generate_digest(session, subscription)

    plan_kwargs = mocks["plan"].await_args.kwargs
    assert f"## Topic\n{prompt}" in plan_kwargs["user_spec"], (
        "generate_digest did not pass user_spec to planner"
    )


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources(mocker) -> None:
    session = _make_session_with_sources([])
    prompt = f"Лекции ИИ {uuid.uuid4().hex[:6]}"
    subscription = _make_subscription(prompt, [0.0] * 1536, "краткая сводка", "ru")

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None when no fixed sources exist"


@pytest.mark.asyncio
async def test_generate_digest_returns_none_when_no_candidates(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(
        f"Новости {uuid.uuid4().hex[:6]}", embedding, "краткая сводка", "en"
    )
    mocker.patch(f"{_PIPELINE}.fetch_candidate_items", new=AsyncMock(return_value=[]))

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None when no candidates exist"


@pytest.mark.asyncio
async def test_generate_digest_computes_embedding_when_missing(mocker) -> None:
    source_id = uuid.uuid4()
    prompt = f"Лекции по физике {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, None, "краткая сводка", "ru")
    computed = [random.random() for _ in range(1536)]
    embed_mock = mocker.patch(f"{_PIPELINE}.embed_text", new=AsyncMock(return_value=computed))
    _patch_pipeline_stages(mocker, f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])

    await digest_pipeline.generate_digest(session, subscription)

    assert embed_mock.await_count == 1, (
        "generate_digest did not call embed_text when embedding was missing"
    )


@pytest.mark.asyncio
async def test_generate_digest_stores_computed_embedding(mocker) -> None:
    source_id = uuid.uuid4()
    prompt = f"Квантовые вычисления {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, None, "краткая сводка", "ru")
    computed = [random.random() for _ in range(1536)]
    mocker.patch(f"{_PIPELINE}.embed_text", new=AsyncMock(return_value=computed))
    _patch_pipeline_stages(mocker, f"Текст {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])

    await digest_pipeline.generate_digest(session, subscription)

    assert subscription.topic_embedding == computed, (
        "generate_digest did not store computed embedding on subscription"
    )


@pytest.mark.asyncio
async def test_generate_digest_returns_none_on_planner_failure(mocker) -> None:
    source_id = uuid.uuid4()
    session = _make_session_with_sources([source_id])
    embedding = [random.random() for _ in range(1536)]
    subscription = _make_subscription(
        f"Новости ИИ {uuid.uuid4().hex[:6]}", embedding, "краткая сводка", "en"
    )
    fake_item = SimpleNamespace(
        id=uuid.uuid4(),
        headline="T",
        body="B",
        url="http://t.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    mocker.patch(f"{_PIPELINE}.fetch_candidate_items", new=AsyncMock(return_value=[fake_item]))
    mocker.patch(f"{_PIPELINE}.build_items_text", return_value="items")
    mocker.patch(
        f"{_PIPELINE}.plan_digest",
        new=AsyncMock(side_effect=RuntimeError("planner failed")),
    )

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None on planner failure"


@pytest.mark.asyncio
async def test_generate_digest_revises_on_judge_feedback(mocker) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(
        f"Новости {uuid.uuid4().hex[:6]}", embedding, "краткая сводка", "en"
    )

    fake_item = SimpleNamespace(
        id=uuid.uuid4(),
        headline="T",
        body="B",
        url="http://t.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    mocker.patch(f"{_PIPELINE}.fetch_candidate_items", new=AsyncMock(return_value=[fake_item]))
    mocker.patch(f"{_PIPELINE}.build_items_text", return_value="items")
    mocker.patch(f"{_PIPELINE}.plan_digest", new=AsyncMock(return_value=_mock_plan()))
    mocker.patch(f"{_PIPELINE}.reflect_on_pipeline", new=AsyncMock(return_value=_mock_reflection()))

    first_text = f"Draft {uuid.uuid4().hex[:6]}"
    revised_text = f"Revised {uuid.uuid4().hex[:6]}"
    compose_mock = mocker.patch(
        f"{_PIPELINE}.compose_digest",
        new=AsyncMock(
            side_effect=[
                _mock_composition(first_text, [str(uuid.uuid4())]),
                _mock_composition(revised_text, [str(uuid.uuid4())]),
            ]
        ),
    )
    mocker.patch(
        f"{_PIPELINE}.judge_digest",
        new=AsyncMock(
            side_effect=[
                _mock_quality(verdict="REVISE", feedback="Too long"),
                _mock_quality(verdict="PASS"),
            ]
        ),
    )

    result = await digest_pipeline.generate_digest(session, subscription)

    assert compose_mock.await_count == 2, (
        "generate_digest did not revise digest after judge feedback"
    )
    assert result == revised_text, "generate_digest did not return revised text"
