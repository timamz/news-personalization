import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.digest import pipeline as digest_pipeline
from news_service.agents.digest.judge import QualityScores
from news_service.agents.digest.writer import DigestComposition
from news_service.core.exceptions import DigestPipelineError

logging.disable(logging.CRITICAL)

_PIPELINE = "news_service.agents.digest.pipeline"
_FMT_RU = "\u0441\u0432\u043e\u0434\u043a\u0430"
_FMT_DETAIL = "\u043e\u0431\u0437\u043e\u0440"


def _make_subscription(
    prompt: str,
    embedding: list[float] | None,
    format_instructions: str,
    digest_language: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        topic_embedding=embedding,
        user_spec=f"## Topic\n{prompt}\n\n## Preferences\n{format_instructions}",
        digest_language=digest_language,
        last_reflected_at=datetime.now(UTC),
    )


def _make_session_with_sources(
    source_ids: list[uuid.UUID],
    sent_rows: list[tuple] | None = None,
) -> SimpleNamespace:
    sent_result = SimpleNamespace(all=lambda: sent_rows or [])
    source_result = SimpleNamespace(
        all=lambda: [(sid,) for sid in source_ids],
    )
    recent_digest_result = SimpleNamespace(all=lambda: [])
    return SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                sent_result,
                source_result,
                recent_digest_result,
            ],
        ),
        flush=AsyncMock(),
        add=lambda x: None,
    )


def _mock_composition(digest_text: str, item_ids: list[str]) -> DigestComposition:
    return DigestComposition(digest_text=digest_text, used_item_ids=item_ids)


def _mock_quality(verdict: str = "PASS", feedback: str = "") -> QualityScores:
    return QualityScores(
        relevance=4,
        format_score=4,
        conciseness=5,
        verdict=verdict,
        feedback=feedback,
    )


def _patch_pipeline_stages(mocker, digest_text: str, item_ids: list[str]) -> dict:
    """Patch pipeline stages with happy-path mocks."""
    source_id = uuid.uuid4()
    fake_item = SimpleNamespace(
        id=uuid.UUID(item_ids[0]) if item_ids else uuid.uuid4(),
        source_id=source_id,
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
        "write": mocker.patch(
            f"{_PIPELINE}.write_digest",
            new=AsyncMock(
                return_value=_mock_composition(digest_text, item_ids),
            ),
        ),
        "judge": mocker.patch(
            f"{_PIPELINE}.judge_digest",
            new=AsyncMock(return_value=_mock_quality()),
        ),
    }
    return mocks


@pytest.mark.asyncio
async def test_generate_digest_returns_composed_digest_text(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    tag = uuid.uuid4().hex[:6]
    prompt = f"\u041b\u0435\u043a\u0446\u0438\u0438 \u0418\u0418 {tag}"
    digest_text = f"\u0414\u0430\u0439\u0434\u0436 {uuid.uuid4().hex[:8]}"
    item_id = str(uuid.uuid4())
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, _FMT_RU, "ru")
    _patch_pipeline_stages(mocker, digest_text, [item_id])

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result == digest_text, "generate_digest did not return the composed digest text"


@pytest.mark.asyncio
async def test_generate_digest_calls_writer_with_user_spec(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    tag = uuid.uuid4().hex[:6]
    prompt = f"\u0421\u0442\u0430\u0442\u044c\u0438 {tag}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding, _FMT_DETAIL, "ru")
    text_tag = uuid.uuid4().hex[:6]
    mocks = _patch_pipeline_stages(
        mocker,
        f"\u0422\u0435\u043a\u0441\u0442 {text_tag}",
        [str(uuid.uuid4())],
    )

    await digest_pipeline.generate_digest(session, subscription)

    write_kwargs = mocks["write"].await_args.kwargs
    assert f"## Topic\n{prompt}" in write_kwargs["user_spec"], (
        "generate_digest did not pass user_spec to writer"
    )


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources(
    mocker,
) -> None:
    session = _make_session_with_sources([])
    tag = uuid.uuid4().hex[:6]
    prompt = f"\u041b\u0435\u043a\u0446\u0438\u0438 {tag}"
    subscription = _make_subscription(prompt, [0.0] * 1536, _FMT_RU, "ru")

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None without fixed sources"


@pytest.mark.asyncio
async def test_generate_digest_returns_none_when_no_candidates(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    session = _make_session_with_sources([source_id])
    tag = uuid.uuid4().hex[:6]
    subscription = _make_subscription(
        f"\u041d\u043e\u0432\u043e\u0441\u0442\u0438 {tag}", embedding, _FMT_RU, "en"
    )
    mocker.patch(
        f"{_PIPELINE}.fetch_candidate_items",
        new=AsyncMock(return_value=[]),
    )

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result is None, "generate_digest did not return None when no candidates exist"


@pytest.mark.asyncio
async def test_generate_digest_computes_embedding_when_missing(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    tag = uuid.uuid4().hex[:6]
    prompt = f"\u0424\u0438\u0437\u0438\u043a\u0430 {tag}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, None, _FMT_RU, "ru")
    computed = [random.random() for _ in range(1536)]
    embed_mock = mocker.patch(
        f"{_PIPELINE}.embed_text",
        new=AsyncMock(return_value=computed),
    )
    text_tag = uuid.uuid4().hex[:6]
    _patch_pipeline_stages(
        mocker,
        f"\u0422\u0435\u043a\u0441\u0442 {text_tag}",
        [str(uuid.uuid4())],
    )

    await digest_pipeline.generate_digest(session, subscription)

    assert embed_mock.await_count == 1, "generate_digest did not call embed_text when missing"


@pytest.mark.asyncio
async def test_generate_digest_stores_computed_embedding(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    tag = uuid.uuid4().hex[:6]
    prompt = f"\u041a\u0432\u0430\u043d\u0442 {tag}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, None, _FMT_RU, "ru")
    computed = [random.random() for _ in range(1536)]
    mocker.patch(
        f"{_PIPELINE}.embed_text",
        new=AsyncMock(return_value=computed),
    )
    text_tag = uuid.uuid4().hex[:6]
    _patch_pipeline_stages(
        mocker,
        f"\u0422\u0435\u043a\u0441\u0442 {text_tag}",
        [str(uuid.uuid4())],
    )

    await digest_pipeline.generate_digest(session, subscription)

    assert subscription.topic_embedding == computed, (
        "generate_digest did not store computed embedding"
    )


@pytest.mark.asyncio
async def test_generate_digest_raises_pipeline_error_on_writer_failure(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    session = _make_session_with_sources([source_id])
    embedding = [random.random() for _ in range(1536)]
    tag = uuid.uuid4().hex[:6]
    subscription = _make_subscription(
        f"\u041d\u043e\u0432\u043e\u0441\u0442\u0438 {tag}", embedding, _FMT_RU, "en"
    )
    fake_item = SimpleNamespace(
        id=uuid.uuid4(),
        source_id=source_id,
        headline="T",
        body="B",
        url="http://t.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    mocker.patch(
        f"{_PIPELINE}.fetch_candidate_items",
        new=AsyncMock(return_value=[fake_item]),
    )
    mocker.patch(f"{_PIPELINE}.build_items_text", return_value="items")
    mocker.patch(
        f"{_PIPELINE}.write_digest",
        new=AsyncMock(side_effect=RuntimeError("writer failed")),
    )

    with pytest.raises(DigestPipelineError):
        await digest_pipeline.generate_digest(session, subscription)


@pytest.mark.asyncio
async def test_generate_digest_revises_on_judge_feedback(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    session = _make_session_with_sources([source_id])
    tag = uuid.uuid4().hex[:6]
    subscription = _make_subscription(
        f"\u041d\u043e\u0432\u043e\u0441\u0442\u0438 {tag}", embedding, _FMT_RU, "en"
    )

    candidate_item_id = uuid.uuid4()
    fake_item = SimpleNamespace(
        id=candidate_item_id,
        source_id=source_id,
        headline="T",
        body="B",
        url="http://t.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    mocker.patch(
        f"{_PIPELINE}.fetch_candidate_items",
        new=AsyncMock(return_value=[fake_item]),
    )
    mocker.patch(f"{_PIPELINE}.build_items_text", return_value="items")
    mocker.patch(
        f"{_PIPELINE}.run_reflector",
        new=AsyncMock(
            return_value={
                "discovery_triggered": False,
                "observations": "ok",
            },
        ),
    )

    first_text = f"Draft {uuid.uuid4().hex[:6]}"
    revised_text = f"Revised {uuid.uuid4().hex[:6]}"
    cid = str(candidate_item_id)
    write_mock = mocker.patch(
        f"{_PIPELINE}.write_digest",
        new=AsyncMock(
            side_effect=[
                _mock_composition(first_text, [cid]),
                _mock_composition(revised_text, [cid]),
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

    assert write_mock.await_count == 2, "generate_digest did not revise after judge feedback"
    assert result == revised_text, "generate_digest did not return revised text"
