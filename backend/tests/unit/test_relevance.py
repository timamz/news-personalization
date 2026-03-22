import logging
import random
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.services import relevance

logging.disable(logging.CRITICAL)


def test_sample_posts_returns_all_when_fewer_than_sample_size() -> None:
    tag = uuid.uuid4().hex[:4]
    posts = [f"а-{tag}", f"б-{tag}", f"в-{tag}"]
    assert relevance.sample_posts(posts, 10) == posts, (
        "sample_posts did not return all items when list is smaller than sample_size"
    )


def test_sample_posts_returns_all_when_exact_sample_size() -> None:
    posts = [f"пост-{uuid.uuid4().hex[:4]}" for _ in range(3)]
    assert relevance.sample_posts(posts, 3) == posts, (
        "sample_posts did not return all items when list equals sample_size"
    )


def test_sample_posts_returns_correct_count_when_evenly_spaced() -> None:
    posts = [f"пост-{i}" for i in range(20)]
    sampled = relevance.sample_posts(posts, 5)
    assert len(sampled) == 5, "sample_posts did not return correct number of sampled items"


def test_sample_posts_picks_evenly_spaced_indices() -> None:
    posts = [f"пост-{i}" for i in range(20)]
    sampled = relevance.sample_posts(posts, 5)
    assert sampled == ["пост-0", "пост-4", "пост-8", "пост-12", "пост-16"], (
        "sample_posts did not pick evenly spaced indices"
    )


def test_sample_posts_returns_empty_for_empty_input() -> None:
    assert relevance.sample_posts([], 10) == [], (
        "sample_posts did not return empty list for empty input"
    )


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    dim = random.randint(3, 10)
    a = [random.random() for _ in range(dim)]
    result = relevance.cosine_similarity(a, a)
    assert result == pytest.approx(1.0), (
        "cosine_similarity did not return 1.0 for identical vectors"
    )


def test_cosine_similarity_returns_zero_for_orthogonal_vectors() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    result = relevance.cosine_similarity(a, b)
    assert result == pytest.approx(0.0), (
        "cosine_similarity did not return 0.0 for orthogonal vectors"
    )


def test_cosine_similarity_returns_negative_one_for_opposite_vectors() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    result = relevance.cosine_similarity(a, b)
    assert result == pytest.approx(-1.0), (
        "cosine_similarity did not return -1.0 for opposite vectors"
    )


def test_cosine_similarity_returns_zero_for_zero_vector() -> None:
    result = relevance.cosine_similarity([0.0, 0.0], [1.0, 0.0])
    assert result == 0.0, "cosine_similarity did not return 0.0 for zero vector"


@pytest.mark.asyncio
async def test_score_source_relevance_computes_top_k_average(mocker) -> None:
    prompt_emb = [1.0, 0.0, 0.0]
    post_embs = [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=post_embs))

    score = await relevance.score_source_relevance(
        [f"текст-{uuid.uuid4().hex[:4]}" for _ in range(3)], prompt_emb, top_k=2
    )

    sim_0 = relevance.cosine_similarity(post_embs[0], prompt_emb)
    sim_1 = relevance.cosine_similarity(post_embs[1], prompt_emb)
    expected = (sim_0 + sim_1) / 2
    assert score == pytest.approx(expected, abs=1e-4), (
        "score_source_relevance did not compute correct top-k average"
    )


@pytest.mark.asyncio
async def test_score_source_relevance_returns_zero_for_empty_posts(mocker) -> None:
    score = await relevance.score_source_relevance([], [1.0, 0.0], top_k=3)
    assert score == 0.0, "score_source_relevance did not return 0.0 for empty posts"


@pytest.mark.asyncio
async def test_score_candidate_returns_positive_score(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[f"пост {i}" for i in range(20)]),
    )
    mocker.patch.object(
        relevance,
        "embed_texts",
        new=AsyncMock(return_value=[[0.9, 0.1]] * 10),
    )

    score, _samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score > 0, "score_candidate did not return a positive score"


@pytest.mark.asyncio
async def test_score_candidate_returns_samples(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[f"пост {i}" for i in range(20)]),
    )
    mocker.patch.object(
        relevance,
        "embed_texts",
        new=AsyncMock(return_value=[[0.9, 0.1]] * 10),
    )

    _score, samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert len(samples) == 10, "score_candidate did not return expected number of samples"


@pytest.mark.asyncio
async def test_score_candidate_returns_zero_on_fetch_failure(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(side_effect=RuntimeError("сетевая ошибка")),
    )

    score, _samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score == 0.0, "score_candidate did not return 0.0 on fetch failure"


@pytest.mark.asyncio
async def test_score_candidate_returns_empty_samples_on_fetch_failure(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(side_effect=RuntimeError("сетевая ошибка")),
    )

    _score, samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert samples == [], "score_candidate did not return empty samples on fetch failure"


@pytest.mark.asyncio
async def test_score_candidate_returns_zero_for_empty_posts(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[]),
    )

    score, _samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score == 0.0, "score_candidate did not return 0.0 for empty posts"


@pytest.mark.asyncio
async def test_score_candidate_returns_empty_samples_for_empty_posts(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[]),
    )

    _score, samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert samples == [], "score_candidate did not return empty samples for empty posts"
