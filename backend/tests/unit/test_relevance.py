from unittest.mock import AsyncMock

import pytest

from news_service.services import relevance


def test_sample_posts_fewer_than_sample_size():
    posts = ["a", "b", "c"]
    assert relevance.sample_posts(posts, 10) == ["a", "b", "c"]


def test_sample_posts_exact_sample_size():
    posts = ["a", "b", "c"]
    assert relevance.sample_posts(posts, 3) == ["a", "b", "c"]


def test_sample_posts_evenly_spaced():
    posts = [f"post-{i}" for i in range(20)]
    sampled = relevance.sample_posts(posts, 5)
    assert len(sampled) == 5
    # Should pick evenly spaced: indices 0, 4, 8, 12, 16
    assert sampled == ["post-0", "post-4", "post-8", "post-12", "post-16"]


def test_sample_posts_empty():
    assert relevance.sample_posts([], 10) == []


def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    assert relevance.cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert relevance.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert relevance.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    assert relevance.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_score_source_relevance_high_similarity(mocker):
    prompt_emb = [1.0, 0.0, 0.0]
    post_embs = [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=post_embs))

    score = await relevance.score_source_relevance(["a", "b", "c"], prompt_emb, top_k=2)

    # Top-2 are the first two embeddings (highest sim to [1,0,0])
    sim_0 = relevance.cosine_similarity(post_embs[0], prompt_emb)
    sim_1 = relevance.cosine_similarity(post_embs[1], prompt_emb)
    assert score == pytest.approx((sim_0 + sim_1) / 2, abs=1e-4)


@pytest.mark.asyncio
async def test_score_source_relevance_empty_posts(mocker):
    score = await relevance.score_source_relevance([], [1.0, 0.0], top_k=3)
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_candidate_returns_score_and_samples(mocker):
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[f"post {i}" for i in range(20)]),
    )
    mocker.patch.object(
        relevance,
        "embed_texts",
        new=AsyncMock(return_value=[[0.9, 0.1]] * 10),
    )

    score, samples = await relevance.score_candidate("https://example.com/feed", "rss", [1.0, 0.0])

    assert score > 0
    assert len(samples) == 10


@pytest.mark.asyncio
async def test_score_candidate_handles_fetch_failure(mocker):
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(side_effect=RuntimeError("network error")),
    )

    score, samples = await relevance.score_candidate("https://example.com/feed", "rss", [1.0, 0.0])

    assert score == 0.0
    assert samples == []


@pytest.mark.asyncio
async def test_score_candidate_handles_empty_posts(mocker):
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[]),
    )

    score, samples = await relevance.score_candidate("https://example.com/feed", "rss", [1.0, 0.0])

    assert score == 0.0
    assert samples == []
