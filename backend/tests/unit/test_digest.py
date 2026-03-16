import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents import digest


def _mock_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_compose_digest_uses_subscription_language(mocker) -> None:
    create_completion = AsyncMock(return_value=_mock_completion("готово"))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
    )
    mocker.patch.object(digest, "_client", fake_client)

    item = SimpleNamespace(
        headline="Лекция",
        body="Анонс лекции",
        source="Telegram",
        url="https://example.com/1",
    )

    result = await digest._compose_digest([item], "кратко", "ru")

    assert result == "готово"
    system_prompt = create_completion.await_args.kwargs["messages"][0]["content"]
    user_prompt = create_completion.await_args.kwargs["messages"][1]["content"]
    assert "language 'ru'" in system_prompt
    assert "Return only the digest itself" in system_prompt
    assert "Use exactly 'Источник:'" in system_prompt
    assert "Ignore stale items and low-signal community chatter" in system_prompt
    assert "Link: https://example.com/1" in user_prompt
    assert "Published at:" in user_prompt
    assert "Source: Telegram" not in user_prompt


@pytest.mark.asyncio
async def test_compose_digest_requires_english_source_label_for_english_digest(mocker) -> None:
    create_completion = AsyncMock(return_value=_mock_completion("done"))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
    )
    mocker.patch.object(digest, "_client", fake_client)

    item = SimpleNamespace(
        headline="Paper roundup",
        body="Top picks from the week",
        source="Telegram",
        url="https://example.com/2",
    )

    await digest._compose_digest([item], "brief summary", "en")

    system_prompt = create_completion.await_args.kwargs["messages"][0]["content"]
    assert "Use exactly 'Source:'" in system_prompt
    assert "never switch to a different language" in system_prompt


@pytest.mark.asyncio
async def test_generate_digest_passes_language_to_composer(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id, "https://t.me/s/lectures")])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription_id = uuid.uuid4()
    subscription = SimpleNamespace(
        id=subscription_id,
        raw_prompt="AI lectures every morning",
        canonical_prompt="AI lectures every morning",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="AI lectures every morning",
        format_instructions="brief summary",
        digest_language="ru",
    )
    item = SimpleNamespace(id=uuid.uuid4())

    mocker.patch.object(digest, "embed_text", new=AsyncMock(return_value=[0.0] * 1536))
    find_similar_news = AsyncMock(return_value=[item])
    mocker.patch.object(digest, "find_similar_news", new=find_similar_news)
    compose_digest = AsyncMock(return_value="digest")
    mark_as_sent = AsyncMock()
    mocker.patch.object(digest, "_compose_digest", new=compose_digest)
    mocker.patch.object(digest, "_mark_as_sent", new=mark_as_sent)

    result = await digest.generate_digest(session, subscription)

    assert result == "digest"
    digest.embed_text.assert_not_awaited()
    find_similar_news.assert_awaited_once_with(
        session,
        [0.0] * 1536,
        exclude_ids=set(),
        allowed_feed_ids={source_feed_id},
        published_after=mocker.ANY,
        limit=15,
    )
    compose_digest.assert_awaited_once_with([item], "brief summary", "ru")
    mark_as_sent.assert_awaited_once_with(session, subscription_id, [item.id])


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    empty_sources_result = SimpleNamespace(all=lambda: [])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, empty_sources_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI lectures every morning",
        canonical_prompt="AI lectures every morning",
        prompt_summary="AI lectures every morning",
        format_instructions="brief summary",
        digest_language="ru",
    )

    embed_text = AsyncMock(return_value=[0.0] * 1536)
    find_similar_news = AsyncMock()
    mocker.patch.object(digest, "embed_text", new=embed_text)
    mocker.patch.object(digest, "find_similar_news", new=find_similar_news)

    result = await digest.generate_digest(session, subscription)

    assert result is None
    embed_text.assert_not_awaited()
    find_similar_news.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_digest_falls_back_to_raw_prompt_embedding_when_missing(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id, "https://t.me/s/lectures")])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI lectures every morning",
        canonical_prompt="AI lectures every morning",
        canonical_prompt_embedding=None,
        prompt_summary="AI lectures every morning",
        format_instructions="brief summary",
        digest_language="ru",
    )
    item = SimpleNamespace(id=uuid.uuid4())

    embed_text = AsyncMock(return_value=[0.0] * 1536)
    find_similar_news = AsyncMock(return_value=[item])
    mocker.patch.object(digest, "embed_text", new=embed_text)
    mocker.patch.object(digest, "find_similar_news", new=find_similar_news)
    mocker.patch.object(digest, "_compose_digest", new=AsyncMock(return_value="digest"))
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    result = await digest.generate_digest(session, subscription)

    assert result == "digest"
    embed_text.assert_awaited_once_with("AI lectures every morning")
    assert subscription.canonical_prompt_embedding == [0.0] * 1536


@pytest.mark.asyncio
async def test_generate_digest_uses_last_sent_at_as_cutoff(mocker) -> None:
    last_sent_at = datetime(2026, 3, 11, 9, 30, tzinfo=UTC)
    sent_news_id = uuid.uuid4()
    sent_result = SimpleNamespace(all=lambda: [(sent_news_id, last_sent_at)])
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(all=lambda: [(source_feed_id, "https://t.me/s/research")])
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="Latest ML papers",
        canonical_prompt="Latest ML papers",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="Latest ML papers",
        format_instructions="brief summary",
        digest_language="en",
    )
    item = SimpleNamespace(id=uuid.uuid4())

    find_similar_news = AsyncMock(return_value=[item])
    mocker.patch.object(digest, "find_similar_news", new=find_similar_news)
    mocker.patch.object(digest, "_compose_digest", new=AsyncMock(return_value="digest"))
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert find_similar_news.await_args.kwargs["published_after"] == last_sent_at


@pytest.mark.asyncio
async def test_generate_digest_excludes_reddit_sources_for_research_prompt(mocker) -> None:
    sent_result = SimpleNamespace(all=lambda: [])
    reddit_feed_id = uuid.uuid4()
    arxiv_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(
        all=lambda: [
            (reddit_feed_id, "https://www.reddit.com/r/machinelearning/new/"),
            (arxiv_feed_id, "https://export.arxiv.org/rss/cs.LG"),
        ]
    )
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="Хочу получать сводку по самым актуальным научным статьям в ML / AI",
        canonical_prompt="Хочу получать сводку по самым актуальным научным статьям в ML / AI",
        canonical_prompt_embedding=[0.0] * 1536,
        prompt_summary="Актуальные научные статьи в ML / AI",
        format_instructions="brief summary",
        digest_language="ru",
    )
    item = SimpleNamespace(id=uuid.uuid4())

    find_similar_news = AsyncMock(return_value=[item])
    mocker.patch.object(digest, "find_similar_news", new=find_similar_news)
    mocker.patch.object(digest, "_compose_digest", new=AsyncMock(return_value="digest"))
    mocker.patch.object(digest, "_mark_as_sent", new=AsyncMock())

    await digest.generate_digest(session, subscription)

    assert find_similar_news.await_args.kwargs["allowed_feed_ids"] == {arxiv_feed_id}
