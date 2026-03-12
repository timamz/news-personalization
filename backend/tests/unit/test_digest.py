import uuid
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
    assert "Link: https://example.com/1" in user_prompt
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
    sent_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [source_feed_id]))
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription_id = uuid.uuid4()
    subscription = SimpleNamespace(
        id=subscription_id,
        raw_prompt="AI lectures every morning",
        raw_prompt_embedding=[0.0] * 1536,
        topics=["lectures"],
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
        limit=15,
    )
    compose_digest.assert_awaited_once_with([item], "brief summary", "ru")
    mark_as_sent.assert_awaited_once_with(session, subscription_id, [item.id])


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources(mocker) -> None:
    sent_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
    empty_sources_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, empty_sources_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI lectures every morning",
        topics=["lectures"],
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
    sent_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
    source_feed_id = uuid.uuid4()
    source_result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [source_feed_id]))
    session = SimpleNamespace(execute=AsyncMock(side_effect=[sent_result, source_result]))

    subscription = SimpleNamespace(
        id=uuid.uuid4(),
        raw_prompt="AI lectures every morning",
        raw_prompt_embedding=None,
        topics=["lectures"],
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
    assert subscription.raw_prompt_embedding == [0.0] * 1536
