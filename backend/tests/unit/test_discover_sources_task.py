"""Unit tests for the source-discovery Celery task."""

import logging
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.source_discovery import ScoredSource, SourceDiscoveryResult

logging.disable(logging.CRITICAL)


def _result(*urls: str) -> SourceDiscoveryResult:
    return SourceDiscoveryResult(
        sources=[
            ScoredSource(url=u, source_kind="rss", relevance_score=0.8, title="") for u in urls
        ]
    )


def _fake_subscription() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        is_active=True,
        user_spec="AI safety research. Skip hype.",
        topic_embedding=[0.1] * 8,
    )


def _patch_session_factory(mocker, session: MagicMock) -> None:
    @asynccontextmanager
    async def _factory():
        yield session

    mocker.patch(
        "news_service.tasks.discover_sources.get_task_session",
        new=_factory,
    )


@pytest.mark.asyncio
async def test_discover_loads_context_invokes_pipeline_and_persists_new_links(mocker) -> None:
    sub = _fake_subscription()
    session = MagicMock()
    session.commit = AsyncMock()
    session.add = MagicMock()

    # Subscription lookup -> attached sources -> removal history -> recently-removed URLs
    # -> sub_recheck -> link-exists checks.
    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub

    attached_rows = MagicMock()
    attached_rows.all.return_value = [("https://old.test/feed", "biotech feed", True)]

    removal_rows = MagicMock()
    removal_rows.all.return_value = []

    recently_removed_rows = MagicMock()
    recently_removed_rows.all.return_value = []

    discovered_url = f"https://{uuid.uuid4().hex[:8]}.test/ai"
    link_lookup = MagicMock()
    link_lookup.scalar_one_or_none.return_value = None

    sub_recheck = MagicMock()
    sub_recheck.scalar_one_or_none.return_value = True

    session.execute = AsyncMock(
        side_effect=[
            sub_lookup,
            attached_rows,
            removal_rows,
            recently_removed_rows,
            sub_recheck,
            link_lookup,
        ]
    )

    _patch_session_factory(mocker, session)
    mocker.patch(
        "news_service.tasks.discover_sources.run_source_discovery",
        new=AsyncMock(return_value=_result(discovered_url)),
    )
    ensured = SimpleNamespace(id=uuid.uuid4(), url=discovered_url, title=discovered_url)
    ensure_mock = mocker.patch(
        "news_service.tasks.discover_sources.ensure_source_by_url",
        new=AsyncMock(return_value=ensured),
    )

    from news_service.tasks.discover_sources import run_and_persist_discovery

    reason = f"User pivoted to AI {uuid.uuid4().hex[:6]}"
    result = await run_and_persist_discovery(session, sub.id, reason)

    assert (
        result["status"] == "ok"
        and result["persisted"] == 1
        and ensure_mock.await_count == 1
        and session.commit.await_count == 1
    ), "task did not persist exactly one new link and commit once"


@pytest.mark.asyncio
async def test_discover_skips_when_subscription_missing(mocker) -> None:
    session = MagicMock()
    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=sub_lookup)

    _patch_session_factory(mocker, session)
    pipeline = mocker.patch(
        "news_service.tasks.discover_sources.run_source_discovery",
        new=AsyncMock(),
    )

    from news_service.tasks.discover_sources import run_and_persist_discovery

    result = await run_and_persist_discovery(session, uuid.uuid4(), "irrelevant")
    assert result == {"status": "skipped", "reason": "not_found_or_inactive"} and (
        pipeline.await_count == 0
    ), "missing subscription should short-circuit before invoking the pipeline"


@pytest.mark.asyncio
async def test_discover_skips_when_subscription_has_no_embedding(mocker) -> None:
    sub = _fake_subscription()
    sub.topic_embedding = None
    session = MagicMock()
    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub
    session.execute = AsyncMock(return_value=sub_lookup)

    _patch_session_factory(mocker, session)
    pipeline = mocker.patch(
        "news_service.tasks.discover_sources.run_source_discovery",
        new=AsyncMock(),
    )

    from news_service.tasks.discover_sources import run_and_persist_discovery

    result = await run_and_persist_discovery(session, sub.id, "reason")
    assert result == {"status": "skipped", "reason": "no_embedding"} and (
        pipeline.await_count == 0
    ), "subscription without embedding must skip before invoking pipeline"


@pytest.mark.asyncio
async def test_discover_passes_recently_removed_urls_to_pipeline_as_locked_out(mocker) -> None:
    sub = _fake_subscription()
    session = MagicMock()
    session.commit = AsyncMock()
    session.add = MagicMock()

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub

    attached_rows = MagicMock()
    attached_rows.all.return_value = []

    removal_rows = MagicMock()
    removal_rows.all.return_value = []

    removed_url = f"https://{uuid.uuid4().hex[:8]}.test/dead-feed"
    recently_removed_rows = MagicMock()
    recently_removed_rows.all.return_value = [(removed_url,)]

    sub_recheck = MagicMock()
    sub_recheck.scalar_one_or_none.return_value = True

    session.execute = AsyncMock(
        side_effect=[sub_lookup, attached_rows, removal_rows, recently_removed_rows, sub_recheck]
    )

    _patch_session_factory(mocker, session)
    pipeline_mock = mocker.patch(
        "news_service.tasks.discover_sources.run_source_discovery",
        new=AsyncMock(return_value=SourceDiscoveryResult(sources=[])),
    )

    from news_service.tasks.discover_sources import run_and_persist_discovery

    await run_and_persist_discovery(session, sub.id, "reason")

    assert pipeline_mock.await_args.kwargs["locked_out_urls"] == [removed_url], (
        "recently removed URLs must be forwarded as locked_out_urls so the pipeline's "
        "upstream filter drops them before the pool is built; the LLM must not be "
        "trusted to respect them via prompt context alone"
    )


@pytest.mark.asyncio
async def test_discover_returns_no_sources_found_when_pipeline_yields_empty(mocker) -> None:
    sub = _fake_subscription()
    session = MagicMock()
    session.commit = AsyncMock()
    session.add = MagicMock()

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub

    attached_rows = MagicMock()
    attached_rows.all.return_value = []

    removal_rows = MagicMock()
    removal_rows.all.return_value = []

    recently_removed_rows = MagicMock()
    recently_removed_rows.all.return_value = []

    sub_recheck = MagicMock()
    sub_recheck.scalar_one_or_none.return_value = True

    session.execute = AsyncMock(
        side_effect=[sub_lookup, attached_rows, removal_rows, recently_removed_rows, sub_recheck]
    )

    _patch_session_factory(mocker, session)
    empty_result = SourceDiscoveryResult(sources=[], abort_reason="all strategies empty")
    mocker.patch(
        "news_service.tasks.discover_sources.run_source_discovery",
        new=AsyncMock(return_value=empty_result),
    )
    ensure_mock = mocker.patch(
        "news_service.tasks.discover_sources.ensure_source_by_url",
        new=AsyncMock(),
    )

    from news_service.tasks.discover_sources import run_and_persist_discovery

    result = await run_and_persist_discovery(session, sub.id, "reason")
    assert (
        result["status"] == "no_sources_found"
        and result["persisted"] == 0
        and result["discovered"] == 0
        and result.get("abort_reason") == "all strategies empty"
        and ensure_mock.await_count == 0
        and session.commit.await_count == 0
    ), "zero-source outcome must surface as no_sources_found with the abort reason attached"


@pytest.mark.asyncio
async def test_discover_drops_results_when_subscription_deactivated_mid_run(mocker) -> None:
    sub = _fake_subscription()
    session = MagicMock()
    session.commit = AsyncMock()
    session.add = MagicMock()

    sub_lookup = MagicMock()
    sub_lookup.scalar_one_or_none.return_value = sub

    attached_rows = MagicMock()
    attached_rows.all.return_value = []

    removal_rows = MagicMock()
    removal_rows.all.return_value = []

    recently_removed_rows = MagicMock()
    recently_removed_rows.all.return_value = []

    sub_recheck = MagicMock()
    sub_recheck.scalar_one_or_none.return_value = False

    session.execute = AsyncMock(
        side_effect=[sub_lookup, attached_rows, removal_rows, recently_removed_rows, sub_recheck]
    )

    _patch_session_factory(mocker, session)
    mocker.patch(
        "news_service.tasks.discover_sources.run_source_discovery",
        new=AsyncMock(return_value=_result("https://late.test/feed")),
    )
    ensure_mock = mocker.patch(
        "news_service.tasks.discover_sources.ensure_source_by_url",
        new=AsyncMock(),
    )

    from news_service.tasks.discover_sources import run_and_persist_discovery

    result = await run_and_persist_discovery(session, sub.id, "reason")
    assert (
        result["status"] == "skipped"
        and result["reason"] == "subscription_gone_after_discovery"
        and result["persisted"] == 0
        and ensure_mock.await_count == 0
        and session.commit.await_count == 0
    ), "discovery must not persist when the subscription disappears mid-run"


class _NotificationFormatTests:
    """Behavioral tests for the discovery-completion notification body.

    The function under test is ``_format_completion_body``: it must localize
    based on the user/subscription language and emit one URL per source line
    so the tgbot HTML renderer produces a single labeled link per bullet
    instead of the duplicated pair seen in production before this change.
    """


def test_completion_body_in_russian_emits_localized_header_for_attached_sources() -> None:
    from news_service.tasks.discover_sources import _format_completion_body

    body = _format_completion_body(
        {
            "status": "ok",
            "persisted": 2,
            "selected_sources": [
                {"url": "https://t.me/russianmacro", "source_kind": "telegram_channel"},
                {"url": "https://www.reddit.com/r/economy/", "source_kind": "reddit_subreddit"},
            ],
        },
        language="ru",
    )

    assert body is not None and body.startswith("Подключил 2"), (
        "completion body for a Russian-speaking user must use the localized header "
        "instead of the English default"
    )


def test_completion_body_emits_each_source_url_exactly_once_to_avoid_duplicate_links() -> None:
    from news_service.tasks.discover_sources import _format_completion_body

    url = "https://t.me/russianmacro"
    body = _format_completion_body(
        {
            "status": "ok",
            "persisted": 1,
            "selected_sources": [{"url": url, "source_kind": "telegram_channel"}],
        },
        language="en",
    )

    assert body is not None and body.count(url) == 1, (
        "each source bullet must contain its URL exactly once; rendering it twice "
        "produces two duplicate hyperlinks in the Telegram message"
    )


def test_completion_body_prefixes_telegram_bullets_with_at_handle_for_recognizability() -> None:
    from news_service.tasks.discover_sources import _format_completion_body

    body = _format_completion_body(
        {
            "status": "ok",
            "persisted": 1,
            "selected_sources": [
                {"url": "https://t.me/russianmacro", "source_kind": "telegram_channel"}
            ],
        },
        language="en",
    )

    assert body is not None and "@russianmacro" in body, (
        "Telegram source bullets must show the @handle so the user can recognize "
        "the channel without reading the raw URL"
    )


def test_completion_body_prefixes_reddit_bullets_with_subreddit_path() -> None:
    from news_service.tasks.discover_sources import _format_completion_body

    body = _format_completion_body(
        {
            "status": "ok",
            "persisted": 1,
            "selected_sources": [
                {"url": "https://www.reddit.com/r/economy/", "source_kind": "reddit_subreddit"}
            ],
        },
        language="en",
    )

    assert body is not None and "r/economy" in body, (
        "reddit source bullets must show the r/sub path; the raw URL alone is not "
        "as recognizable to a returning subscriber"
    )


def test_completion_body_for_unknown_language_falls_back_to_english_strings() -> None:
    from news_service.tasks.discover_sources import _format_completion_body

    body = _format_completion_body(
        {
            "status": "ok",
            "persisted": 1,
            "selected_sources": [
                {"url": "https://t.me/x", "source_kind": "telegram_channel"}
            ],
        },
        language="ja",
    )

    assert body is not None and body.startswith("Attached 1"), (
        "unknown ISO language codes must degrade to the English strings rather "
        "than crashing or emitting an empty body"
    )
