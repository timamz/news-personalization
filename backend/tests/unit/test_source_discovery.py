"""Tests for the source discovery pipeline (ReAct orchestrator shape)."""

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.agents.source_discovery import ScoredSource

logging.disable(logging.CRITICAL)


def _scored(url: str, score: float = 0.8, **kw) -> ScoredSource:
    return ScoredSource(
        url=url, source_kind=kw.pop("source_kind", "rss"), relevance_score=score, **kw
    )


def _runner_that(calls):
    """Build a fake ADK runner that invokes ``calls(tools, message)`` once."""

    captured: dict[str, str] = {}

    async def fake(*, agent, message, user_id="system"):
        captured["message"] = message
        tools = {t.__name__: t for t in agent.tools}
        await calls(tools, message)
        return "Done"

    return AsyncMock(side_effect=fake), captured


@pytest.mark.asyncio
async def test_pipeline_threads_spec_attached_and_reason_into_input(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[]),
    )

    async def _calls(tools, _message):
        await tools["abort"]("nothing to do")

    runner, captured = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    spec = f"AI safety research {uuid.uuid4().hex[:6]}. Skip hype."
    attached_url = f"https://{uuid.uuid4().hex[:8]}.test/biotech"
    reason = f"User shifted focus {uuid.uuid4().hex[:6]} from biotech to AI"

    await run_source_discovery(
        session=AsyncMock(),
        topic_text="AI safety, alignment, interpretability",
        prompt_embedding=[0.1] * 10,
        user_spec=spec,
        attached_sources=[(attached_url, "rss", True)],
        reason=reason,
    )

    message = captured["message"]
    assert (
        spec in message
        and attached_url in message
        and "user-specified" in message
        and reason in message
    ), "discovery agent input did not contain user_spec, attached source, and reason"


@pytest.mark.asyncio
async def test_pipeline_returns_only_urls_the_orchestrator_submitted(mocker) -> None:
    a = f"https://{uuid.uuid4().hex[:8]}.test/a"
    b = f"https://{uuid.uuid4().hex[:8]}.test/b"
    c = f"https://{uuid.uuid4().hex[:8]}.test/c"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(a, 0.9), _scored(b, 0.6), _scored(c, 0.45)]),
    )

    async def _calls(tools, _message):
        await tools["spawn_finder"]("one strategy")
        await tools["submit_selection"](a)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    urls = [s.url for s in result.sources]
    assert urls == [a], (
        "pipeline must accept the orchestrator's selection verbatim with no "
        "backfill of unpicked pool entries"
    )


@pytest.mark.asyncio
async def test_pipeline_accepts_low_score_pick_without_post_hoc_filter(mocker) -> None:
    low = f"https://{uuid.uuid4().hex[:8]}.test/low"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(low, 0.1)]),
    )

    async def _calls(tools, _message):
        await tools["spawn_finder"]("one strategy")
        await tools["submit_selection"](low)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    urls = {s.url for s in result.sources}
    assert urls == {low}, (
        "pipeline must not rescore or drop the orchestrator's picks; low scores "
        "are informational and anything submitted is accepted"
    )


@pytest.mark.asyncio
async def test_pipeline_surfaces_abort_reason_when_orchestrator_aborts(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[]),
    )

    abort_text = f"all finders returned nothing {uuid.uuid4().hex[:4]}"

    async def _calls(tools, _message):
        await tools["spawn_finder"]("one strategy")
        await tools["abort"](abort_text)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text=f"Topic {uuid.uuid4().hex[:4]}",
        prompt_embedding=[0.1] * 10,
    )
    assert result.sources == [] and result.abort_reason == abort_text, (
        "orchestrator's abort reason must be propagated to the caller so the "
        "downstream task can surface it to the user"
    )


@pytest.mark.asyncio
async def test_spawn_finder_excludes_already_attached_urls_from_pool(mocker) -> None:
    attached = f"https://{uuid.uuid4().hex[:8]}.test/attached"
    new_url = f"https://{uuid.uuid4().hex[:8]}.test/new"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(attached, 0.9), _scored(new_url, 0.7)]),
    )

    async def _calls(tools, _message):
        await tools["spawn_finder"]("strategy")
        rejection = await tools["submit_selection"](attached)
        assert "not in the candidate pool" in rejection
        await tools["submit_selection"](new_url)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
        attached_sources=[(attached, "rss", True)],
    )
    urls = [s.url for s in result.sources]
    assert urls == [new_url], (
        "attached URLs must be filtered from the pool so they cannot be re-accepted"
    )


@pytest.mark.asyncio
async def test_submit_selection_rejects_urls_not_in_the_pool(mocker) -> None:
    real = f"https://{uuid.uuid4().hex[:8]}.test/real"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(real, 0.9)]),
    )
    rejection_messages: list[str] = []

    async def _calls(tools, _message):
        await tools["spawn_finder"]("s")
        rejection_messages.append(await tools["submit_selection"]("https://not-real.test/feed"))
        await tools["submit_selection"](real)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
    )
    assert any("not in the candidate pool" in m for m in rejection_messages) and [
        s.url for s in result.sources
    ] == [real], "invalid URLs must be rejected before the valid submission lands"


@pytest.mark.asyncio
async def test_spawn_finder_survives_a_crash_and_returns_error_text(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(side_effect=RuntimeError("network down")),
    )
    seen: list[str] = []

    async def _calls(tools, _message):
        seen.append(await tools["spawn_finder"]("strategy"))
        await tools["abort"]("all finders failed")

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
    )
    assert result.sources == [] and any("finder crashed" in s for s in seen), (
        "orchestrator should receive a crash notice, not an exception"
    )


@pytest.mark.asyncio
async def test_inspect_source_returns_content_preview_for_pooled_candidate(mocker) -> None:
    url = f"https://{uuid.uuid4().hex[:8]}.test/feed"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(url, 0.8)]),
    )
    from news_service.services.relevance import DatedPost

    posts = [DatedPost(text=f"post body {uuid.uuid4().hex[:4]}", published_at=None)]
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.fetch_source_posts",
        new=AsyncMock(return_value=posts),
    )

    captured: dict[str, str] = {}

    async def _calls(tools, _message):
        await tools["spawn_finder"]("s")
        captured["preview"] = await tools["inspect_source"](url)
        await tools["submit_selection"](url)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
    )
    assert "post body" in captured["preview"] and url in captured["preview"], (
        "inspect_source preview did not include the fetched content and URL"
    )


@pytest.mark.asyncio
async def test_inspect_source_refuses_urls_outside_the_pool(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[]),
    )

    captured: dict[str, str] = {}

    async def _calls(tools, _message):
        captured["resp"] = await tools["inspect_source"]("https://ghost.test/feed")
        await tools["abort"]("done")

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
    )
    assert "not in candidate pool" in captured["resp"], (
        "inspect_source must refuse to fetch URLs the finders did not return"
    )


@pytest.mark.asyncio
async def test_pipeline_emits_status_events_through_the_queue(mocker) -> None:
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[]),
    )

    async def _calls(tools, _message):
        await tools["spawn_finder"]("s")
        await tools["abort"]("ok")

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)
    queue: asyncio.Queue[dict] = asyncio.Queue()

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
        status_queue=queue,
    )

    phases: list[str] = []
    while not queue.empty():
        phases.append(queue.get_nowait()["phase"])
    assert "planning" in phases and "searching" in phases, (
        "pipeline did not emit the expected discovery_progress phases onto the queue"
    )


@pytest.mark.asyncio
async def test_submit_selection_truncates_picks_past_hard_max_new(mocker) -> None:
    urls = [f"https://{uuid.uuid4().hex[:8]}.test/{i}" for i in range(5)]
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(u, 0.9 - 0.1 * i) for i, u in enumerate(urls)]),
    )

    submission_response: dict[str, str] = {}

    async def _calls(tools, _message):
        await tools["spawn_finder"]("strategy")
        submission_response["text"] = await tools["submit_selection"](",".join(urls))

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
        soft_max_new=2,
        hard_max_new=3,
    )

    accepted = [s.url for s in result.sources]
    assert accepted == urls[:3], (
        "submit_selection must truncate at hard_max_new in submission order so an "
        "over-eager agent cannot bypass the absolute ceiling, even when the soft "
        "cap suggests fewer"
    )
    assert "2 extra URL(s) were dropped" in submission_response["text"], (
        "the submit_selection return text did not flag the truncation back to the agent"
    )


@pytest.mark.asyncio
async def test_submit_selection_allows_soft_cap_overshoot_within_hard_cap(mocker) -> None:
    urls = [f"https://{uuid.uuid4().hex[:8]}.test/{i}" for i in range(3)]
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(u, 0.9 - 0.1 * i) for i, u in enumerate(urls)]),
    )

    submission_response: dict[str, str] = {}

    async def _calls(tools, _message):
        await tools["spawn_finder"]("strategy")
        submission_response["text"] = await tools["submit_selection"](",".join(urls))

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
        soft_max_new=1,
        hard_max_new=5,
    )

    assert [s.url for s in result.sources] == urls, (
        "agent must be allowed to exceed the soft cap when staying under the hard "
        "cap so the soft cap remains a recommendation, not a forced ceiling"
    )
    assert "extra" not in submission_response["text"], (
        "no truncation should be reported when the submission is between soft and hard cap"
    )


@pytest.mark.asyncio
async def test_hard_max_new_zero_forces_orchestrator_to_select_nothing(mocker) -> None:
    a = f"https://{uuid.uuid4().hex[:8]}.test/a"
    mocker.patch(
        "news_service.agents.source_discovery.pipeline.run_finder",
        new=AsyncMock(return_value=[_scored(a, 0.9)]),
    )

    async def _calls(tools, _message):
        await tools["spawn_finder"]("strategy")
        await tools["submit_selection"](a)

    runner, _ = _runner_that(_calls)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    result = await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
        soft_max_new=0,
        hard_max_new=0,
    )

    assert result.sources == [], (
        "hard_max_new=0 must clamp every submission to zero accepted sources so a "
        "subscription already at the hard ceiling cannot grow further via the LLM"
    )


@pytest.mark.asyncio
async def test_prompt_surfaces_soft_cap_hard_cap_and_attached_count(mocker) -> None:
    captured_prompt: dict[str, str] = {}

    async def fake(*, agent, message, user_id="system"):
        captured_prompt["instruction"] = agent.instruction
        for tool in agent.tools:
            if tool.__name__ == "abort":
                await tool("nothing to do")
        return "Done"

    runner = AsyncMock(side_effect=fake)
    mocker.patch("news_service.agents.source_discovery.pipeline.run_finder", new=AsyncMock())
    mocker.patch("news_service.agents.source_discovery.pipeline.run_agent_text", new=runner)

    from news_service.agents.source_discovery.pipeline import run_source_discovery

    attached = [
        (f"https://{uuid.uuid4().hex[:8]}.test/x", "rss", True),
        (f"https://{uuid.uuid4().hex[:8]}.test/y", "rss", False),
    ]
    await run_source_discovery(
        session=AsyncMock(),
        topic_text="topic",
        prompt_embedding=[0.1] * 10,
        attached_sources=attached,
        soft_max_new=3,
        hard_max_new=8,
    )

    instruction = captured_prompt["instruction"]
    assert all(
        marker in instruction
        for marker in (
            "already has 2 attached",
            "add at most 3 new",
            "go up to 8 new",
        )
    ), (
        "prompt did not surface attached count, the soft target, and the hard ceiling "
        "together; the LLM must see all three to balance recommendation against limit"
    )
