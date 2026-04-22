"""
Matrix orchestrator: iterates scenarios x models, drives one run per combo.

End-to-end lifecycle per (scenario, model):
  1. Set DATABASE_URL + REDIS_URL env vars to point at the throwaway DB
     created on devbox for this combo. Must happen BEFORE importing
     any news_service.* module because those modules cache settings
     at import time.
  2. Import news_service.* (lazy).
  3. Install the World fakes by monkey-patching module-level refs
     (search_web, deliver, fetch_article_text, adapter registry).
  4. Create the User row with the scenario persona's language + webhook
     URL so tool calls produce the right `delivery_webhook_url`.
  5. Drive the Simulator against `run_conversation_turn_streaming` for
     every scripted turn (each scripted turn drains until the agent's
     "done" event).
  6. After all scripted turns, run the virtual scheduler loop for the
     remaining simulated days: poll ticks every 30 min, schedule_digests
     every minute-resolution cron check, event deliveries piggyback on
     polling.
  7. Evaluate action_correctness assertions against live DB + fake
     delivery log.
  8. Score classification + judge rubrics on captured webhooks.
  9. Write scenario JSON + transcript + append to summary matrix.
  10. Drop throwaway DB, flush Redis prefix.

This module intentionally contains the integration complexity; the
supporting components (clock, scheduler, fakes, judges) are generic.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from news_benchmark.clock import CLOCK
from news_benchmark.config import BenchmarkConfig
from news_benchmark.cost_ledger import LEDGER
from news_benchmark.db import create_bench_db, drop_bench_db, run_alembic_upgrade
from news_benchmark.fakes.world import World
from news_benchmark.redis_ns import NamespacedRedis
from news_benchmark.report.json_writer import write_scenario_json
from news_benchmark.report.markdown import write_summary
from news_benchmark.report.transcript import render_transcript
from news_benchmark.scenarios.base import Scenario, load_scenario
from news_benchmark.scheduler import VirtualScheduler

logger = logging.getLogger(__name__)


def _say(msg: str) -> None:
    """Unbuffered progress print to stdout for smoke-test visibility."""
    print(f"[bench] {msg}", flush=True)


async def run_matrix(
    *,
    run_id: str,
    out_dir: Path,
    scenario_ids: list[str],
    model_labels: list[str],
    seed: int,
    repeat: int,
    keep_db_on_failure: bool,
) -> None:
    cfg = BenchmarkConfig(
        scenarios=scenario_ids,
        repeat=repeat,
        seed=seed,
        out_dir=out_dir,
        keep_db_on_failure=keep_db_on_failure,
    )
    cfg.ensure_paths()
    data_dir = Path(__file__).resolve().parents[2] / "data"

    for scenario_id in scenario_ids:
        scenario = load_scenario(data_dir / "scenarios", scenario_id)
        for model_label in model_labels:
            CLOCK.reset_to(datetime.fromisoformat(scenario.start_date_iso).replace(tzinfo=UTC))
            record = await _run_one(
                cfg=cfg,
                run_id=run_id,
                scenario=scenario,
                scenario_id=scenario_id,
                model_column=model_label,
            )
            write_scenario_json(
                out_dir / "scenarios" / f"{scenario_id}__{model_label}.json",
                record,
            )
            if record.get("conversation_transcript"):
                render_transcript(
                    out_dir / "transcripts" / f"{scenario_id}__{model_label}.md",
                    scenario_id=scenario_id,
                    model_column=model_label,
                    turns=record["conversation_transcript"],
                )

    write_summary(out_dir)


async def _run_one(
    *,
    cfg: BenchmarkConfig,
    run_id: str,
    scenario: Scenario,
    scenario_id: str,
    model_column: str,
) -> dict[str, Any]:
    """Drive one (scenario, model) combo end-to-end."""
    combo_id = f"{run_id}_{scenario_id}_{model_column}".replace("/", "_").replace(":", "_")
    _say(f"=== {scenario_id} / {model_column} / combo={combo_id}")
    _say("creating throwaway DB on devbox")
    bench_url = await create_bench_db(cfg, combo_id)
    _say("installing schema")
    run_alembic_upgrade(bench_url)
    _say("schema ready")

    bench_redis_url = cfg.benchmark_redis_url
    prefix = f"bench_{combo_id}:"

    os.environ["DATABASE_URL"] = bench_url
    os.environ["REDIS_URL"] = bench_redis_url
    if model_column != "default":
        os.environ["LITELLM_MODEL"] = model_column

    LEDGER.clear()
    LEDGER.set_context(run_id=run_id, scenario_id=scenario_id, model_column=model_column)

    redis = NamespacedRedis.from_url(bench_redis_url, prefix=prefix)
    world = World()
    world.load_scenario(scenario.to_items(), scenario.to_search_corpus(), scenario=scenario)

    record: dict[str, Any] = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "model_column": model_column,
        "action_correctness": {"overall_pass": False, "outcomes": []},
        "classification": {"per_sub": {}, "dedup_failures_per_sub": {}},
        "judge_rubrics": {"digest": [], "event": [], "conversation": []},
        "cost": {"total_usd": 0.0, "by_agent": {}},
        "captured_webhooks": [],
        "conversation_transcript": [],
        "notes": [],
    }

    try:
        _reset_news_service_settings()
        world.install()
        _say("world fakes installed")
        try:
            await _drive_scenario(
                cfg=cfg,
                scenario=scenario,
                world=world,
                record=record,
            )
            await _evaluate(scenario=scenario, world=world, record=record)
        finally:
            world.uninstall()
            await redis.flush_prefix()
            await redis.close()
    except Exception as exc:
        logger.exception("Scenario run failed: %s", exc)
        record["notes"].append(f"run raised: {type(exc).__name__}: {exc}")
        _say(f"run raised: {type(exc).__name__}: {exc}")

    record["cost"] = {
        "total_usd": LEDGER.total_usd(),
        "by_agent": LEDGER.by_agent(),
        "rows": [r.to_dict() for r in LEDGER.rows()],
    }

    passed = record["action_correctness"].get("overall_pass", False)
    if not (cfg.keep_db_on_failure and not passed):
        await drop_bench_db(cfg, combo_id)

    return record


def _reset_news_service_settings() -> None:
    """Force news_service.core.config.get_settings to re-read env, and
    re-bind the module-level ``settings`` / ``engine`` references that
    several backend modules cache at import time.

    Without this, ``db/session.py`` keeps the first scenario's
    DATABASE_URL forever: subsequent scenarios open sessions against
    the already-dropped previous DB and raise ``InvalidCatalogNameError``.
    """
    try:
        from news_service.core import config as ns_config

        # ``get_settings`` is a plain function on this backend (no
        # ``functools.lru_cache`` wrapper), so ``cache_clear()`` doesn't
        # exist. Call it to materialise a fresh Settings that re-reads
        # ``os.environ`` (pydantic-settings is eager on construction).
        if hasattr(ns_config.get_settings, "cache_clear"):
            ns_config.get_settings.cache_clear()
        settings = ns_config.get_settings()

        from news_service.db import session as session_mod
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        session_mod.settings = settings  # type: ignore[attr-defined]
        session_mod.engine = create_async_engine(
            settings.database_url, echo=False, pool_size=5, max_overflow=10
        )
        session_mod.async_session_factory = async_sessionmaker(
            session_mod.engine, expire_on_commit=False
        )
        _say(f"settings refreshed -> database_url={settings.database_url[-40:]}")

        # Any other module that does ``settings = get_settings()`` at
        # import time: rebind so queries use the new DB URL. We list
        # them explicitly rather than walking sys.modules so the set
        # of mutated modules is auditable.
        for mod_path in (
            "news_service.core.llm",
            "news_service.services.relevance",
            "news_service.services.search",
            "news_service.services.article_fetch",
            "news_service.services.coverage",
            "news_service.agents.digest.pipeline",
            "news_service.agents.digest.writer",
            "news_service.agents.event.verifier",
            "news_service.agents.source_discovery.pipeline",
            "news_service.agents.source_discovery.finder",
            "news_service.tasks.poll_feeds",
            "news_service.tasks.poll_adapters",
            "news_service.tasks.reflect_events",
            "news_service.tasks.schedule_digests",
            "news_service.db.vector_store",
        ):
            import sys as _sys

            mod = _sys.modules.get(mod_path)
            if mod is not None and hasattr(mod, "settings"):
                mod.settings = settings
    except Exception:
        logger.exception("Failed to refresh news_service settings between scenarios")


async def _drive_scenario(
    *,
    cfg: BenchmarkConfig,
    scenario: Scenario,
    world: World,
    record: dict[str, Any],
) -> None:
    """Run scripted conversational turns, then a bounded persona follow-up,
    then advance the scheduler.

    The scripted turns anchor fake-clock advancement (each turn may live at
    a different ``fake_day``). After they're exhausted, if any goal is
    still unmet (e.g. the agent asked a clarifying question instead of
    closing the subscription), the persona LLM takes over until all goals
    are met, it emits ``<END>``, it stalls for three turns without goal
    progress, or ``simulator_max_turns`` is reached.
    """
    _say("creating bench user row")
    user_id = await _create_user_row(scenario)

    transcript: list[dict[str, Any]] = []
    for i, turn in enumerate(scenario.scripted_turns):
        target = datetime.fromisoformat(scenario.start_date_iso).replace(tzinfo=UTC) + timedelta(
            days=turn.fake_day, hours=9
        )
        if target > CLOCK.now():
            CLOCK.advance_to(target)

        _say(f"scripted turn {i + 1}/{len(scenario.scripted_turns)}: {turn.message[:80]}...")
        agent_text = await _run_agent_turn(
            user_id=user_id, user_message=turn.message, transcript=transcript
        )
        _say(f"  agent: {agent_text[:120]}")
        transcript.append({"speaker": "user", "text": turn.message})
        transcript.append({"speaker": "agent", "text": agent_text})

    await _run_persona_followup(
        cfg=cfg,
        scenario=scenario,
        user_id=user_id,
        transcript=transcript,
        record=record,
    )

    record["conversation_transcript"] = transcript

    _say("starting scheduler loop")
    await _run_scheduler_loop(scenario=scenario, world=world, user_id=user_id, record=record)
    _say("scheduler loop complete")


async def _run_agent_turn(
    *,
    user_id: Any,
    user_message: str,
    transcript: list[dict[str, Any]],
) -> str:
    """Run a single conversational agent turn inside its own DB session.

    No wall-clock timeout: some agents legitimately spend minutes on
    source discovery (Yandex queries, page fetches, embedding validation).
    Instead, ``_drive_one_turn`` streams every status / discovery_progress
    event out to stdout so a human watching the run can tell "still
    working" from "actually stuck."
    """
    from news_service.db.session import get_task_session
    from news_service.models.user import User

    async with get_task_session() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise RuntimeError("bench user vanished between turns")
        agent_text = await _drive_one_turn(
            session=session,
            user=user,
            user_message=user_message,
            transcript=transcript,
        )
        await session.commit()
    return agent_text


async def _check_goal_status(*, user_id: Any, scenario: Scenario) -> dict[str, bool]:
    """Return ``{goal_id: met}``. A goal is 'met' when a subscription row
    exists for this user that satisfies the same criteria the
    ``subscription_exists_matching`` assertion checks: expected webhook
    URL, delivery mode, schedule cron, digest language, and every
    keyword in ``expected_user_spec_keywords`` appearing in
    ``user_spec``. Mirroring the assertion keeps the follow-up loop
    honest -- if the agent only half-created the subscription, the
    persona will push it further instead of terminating prematurely."""
    from news_service.db.session import get_task_session
    from news_service.models.subscription import Subscription
    from sqlalchemy import select

    async with get_task_session() as session:
        subs = (
            (await session.execute(select(Subscription).where(Subscription.user_id == user_id)))
            .scalars()
            .all()
        )

    out: dict[str, bool] = {}
    for goal in scenario.goals:
        met = False
        for sub in subs:
            if goal.expected_webhook_url and sub.delivery_webhook_url != goal.expected_webhook_url:
                continue
            if sub.delivery_mode != goal.expected_delivery_mode:
                continue
            if (
                goal.expected_schedule_cron is not None
                and sub.schedule_cron != goal.expected_schedule_cron
            ):
                continue
            if goal.expected_digest_language is not None and (
                (sub.digest_language or "").lower() != goal.expected_digest_language.lower()
            ):
                continue
            spec_lower = (sub.user_spec or "").lower()
            if not all(kw.lower() in spec_lower for kw in goal.expected_user_spec_keywords):
                continue
            met = True
            break
        out[goal.goal_id] = met
    return out


async def _run_persona_followup(
    *,
    cfg: BenchmarkConfig,
    scenario: Scenario,
    user_id: Any,
    transcript: list[dict[str, Any]],
    record: dict[str, Any],
) -> None:
    """Drive persona-LLM follow-up turns until goals met / <END> / stall / budget."""
    from news_benchmark.simulator.driver import next_user_message

    status = await _check_goal_status(user_id=user_id, scenario=scenario)
    remaining = [g for g in scenario.goals if not status.get(g.goal_id, False)]
    if not remaining:
        record["notes"].append("persona follow-up skipped: all goals met after scripted turns")
        return

    turns_used = len(scenario.scripted_turns)
    budget = max(0, cfg.simulator_max_turns - turns_used)
    if budget == 0:
        record["notes"].append(
            "persona follow-up skipped: scripted turns already at budget "
            f"({cfg.simulator_max_turns})"
        )
        return

    _say(f"starting persona follow-up (budget={budget}, remaining_goals={len(remaining)})")
    stall = 0
    terminated = "turn_budget"
    for step in range(budget):
        try:
            user_msg = await next_user_message(
                persona=scenario.persona,
                remaining_goals=remaining,
                max_turns=cfg.simulator_max_turns,
                simulator_model=cfg.litellm_model,
                simulator_temperature=cfg.simulator_temperature,
                transcript=transcript,
            )
        except Exception as exc:
            record["notes"].append(f"persona LLM failed: {type(exc).__name__}: {exc}")
            _say(f"  persona LLM failed: {exc}")
            terminated = "persona_error"
            break

        stripped = user_msg.strip()
        if not stripped:
            terminated = "persona_empty_reply"
            break
        if stripped == "<END>" or stripped.endswith("<END>"):
            transcript.append({"speaker": "user", "text": user_msg})
            terminated = "simulator_end_sentinel"
            break

        _say(f"persona turn {step + 1}/{budget}: {stripped[:80]}...")
        agent_text = await _run_agent_turn(
            user_id=user_id, user_message=user_msg, transcript=transcript
        )
        _say(f"  agent: {agent_text[:120]}")
        transcript.append({"speaker": "user", "text": user_msg})
        transcript.append({"speaker": "agent", "text": agent_text})

        status = await _check_goal_status(user_id=user_id, scenario=scenario)
        new_remaining = [g for g in scenario.goals if not status.get(g.goal_id, False)]
        if len(new_remaining) < len(remaining):
            stall = 0
        else:
            stall += 1
        remaining = new_remaining
        if not remaining:
            terminated = "all_goals_met"
            break
        if stall >= 3:
            terminated = "stall"
            break

    record["notes"].append(
        f"persona follow-up terminated: {terminated} after "
        f"{len([t for t in transcript if t['speaker'] == 'user']) - turns_used} persona turn(s)"
    )


async def _create_user_row(scenario: Scenario) -> Any:
    """Create the bench User row + pre-register per-goal webhook URLs."""
    from uuid import uuid4

    from news_service.db.session import get_task_session
    from news_service.models.user import User

    user_id = uuid4()
    async with get_task_session() as session:
        user = User(
            id=user_id,
            api_key=f"bench_{user_id.hex[:12]}",
            timezone=scenario.persona.timezone,
            language=scenario.persona.language,
            delivery_webhook_url=_primary_webhook_url(scenario),
            has_onboarded=False,
        )
        session.add(user)
        await session.commit()
    return user_id


def _primary_webhook_url(scenario: Scenario) -> str:
    for goal in scenario.goals:
        if goal.expected_webhook_url:
            return goal.expected_webhook_url
    return "https://bench.invalid/default/digest"


async def _drive_one_turn(
    *,
    session,
    user,
    user_message: str,
    transcript: list[dict[str, Any]],
) -> str:
    """Stream one conversational turn and return the final agent text.

    Every intermediate event (status / discovery_progress / error) is
    printed to stdout so the operator can watch what the agent is doing
    in real time. A heartbeat task prints an "..elapsed Xs, silent Ys.."
    line every 30s while the turn is in flight so that silence becomes
    visible instead of looking like a stuck harness.
    """
    import asyncio as _asyncio
    import contextlib as _contextlib
    import time as _time

    from news_service.agents.conversational.agent import run_conversation_turn_streaming

    from news_benchmark.tagging import agent_tag

    messages: list[dict[str, str]] = []
    for prior in transcript:
        role = "user" if prior["speaker"] == "user" else "assistant"
        messages.append({"role": role, "content": prior["text"]})
    messages.append({"role": "user", "content": user_message})

    started = _time.monotonic()
    last_event_at = started
    final_text = ""
    done = _asyncio.Event()

    async def _heartbeat() -> None:
        try:
            while not done.is_set():
                await _asyncio.wait_for(done.wait(), timeout=30.0)
        except TimeoutError:
            while not done.is_set():
                silent = _time.monotonic() - last_event_at
                elapsed = _time.monotonic() - started
                _say(f"  .. agent still working: elapsed {elapsed:.0f}s, silent {silent:.0f}s")
                try:
                    await _asyncio.wait_for(done.wait(), timeout=30.0)
                except TimeoutError:
                    continue

    hb_task = _asyncio.create_task(_heartbeat())
    try:
        async with agent_tag("conversational"):
            async for event in run_conversation_turn_streaming(
                messages,
                db_session=session,
                user=user,
                conversation_summary="",
                user_language=user.language,
            ):
                last_event_at = _time.monotonic()
                kind = event.get("event")
                if kind == "status":
                    key = event.get("status_key", "?")
                    extras = {k: v for k, v in event.items() if k not in ("event", "status_key")}
                    suffix = f" {extras}" if extras else ""
                    _say(f"  status: {key}{suffix}")
                elif kind == "discovery_progress":
                    phase = event.get("phase", "?")
                    text = event.get("display_text") or ""
                    _say(f"  discovery[{phase}]: {text[:140]}")
                elif kind == "done":
                    final_text = (event.get("output") or {}).get("message", "")
                elif kind == "error":
                    final_text = f"[agent error] {event.get('detail', '')}"
                    _say(f"  agent error: {event.get('detail', '')[:200]}")
    finally:
        done.set()
        hb_task.cancel()
        with _contextlib.suppress(_asyncio.CancelledError, Exception):
            await hb_task
    return final_text


async def _run_scheduler_loop(
    *,
    scenario: Scenario,
    world: World,
    user_id: Any,
    record: dict[str, Any],
) -> None:
    """Fire poll ticks and digest cron checks until EOS.

    Both ticks simply invoke the production async task functions. The
    downstream dispatch (event-batch delivery queued by the poll, digest
    delivery queued by the cron, source discovery queued by a reflector)
    all goes through ``celery_app.send_task`` / ``task.delay``, which
    the World's CeleryShim routes back to inline ``asyncio.create_task``
    on this same loop. We ``await world.celery.drain()`` at the end so
    delivery side-effects finish before scoring.
    """
    from news_service.tasks.poll_feeds import _poll_all_feeds
    from news_service.tasks.reflect_events import _reflect_event_subscriptions
    from news_service.tasks.schedule_digests import _schedule_due_digests

    from news_benchmark.tagging import agent_tag

    start = datetime.fromisoformat(scenario.start_date_iso).replace(tzinfo=UTC)
    smoke_days = int(os.environ.get("BENCHMARK_SMOKE_DAYS", "0") or 0)
    effective_days = smoke_days if smoke_days > 0 else scenario.simulated_days
    end = start + timedelta(days=effective_days)
    _say(f"scheduler will run {effective_days} simulated days")

    sched = VirtualScheduler()
    poll_step = timedelta(minutes=30)
    cron_step = timedelta(minutes=60)
    verifier_step = timedelta(days=1)

    poll_count = [0]
    cron_count = [0]
    verifier_count = [0]

    # Share the CeleryShim's serializer lock across every tick so
    # poll_tick / cron_tick / verifier_tick / shim-dispatched tasks
    # never hold overlapping DB sessions. Without this the asyncpg
    # driver trips on "another operation is in progress" whenever a
    # scheduled poll collides with an in-flight digest delivery on the
    # same event loop.
    tick_lock = world.celery._serializer_lock

    async def poll_tick() -> None:
        poll_count[0] += 1
        if poll_count[0] <= 5 or poll_count[0] % 48 == 0:
            _say(f"poll tick #{poll_count[0]} at {CLOCK.now().isoformat()}")
        assert tick_lock is not None
        async with tick_lock, agent_tag("pipeline.poll"):
            await _poll_all_feeds()
        next_at = CLOCK.now() + poll_step
        if next_at <= end:
            sched.schedule(next_at, poll_tick, label="poll")

    async def cron_tick() -> None:
        cron_count[0] += 1
        assert tick_lock is not None
        async with tick_lock, agent_tag("pipeline.schedule_digests"):
            out = await _schedule_due_digests(now=CLOCK.now())
        queued = out.get("queued", 0) if isinstance(out, dict) else 0
        if queued:
            _say(f"cron tick #{cron_count[0]}: {queued} digest(s) queued")
        next_at = CLOCK.now() + cron_step
        if next_at <= end:
            sched.schedule(next_at, cron_tick, label="cron")

    async def verifier_tick() -> None:
        verifier_count[0] += 1
        _say(f"verifier tick #{verifier_count[0]} at {CLOCK.now().isoformat()}")
        assert tick_lock is not None
        async with tick_lock, agent_tag("pipeline.verifier"):
            await _reflect_event_subscriptions()
        next_at = CLOCK.now() + verifier_step
        if next_at <= end:
            sched.schedule(next_at, verifier_tick, label="verifier")

    first_poll = max(start, CLOCK.now()) + timedelta(minutes=5)
    first_cron = max(start, CLOCK.now()) + timedelta(minutes=10)
    first_verifier = max(start, CLOCK.now()) + timedelta(days=1)
    sched.schedule(first_poll, poll_tick, label="poll")
    sched.schedule(first_cron, cron_tick, label="cron")
    sched.schedule(first_verifier, verifier_tick, label="verifier")

    await sched.run(until=end)
    await world.celery.drain()
    record["notes"].append(
        f"scheduler ran from {start.isoformat()} to {end.isoformat()} "
        f"({effective_days} simulated days, "
        f"{poll_count[0]} poll ticks, {cron_count[0]} cron ticks, "
        f"{verifier_count[0]} verifier ticks)"
    )


async def _evaluate(
    *,
    scenario: Scenario,
    world: World,
    record: dict[str, Any],
) -> None:
    """Evaluate deterministic assertions + classification + rubrics."""
    from news_service.db.session import get_task_session

    from news_benchmark.judge.action_correctness import evaluate
    from news_benchmark.judge.classification_metrics import score_classification

    async with get_task_session() as session:
        report = await evaluate(
            scenario=scenario,
            session=session,
            delivery=world.delivery,
        )

    record["action_correctness"] = {
        "overall_pass": report.overall_pass(),
        "outcomes": [
            {"kind": o.kind, "passed": o.passed, "detail": o.detail} for o in report.outcomes
        ],
    }

    delivered_bodies_per_sub: dict[str, list[str]] = {}
    for goal in scenario.goals:
        url = goal.expected_webhook_url or ""
        hits = world.delivery.for_url(url)
        delivered_bodies_per_sub[goal.goal_id] = [
            f"{h.subject}\n{h.body}" if h.subject else h.body for h in hits
        ]

    cls_report = score_classification(scenario, delivered_bodies_per_sub)
    record["classification"] = cls_report.to_dict()

    record["captured_webhooks"] = [
        {
            "url": c.url,
            "subject": c.subject,
            "body": c.body[:2000],
            "fake_clock_iso": c.fake_clock.isoformat(),
        }
        for c in world.delivery.captured
    ]
