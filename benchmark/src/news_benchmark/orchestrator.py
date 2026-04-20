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
            CLOCK.advance_to(datetime.fromisoformat(scenario.start_date_iso).replace(tzinfo=UTC))
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
    world.load_scenario(scenario.to_items(), scenario.to_search_corpus())

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
    """Force news_service.core.config.get_settings to re-read env."""
    try:
        from news_service.core import config as ns_config

        ns_config.get_settings.cache_clear()
    except Exception:
        pass


async def _drive_scenario(
    *,
    scenario: Scenario,
    world: World,
    record: dict[str, Any],
) -> None:
    """Run scripted conversational turns, then advance the scheduler."""
    from news_service.db.session import get_task_session
    from news_service.models.user import User

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
        async with get_task_session() as session:
            user = await session.get(User, user_id)
            if user is None:
                raise RuntimeError("bench user vanished between turns")
            try:
                import asyncio as _asyncio

                agent_text = await _asyncio.wait_for(
                    _drive_one_turn(
                        session=session,
                        user=user,
                        user_message=turn.message,
                        transcript=transcript,
                    ),
                    timeout=180.0,
                )
            except TimeoutError:
                agent_text = "[timeout after 180s]"
                _say("  turn TIMED OUT after 180s")
            await session.commit()
        _say(f"  agent: {agent_text[:120]}")
        transcript.append({"speaker": "user", "text": turn.message})
        transcript.append({"speaker": "agent", "text": agent_text})

    record["conversation_transcript"] = transcript

    _say("starting scheduler loop")
    await _run_scheduler_loop(scenario=scenario, world=world, user_id=user_id, record=record)
    _say("scheduler loop complete")


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
    """Stream one conversational turn and return the final agent text."""
    from news_service.agents.conversational.agent import run_conversation_turn_streaming

    from news_benchmark.tagging import agent_tag

    messages: list[dict[str, str]] = []
    for prior in transcript:
        role = "user" if prior["speaker"] == "user" else "assistant"
        messages.append({"role": role, "content": prior["text"]})
    messages.append({"role": "user", "content": user_message})

    final_text = ""
    async with agent_tag("conversational"):
        async for event in run_conversation_turn_streaming(
            messages,
            db_session=session,
            user=user,
            conversation_summary="",
            user_language=user.language,
        ):
            if event.get("event") == "done":
                final_text = (event.get("output") or {}).get("message", "")
            elif event.get("event") == "error":
                final_text = f"[agent error] {event.get('detail', '')}"
    return final_text


async def _run_scheduler_loop(
    *,
    scenario: Scenario,
    world: World,
    user_id: Any,
    record: dict[str, Any],
) -> None:
    """Fire poll ticks, digest cron checks, and verifier ticks until EOS."""
    from news_service.tasks.deliver_digest import _deliver_digest
    from news_service.tasks.poll_feeds import _poll_all_feeds
    from news_service.tasks.schedule_digests import _schedule_due_digests

    from news_benchmark.tagging import agent_tag

    start = datetime.fromisoformat(scenario.start_date_iso).replace(tzinfo=UTC)
    # Cap simulated days for smoke runs to keep scheduler loop tractable.
    # Real embedding calls on every poll tick × 1440 poll cycles at
    # 30-day simulation exceeds any reasonable wall-clock budget.
    smoke_days = int(os.environ.get("BENCHMARK_SMOKE_DAYS", "0") or 0)
    effective_days = smoke_days if smoke_days > 0 else scenario.simulated_days
    end = start + timedelta(days=effective_days)
    _say(f"scheduler will run {effective_days} simulated days")

    sched = VirtualScheduler()
    poll_step = timedelta(minutes=30)
    cron_step = timedelta(minutes=60)

    delivered_event_ids: list[str] = []

    poll_count = [0]
    cron_count = [0]

    async def poll_tick() -> None:
        poll_count[0] += 1
        if poll_count[0] <= 5 or poll_count[0] % 48 == 0:
            _say(f"poll tick #{poll_count[0]} at {CLOCK.now().isoformat()}")
        async with agent_tag("pipeline.poll"):
            out = await _poll_all_feeds()
        new_ids = out.get("_event_item_ids") if isinstance(out, dict) else None
        if new_ids:
            delivered_event_ids.extend(new_ids)
        next_at = CLOCK.now() + poll_step
        if next_at <= end:
            sched.schedule(next_at, poll_tick, label="poll")

    async def cron_tick() -> None:
        cron_count[0] += 1
        async with agent_tag("pipeline.schedule_digests"):
            out = await _schedule_due_digests(now=CLOCK.now())
        queued = out.get("queued", 0) if isinstance(out, dict) else 0
        if queued:
            _say(f"cron tick #{cron_count[0]}: {queued} digest(s) queued")
        due_subs = out.get("_queued_subscription_ids") if isinstance(out, dict) else None
        if due_subs:
            for sub_id in due_subs:
                async with agent_tag("pipeline.digest"):
                    try:
                        await _deliver_digest(sub_id)
                    except Exception:
                        logger.exception("deliver_digest raised for %s", sub_id)
        next_at = CLOCK.now() + cron_step
        if next_at <= end:
            sched.schedule(next_at, cron_tick, label="cron")

    first_poll = max(start, CLOCK.now()) + timedelta(minutes=5)
    first_cron = max(start, CLOCK.now()) + timedelta(minutes=10)
    sched.schedule(first_poll, poll_tick, label="poll")
    sched.schedule(first_cron, cron_tick, label="cron")

    # For smoke runs against devbox, cap real wall time at 10 minutes.
    # The virtual scheduler is not wall-clock bound but each tick calls
    # real LLM + embedding endpoints, so runaway rate-limiting would
    # otherwise burn the benchmark budget.
    import asyncio as _asyncio

    try:
        await _asyncio.wait_for(sched.run(until=end), timeout=600.0)
    except TimeoutError:
        record["notes"].append("scheduler loop halted after 600s wall-clock timeout (smoke cap)")
        _say("scheduler wall-clock timeout; partial run")
    record["notes"].append(
        f"scheduler ran from {start.isoformat()} to {end.isoformat()} "
        f"({effective_days} simulated days, "
        f"{poll_count[0]} poll ticks, {cron_count[0]} cron ticks)"
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

    delivered_headlines_per_sub: dict[str, list[str]] = {}
    for goal in scenario.goals:
        url = goal.expected_webhook_url or ""
        hits = world.delivery.for_url(url)
        delivered_headlines_per_sub[goal.goal_id] = [h.subject for h in hits]

    cls_report = score_classification(scenario, delivered_headlines_per_sub)
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
