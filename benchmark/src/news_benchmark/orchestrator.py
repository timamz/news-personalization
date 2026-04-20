"""
Matrix orchestrator: iterates scenarios x models, drives one run per combo.

Holds the DB/Redis/fakes lifecycle, invokes the scheduler, lets the
simulator drive the Conversational Agent, then triggers digest/event
ticks according to each scenario's content timeline.

This module is the integration surface with news_service. It is the
thinnest layer that can execute a scenario end-to-end. Individual
entry-point hooks (generate_digest, _poll_all_feeds,
_deliver_event_notifications_batch, run_event_verifier) are called
directly — Celery is bypassed entirely.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from news_benchmark.clock import CLOCK
from news_benchmark.config import BenchmarkConfig
from news_benchmark.cost_ledger import LEDGER
from news_benchmark.db import create_bench_db, drop_bench_db, run_alembic_upgrade
from news_benchmark.fakes.world import World
from news_benchmark.redis_ns import NamespacedRedis
from news_benchmark.report.json_writer import write_scenario_json
from news_benchmark.report.markdown import write_summary
from news_benchmark.scenarios.base import load_scenario


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
    any_failed = False

    for scenario_id in scenario_ids:
        scenario = load_scenario(data_dir / "scenarios", scenario_id)
        CLOCK.advance_to(datetime.fromisoformat(scenario.start_date_iso))

        for model_label in model_labels:
            model_env = _resolve_model_env(model_label, cfg)
            with _env_overrides(model_env):
                record, passed = await _run_one(
                    cfg=cfg,
                    run_id=run_id,
                    scenario_id=scenario_id,
                    model_column=model_label,
                    scenario=scenario,
                    out_dir=out_dir,
                )
            write_scenario_json(
                out_dir / "scenarios" / f"{scenario_id}__{model_label}.json",
                record,
            )
            if not passed:
                any_failed = True

    write_summary(out_dir)
    if any_failed and keep_db_on_failure:
        print("Leaving throwaway DBs intact for post-mortem.")


def _resolve_model_env(label: str, cfg: BenchmarkConfig) -> dict[str, str]:
    if label == "default":
        return {}
    return {"LITELLM_MODEL": label}


class _env_overrides:
    def __init__(self, overrides: dict[str, str]) -> None:
        self._overrides = overrides
        self._originals: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self._overrides.items():
            self._originals[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._originals.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _run_one(
    *,
    cfg: BenchmarkConfig,
    run_id: str,
    scenario_id: str,
    model_column: str,
    scenario,
    out_dir: Path,
) -> tuple[dict, bool]:
    """Run a single (scenario, model) combo. Returns (record, passed)."""
    combo_id = f"{run_id}_{scenario_id}_{model_column}".replace("/", "_")
    bench_url = await create_bench_db(cfg, combo_id)
    run_alembic_upgrade(bench_url)

    LEDGER.clear()
    LEDGER.set_context(run_id=run_id, scenario_id=scenario_id, model_column=model_column)

    redis = NamespacedRedis.from_url(cfg.benchmark_redis_url, prefix=f"bench_{combo_id}:")
    world = World()
    world.load_scenario(scenario.to_items(), scenario.to_search_corpus())

    record = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "model_column": model_column,
        "action_correctness": {"overall_pass": False, "outcomes": []},
        "classification": {"per_sub": {}, "dedup_failures_per_sub": {}},
        "judge_rubrics": {"digest": [], "event": [], "conversation": []},
        "cost": {
            "total_usd": LEDGER.total_usd(),
            "by_agent": LEDGER.by_agent(),
        },
        "captured_webhooks": [],
        "conversation_transcript": [],
        "notes": [
            "Harness-to-news_service wiring is scaffolded but the full "
            "orchestration (scheduler ticks + simulator driver + real agent hooks) "
            "is intended to run end-to-end against devbox; see the methodology "
            "report for the current integration state."
        ],
    }

    try:
        world.install()
        # Full orchestration (scheduler loop, simulator driver, agent hooks,
        # classification scoring, rubric judging) is wired in follow-up
        # work. All supporting modules are present; this function will be
        # extended to drive them in sequence.
    finally:
        world.uninstall()
        await redis.flush_prefix()
        await redis.close()

    passed = record["action_correctness"]["overall_pass"]
    if not (cfg.keep_db_on_failure and not passed):
        await drop_bench_db(cfg, combo_id)
    return record, passed
