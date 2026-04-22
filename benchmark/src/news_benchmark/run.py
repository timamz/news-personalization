"""
CLI entry point for the benchmark harness.

Boot order is load-bearing:
  1. parse args
  2. install FakeClock datetime patch (must precede any news_service import)
  3. install litellm wrappers (cost ledger)
  4. lazy-import news_service and wire up the World fakes
  5. create throwaway DB on devbox, run alembic upgrade head
  6. for each (scenario, model) combo: drive one run
  7. write per-scenario JSON + transcripts + summary.md
  8. drop DB unless --keep-db-on-failure and at least one scenario failed

Not all integration surfaces are exercised in a single run — individual
scenarios tell the scheduler which ticks to schedule (poll, digest cron,
verifier) and the simulator when to push scripted turns.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    for candidate in (Path(".env"), Path("../backend/.env")):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _configure_logging() -> None:
    """Route backend INFO+ logs to stdout so digest / poll / delivery paths are visible.

    The production backend calls ``setup_logging()`` at FastAPI startup;
    the benchmark never boots FastAPI, so the root logger has no handler
    and ``logger.info(...)`` calls silently vanish. That hides critical
    signal like ``"No candidates"`` / ``"No fixed sources"`` during
    digest delivery. Install a plain StreamHandler at INFO level.
    """
    import logging
    import sys

    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    _load_env()
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run the LLM-as-judge benchmark.")
    parser.add_argument("--scenarios", default="s01", help="Comma-separated scenario ids.")
    parser.add_argument("--models", default="default", help="'default' or comma-separated labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--keep-db-on-failure", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every LLM request/response (trimmed) via news_benchmark.llm_trace.",
    )
    args = parser.parse_args(argv)
    if args.verbose:
        import os

        os.environ["BENCH_VERBOSE_LLM"] = "1"

    from news_benchmark.clock import install_clock_patch

    install_clock_patch(start=datetime.fromisoformat("2026-03-01T00:00:00+00:00"))

    from news_benchmark.cost_ledger import install_litellm_wrappers

    install_litellm_wrappers()

    from news_benchmark.orchestrator import run_matrix

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(args.out_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"

    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    model_labels = [m.strip() for m in args.models.split(",") if m.strip()]

    asyncio.run(
        run_matrix(
            run_id=run_id,
            out_dir=out_dir,
            scenario_ids=scenario_ids,
            model_labels=model_labels,
            seed=args.seed,
            repeat=args.repeat,
            keep_db_on_failure=args.keep_db_on_failure,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
