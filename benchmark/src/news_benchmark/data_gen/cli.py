"""
CLI for the offline data-generation pipeline.

Usage:
    uv run python -m news_benchmark.data_gen.cli s01
    uv run python -m news_benchmark.data_gen.cli s01 s02 s03

Generates (or refreshes) the fabric for each named scenario, runs every
audit, and writes the scenario-data artifacts to data/scenarios/<id>/.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from news_benchmark.data_gen.pipeline import generate_for_scenario


def _load_env() -> None:
    for candidate in (Path(".env"), Path("../backend/.env")):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(description="Generate scenario data for the benchmark.")
    parser.add_argument("scenarios", nargs="+", help="Scenario ids to generate (e.g. s01 s02).")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory where data/scenarios/<id>/ is written.",
    )
    parser.add_argument(
        "--datagen-model",
        default=os.environ.get("BENCHMARK_DATAGEN_MODEL", "openai/gpt-5.4-nano"),
        help="LiteLLM model string used for body + fluff generation.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("LITELLM_JUDGE_MODEL", "openai/gpt-5.4-nano"),
        help="LiteLLM model string used for blind label-consistency audit.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-based label-consistency audit (fast mode).",
    )
    args = parser.parse_args(argv)

    judge_model = None if args.no_judge else args.judge_model

    async def run() -> None:
        for sid in args.scenarios:
            print(f"==> generating {sid}", flush=True)
            result = await generate_for_scenario(
                sid,
                data_dir=Path(args.data_dir),
                datagen_model=args.datagen_model,
                judge_model=judge_model,
            )
            print(
                f"    items={len(result.scenario.timeline)} accepted={result.audit.is_accepted()}",
                flush=True,
            )
            print(f"    diversity -> {result.diversity_path}", flush=True)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
