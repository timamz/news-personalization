"""
Data-generation pipeline: skeleton -> fabric -> audits -> diversity report.

Entry point is generate_for_scenario(scenario_id). It loads (or imports)
the hand-authored skeleton module at data_gen/skeletons/<id>.py, fills
bodies and search fluff via LLM calls, runs every audit in audits.py,
writes the fabric + audit report + diversity.md to the data directory,
and raises if any hard-gate audit fails.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from news_benchmark.data_gen.audits import (
    AuditResult,
    run_audits,
    write_diversity_report,
)
from news_benchmark.data_gen.body_generator import BodyGenerator
from news_benchmark.data_gen.search_fluff import SearchFluffGenerator
from news_benchmark.scenarios.base import (
    Scenario,
    load_scenario,
    save_fabric,
    save_skeleton,
)


@dataclass
class GenerationResult:
    scenario: Scenario
    audit: AuditResult
    skeleton_path: Path
    fabric_path: Path
    diversity_path: Path


async def generate_for_scenario(
    scenario_id: str,
    *,
    data_dir: Path,
    datagen_model: str,
    judge_model: str | None,
) -> GenerationResult:
    """Generate or refresh the fabric for a scenario and run all audits."""
    skeleton_mod = importlib.import_module(f"news_benchmark.data_gen.skeletons.{scenario_id}")
    scenario: Scenario = skeleton_mod.build()
    scenarios_dir = data_dir / "scenarios"
    save_skeleton(scenario, scenarios_dir)

    fabric_cache = scenarios_dir / scenario.scenario_id / "bodies_cache.json"
    fluff_cache = scenarios_dir / scenario.scenario_id / "fluff_cache.json"

    body_gen = BodyGenerator(model=datagen_model, cache_file=fabric_cache)
    scenario.bodies_by_headline_hash = await body_gen.generate_for_scenario(scenario)

    fluff_gen = SearchFluffGenerator(model=datagen_model, cache_file=fluff_cache)
    anchors = [
        (a.query_prefix, a.fluff_count, a.fluff_topic_hint)
        for a in scenario.search_corpus
        if a.fluff_count > 0
    ]
    if anchors:
        scenario.search_fluff_by_prefix = await fluff_gen.generate(anchors)

    save_fabric(scenario, scenarios_dir)

    reloaded = load_scenario(scenarios_dir, scenario.scenario_id)
    audit = await run_audits(reloaded, judge_model=judge_model)
    diversity_path = write_diversity_report(reloaded, audit, scenarios_dir)

    manifest = {
        "scenario_id": scenario.scenario_id,
        "datagen_model": datagen_model,
        "judge_model": judge_model,
        "num_items": len(scenario.timeline),
        "num_sources": len(scenario.source_universe),
        "num_search_anchors": len(scenario.search_corpus),
        "accepted": audit.is_accepted(),
    }
    (scenarios_dir / scenario.scenario_id / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    if not audit.is_accepted():
        raise RuntimeError(f"scenario {scenario.scenario_id} failed audits: {audit.hard_failures}")

    return GenerationResult(
        scenario=reloaded,
        audit=audit,
        skeleton_path=scenarios_dir / scenario.scenario_id / "skeleton.json",
        fabric_path=scenarios_dir / scenario.scenario_id / "fabric.json",
        diversity_path=diversity_path,
    )
