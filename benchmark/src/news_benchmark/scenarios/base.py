"""
Scenario schema and serialization.

A scenario bundles everything needed to drive one end-to-end run: the
persona the simulator plays, the subscription goals the simulator is
trying to achieve, the source universe the Finder can discover, the
content timeline the fake adapters replay, the web-search corpus the
agents hit during research, and the deterministic post-conditions the
harness checks.

Scenarios are split into two files on disk:

  data/scenarios/<id>/skeleton.json   - human-authored: persona, goals,
                                          source universe, headlines, labels,
                                          web-search corpus anchors, assertions
  data/scenarios/<id>/fabric.json     - generated: article bodies, fluff search
                                          snippets. Keyed by content hash of
                                          the corresponding skeleton entry.

Skeleton is the source of truth; fabric is reproducible from it by the
data-generation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from news_benchmark.fakes.adapters import ScenarioItem
from news_benchmark.fakes.search import SearchResult


@dataclass
class Persona:
    """Traits that shape the user simulator's voice and behaviour."""

    language: str = "en"
    timezone: str = "UTC"
    tech_literacy: str = "medium"
    verbosity: str = "medium"
    can_suggest_urls: bool = False


@dataclass
class SubscriptionGoal:
    """One subscription the persona wants to end up with."""

    goal_id: str
    description: str
    expected_user_spec_keywords: list[str]
    expected_delivery_mode: str
    expected_schedule_cron: str | None = None
    expected_digest_language: str | None = None
    expected_webhook_url: str | None = None
    expected_sources_min: int = 3
    expected_sources_max: int = 8


@dataclass
class AssertionSpec:
    """One deterministic post-condition."""

    kind: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass
class SourceEntry:
    """One URL that 'exists' in this scenario's world."""

    url: str
    source_type: str
    description: str
    should_be_picked_by_finder: bool = True


@dataclass
class TimelineEntry:
    """One headline with its label. Body is filled from fabric.json."""

    fake_ts: str
    source_url: str
    headline: str
    difficulty: str
    should_notify_per_sub: dict[str, bool] = field(default_factory=dict)
    should_contribute_to_digest_per_sub: dict[str, bool] = field(default_factory=dict)
    body_style_hint: str = "newsroom"
    body_adversarial: bool = False
    body_language: str = "en"


@dataclass
class SearchCorpusAnchor:
    """One anchor entry: a query prefix plus curated hits, padded with fluff later."""

    query_prefix: str
    curated_results: list[dict[str, str]] = field(default_factory=list)
    fluff_count: int = 4
    fluff_topic_hint: str = ""


@dataclass
class ConversationTurn:
    """One scripted message the simulator must send regardless of agent reply."""

    fake_day: int
    message: str
    comment: str = ""


@dataclass
class Scenario:
    """Fully-resolved scenario with both skeleton labels and generated fabric."""

    scenario_id: str
    persona: Persona
    goals: list[SubscriptionGoal]
    simulated_days: int
    start_date_iso: str
    source_universe: list[SourceEntry]
    timeline: list[TimelineEntry]
    search_corpus: list[SearchCorpusAnchor]
    scripted_turns: list[ConversationTurn]
    assertions: list[AssertionSpec]
    bodies_by_headline_hash: dict[str, str] = field(default_factory=dict)
    search_fluff_by_prefix: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def to_items(self) -> list[ScenarioItem]:
        """Materialize the timeline into runtime ScenarioItem objects."""
        out: list[ScenarioItem] = []
        for t in self.timeline:
            body = self.bodies_by_headline_hash.get(headline_hash(t), "")
            out.append(
                ScenarioItem(
                    fake_ts=datetime.fromisoformat(t.fake_ts),
                    source_url=t.source_url,
                    headline=t.headline,
                    body=body,
                    difficulty=t.difficulty,
                    should_notify_per_sub=t.should_notify_per_sub,
                    should_contribute_to_digest_per_sub=t.should_contribute_to_digest_per_sub,
                )
            )
        return out

    def to_search_corpus(self) -> dict[str, list[SearchResult]]:
        """Resolve the search corpus to runtime SearchResult rows."""
        out: dict[str, list[SearchResult]] = {}
        for anchor in self.search_corpus:
            rows: list[SearchResult] = []
            for r in anchor.curated_results:
                rows.append(
                    SearchResult(title=r["title"], url=r["url"], snippet=r.get("snippet", ""))
                )
            for r in self.search_fluff_by_prefix.get(anchor.query_prefix, []):
                rows.append(
                    SearchResult(title=r["title"], url=r["url"], snippet=r.get("snippet", ""))
                )
            out[anchor.query_prefix] = rows
        return out


def headline_hash(entry: TimelineEntry) -> str:
    """Stable content-addressed key for caching generated bodies."""
    import hashlib

    blob = (
        entry.source_url
        + "|"
        + entry.headline
        + "|"
        + entry.body_style_hint
        + "|"
        + ("A" if entry.body_adversarial else "F")
        + "|"
        + entry.body_language
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def skeleton_path(scenario_dir: Path, scenario_id: str) -> Path:
    return scenario_dir / scenario_id / "skeleton.json"


def fabric_path(scenario_dir: Path, scenario_id: str) -> Path:
    return scenario_dir / scenario_id / "fabric.json"


def save_skeleton(scenario: Scenario, scenario_dir: Path) -> None:
    """Persist only the human-authored skeleton portion."""
    p = skeleton_path(scenario_dir, scenario.scenario_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    skel = {
        "scenario_id": scenario.scenario_id,
        "persona": asdict(scenario.persona),
        "goals": [asdict(g) for g in scenario.goals],
        "simulated_days": scenario.simulated_days,
        "start_date_iso": scenario.start_date_iso,
        "source_universe": [asdict(s) for s in scenario.source_universe],
        "timeline": [asdict(t) for t in scenario.timeline],
        "search_corpus": [asdict(s) for s in scenario.search_corpus],
        "scripted_turns": [asdict(t) for t in scenario.scripted_turns],
        "assertions": [asdict(a) for a in scenario.assertions],
    }
    p.write_text(json.dumps(skel, indent=2, ensure_ascii=False))


def save_fabric(scenario: Scenario, scenario_dir: Path) -> None:
    """Persist only the generated fabric (bodies + search fluff)."""
    p = fabric_path(scenario_dir, scenario.scenario_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    fab = {
        "bodies_by_headline_hash": scenario.bodies_by_headline_hash,
        "search_fluff_by_prefix": scenario.search_fluff_by_prefix,
    }
    p.write_text(json.dumps(fab, indent=2, ensure_ascii=False))


def load_scenario(scenario_dir: Path, scenario_id: str) -> Scenario:
    """Load the skeleton + fabric for a scenario id from disk."""
    skel = json.loads(skeleton_path(scenario_dir, scenario_id).read_text())
    scenario = Scenario(
        scenario_id=skel["scenario_id"],
        persona=Persona(**skel["persona"]),
        goals=[SubscriptionGoal(**g) for g in skel["goals"]],
        simulated_days=skel["simulated_days"],
        start_date_iso=skel["start_date_iso"],
        source_universe=[SourceEntry(**s) for s in skel["source_universe"]],
        timeline=[TimelineEntry(**t) for t in skel["timeline"]],
        search_corpus=[SearchCorpusAnchor(**s) for s in skel["search_corpus"]],
        scripted_turns=[ConversationTurn(**t) for t in skel["scripted_turns"]],
        assertions=[AssertionSpec(**a) for a in skel["assertions"]],
    )
    fab_p = fabric_path(scenario_dir, scenario_id)
    if fab_p.exists():
        fab = json.loads(fab_p.read_text())
        scenario.bodies_by_headline_hash = fab.get("bodies_by_headline_hash", {})
        scenario.search_fluff_by_prefix = fab.get("search_fluff_by_prefix", {})
    return scenario
