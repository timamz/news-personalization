"""
summary.md matrix writer: scenario x model -> PASS/FAIL, scores, USD, tokens.

Reads every scenario JSON produced this run and renders a compact
human-readable matrix. First deliverable the user looks at.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_summary(run_dir: Path) -> Path:
    """Read every scenario JSON under run_dir/scenarios/ and render summary.md."""
    scenarios_dir = run_dir / "scenarios"
    if not scenarios_dir.exists():
        summary = run_dir / "summary.md"
        summary.write_text("# Benchmark run\n\nNo scenario records found.\n")
        return summary

    by_model: dict[str, dict[str, dict[str, object]]] = {}
    for p in sorted(scenarios_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        scen = rec["scenario_id"]
        model = rec["model_column"]
        by_model.setdefault(model, {})[scen] = rec

    lines: list[str] = []
    lines.append("# Benchmark run\n")
    lines.append(f"- Run id: `{_first_val(by_model, 'run_id')}`")
    lines.append(f"- Total USD: **{_sum_usd(by_model):.4f}**")
    lines.append("")
    lines.append("## Scenario × model matrix\n")
    all_scens = sorted({s for m in by_model.values() for s in m})
    header = "| scenario | " + " | ".join(f"`{m}`" for m in by_model) + " |"
    sep = "|---|" + "|".join(["---"] * len(by_model)) + "|"
    lines.append(header)
    lines.append(sep)
    for scen in all_scens:
        cells = [f"**{scen}**"]
        for m in by_model:
            rec = by_model[m].get(scen)
            if rec is None:
                cells.append("—")
                continue
            passed = rec.get("action_correctness", {}).get("overall_pass")
            usd = rec.get("cost", {}).get("total_usd", 0.0)
            f1 = _mean_f1(rec)
            jm = _mean_judge(rec)
            flag = "PASS" if passed else "FAIL"
            cells.append(f"{flag} · F1 {f1:.2f} · judge {jm:.2f} · ${usd:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    summary = run_dir / "summary.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary


def _first_val(by_model: dict, key: str) -> str:
    for m in by_model.values():
        for rec in m.values():
            if key in rec:
                return str(rec[key])
    return ""


def _sum_usd(by_model: dict) -> float:
    total = 0.0
    for m in by_model.values():
        for rec in m.values():
            total += float(rec.get("cost", {}).get("total_usd", 0.0))
    return total


def _mean_f1(rec: dict) -> float:
    metrics = rec.get("classification", {}).get("per_sub", {})
    vals = [v["f1"] for v in metrics.values() if "f1" in v]
    return sum(vals) / len(vals) if vals else 0.0


def _mean_judge(rec: dict) -> float:
    ratings = rec.get("judge_rubrics", {}).get("digest", [])
    scores: list[float] = []
    for r in ratings:
        for k in ("goal_relevance", "format_adherence", "factual_grounding"):
            if k in r:
                scores.append(float(r[k]))
    return sum(scores) / len(scores) if scores else 0.0
