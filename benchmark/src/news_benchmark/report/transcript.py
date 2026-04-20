"""
Renders the simulator-agent conversation as a readable markdown transcript.

Used for human post-mortem of a scenario run.
"""

from __future__ import annotations

from pathlib import Path


def render_transcript(
    path: Path, *, scenario_id: str, model_column: str, turns: list[dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Transcript — `{scenario_id}` / `{model_column}`\n")
    for t in turns:
        role = t["speaker"]
        banner = "**User**" if role == "user" else "**Agent**"
        lines.append(banner)
        lines.append("")
        lines.append(t["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines))
