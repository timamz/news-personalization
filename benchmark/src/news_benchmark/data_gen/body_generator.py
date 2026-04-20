"""
Generates article bodies from headline + style hint, cached by content hash.

Called once per scenario during the data-generation pass. Output is
committed to git alongside the skeleton so every benchmark run sees
byte-identical input regardless of host or LLM non-determinism.

Style hints map to distinct generation prompts so the corpus does not
sound like one author wrote all 300 articles. The LLM temperature is
fixed at 0.2 for fabric generation — low enough to suppress wandering,
high enough to inject word-choice variation across headlines.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import litellm

from news_benchmark.scenarios.base import Scenario, TimelineEntry, headline_hash

STYLE_PROMPTS: dict[str, str] = {
    "newsroom": (
        "Write the article body implied by the headline as a straight newsroom report. "
        "Reuters-style: short lead sentence stating the event, 4-6 paragraphs, plain "
        "declarative voice, one attributed quote if natural. 350-420 words. No opinions."
    ),
    "opinion": (
        "Write the article body as a newspaper op-ed. First-person where useful, clearly "
        "argumentative, takes a side. 380-450 words. One rhetorical question, at most."
    ),
    "wire": (
        "Write the article as a terse wire-service bulletin. No quotes, no colour. "
        "150-220 words, five to seven sentences, dense with numbers and dates."
    ),
    "reddit": (
        "Write as a self-post on a subreddit that would cover this topic. Informal, "
        "opinion-heavy, occasional parenthetical aside, maybe one numbered list. "
        "300-400 words. First-person. Ends with a question to the reader."
    ),
    "telegram": (
        "Write as a Telegram channel post. Short, declarative, one thought per line, "
        "80-140 words total, no quotes, no bylines. Use line breaks between points."
    ),
    "techcrunch": (
        "Write as a tech-blog hot take on the headline. Punchy lead, 3-4 paragraphs, "
        "300-380 words. Mix of reporting and mild editorial framing. Ends on a "
        "forward-looking sentence."
    ),
}

ADVERSARIAL_TWIST = (
    "\n\nIMPORTANT: The *body* should soft-contradict the headline — the headline "
    "implies one thing but the body, read closely, reveals the real story is less "
    "dramatic or even opposite. Do not flag the contradiction explicitly."
)


@dataclass
class BodyGenerator:
    """LLM-backed article-body generator with a content-addressed cache."""

    model: str
    cache_file: Path

    def __post_init__(self) -> None:
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            self._cache = json.loads(self.cache_file.read_text())

    async def generate_for_scenario(self, scenario: Scenario) -> dict[str, str]:
        """Fill bodies for every TimelineEntry; return the updated hash map."""
        tasks: list[tuple[str, asyncio.Task[str]]] = []
        for entry in scenario.timeline:
            h = headline_hash(entry)
            if h in self._cache and self._cache[h]:
                continue
            tasks.append((h, asyncio.create_task(self._one(entry))))

        for h, t in tasks:
            self._cache[h] = await t

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False))
        return {headline_hash(e): self._cache.get(headline_hash(e), "") for e in scenario.timeline}

    async def _one(self, entry: TimelineEntry) -> str:
        style = STYLE_PROMPTS.get(entry.body_style_hint, STYLE_PROMPTS["newsroom"])
        system = (
            "You are a disciplined staff writer producing a realistic news article "
            "body from a headline. Write only the article body (no title, no byline, "
            "no trailing metadata). Language: " + entry.body_language + ". Style:\n" + style
        )
        if entry.body_adversarial:
            system += ADVERSARIAL_TWIST
        user = (
            f"Headline: {entry.headline}\n"
            f"Source: {entry.source_url}\n"
            f"Date: {entry.fake_ts}\n\n"
            "Write the article body now."
        )
        resp = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=900,
        )
        content = resp["choices"][0]["message"]["content"] or ""
        return content.strip()
