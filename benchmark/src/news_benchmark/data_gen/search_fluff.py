"""
Generates noise/fluff search results around curated anchor hits.

The scenario author writes the curated results per query prefix (what we
WANT the Finder/Writer/Verifier to find). The fluff is LLM-generated:
plausible-but-irrelevant snippets that pad the result list to 8-10 rows,
simulating how real search engines return a mix of good and bad hits.

Cached per query prefix. Regenerated only when the prefix changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import litellm


def _prefix_key(prefix: str, count: int, hint: str) -> str:
    blob = f"{prefix}|{count}|{hint}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


@dataclass
class SearchFluffGenerator:
    """Generates plausible off-topic search snippets around a query prefix."""

    model: str
    cache_file: Path

    def __post_init__(self) -> None:
        self._cache: dict[str, list[dict[str, str]]] = {}
        if self.cache_file.exists():
            self._cache = json.loads(self.cache_file.read_text())

    async def generate(
        self, anchors: list[tuple[str, int, str]]
    ) -> dict[str, list[dict[str, str]]]:
        """For each (prefix, count, hint), return `count` fluff rows."""
        tasks: list[tuple[str, str, asyncio.Task[list[dict[str, str]]]]] = []
        for prefix, count, hint in anchors:
            key = _prefix_key(prefix, count, hint)
            if key in self._cache and len(self._cache[key]) >= count:
                continue
            tasks.append((prefix, key, asyncio.create_task(self._fluff(prefix, count, hint))))

        for _prefix, key, t in tasks:
            self._cache[key] = await t

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False))

        out: dict[str, list[dict[str, str]]] = {}
        for prefix, count, hint in anchors:
            out[prefix] = self._cache[_prefix_key(prefix, count, hint)]
        return out

    async def _fluff(self, prefix: str, count: int, hint: str) -> list[dict[str, str]]:
        system = (
            "You generate realistic-looking but OFF-TOPIC search result snippets. "
            "Return exactly "
            f"{count}"
            " JSON objects in a JSON array. Each object has keys title, url, snippet. "
            "Snippets must read as if they came from a real search engine but be "
            "tangentially related at best — they are distractors. URLs must look real "
            "but be plausible domains (news sites, blogs, wikipedia, reddit, etc.)."
        )
        user = (
            f"Query prefix: {prefix}\nTopic hint for the distractors: {hint}\n"
            "Return the JSON array only."
        )
        resp = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = resp["choices"][0]["message"]["content"] or "{}"
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        if not isinstance(parsed, list):
            parsed = []
        cleaned: list[dict[str, str]] = []
        for r in parsed[:count]:
            if isinstance(r, dict) and "title" in r and "url" in r:
                cleaned.append(
                    {
                        "title": str(r["title"])[:200],
                        "url": str(r["url"])[:400],
                        "snippet": str(r.get("snippet", ""))[:400],
                    }
                )
        return cleaned
