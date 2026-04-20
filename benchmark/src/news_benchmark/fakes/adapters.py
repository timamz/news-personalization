"""
Fake RSS / Telegram / Reddit adapters driven by the scenario timeline.

The scenario declares a content_timeline of (fake_ts, source_url, headline,
body) tuples. Each poll cycle asks the fake adapter "what new items have
appeared since the last poll?" The adapter returns items whose fake_ts is
strictly greater than last_polled_at and <= the current virtual clock.

All three adapter types share one implementation (FakeAdapter) — the real
backend differentiates by shape-of-source-record only, not by poll
semantics. The source_type field on the Source row is what the real
system reads to pick which adapter, so we just install FakeAdapter for
every type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ScenarioItem:
    """One line in a scenario's content_timeline."""

    fake_ts: datetime
    source_url: str
    headline: str
    body: str
    text_to_embed: str = ""
    difficulty: str = "easy_positive"
    should_notify_per_sub: dict[str, bool] = field(default_factory=dict)
    should_contribute_to_digest_per_sub: dict[str, bool] = field(default_factory=dict)

    def to_normalized(self) -> dict[str, object]:
        """Shape the real polling pipeline consumes (NormalizedPost fields)."""
        return {
            "url": self._synth_url(),
            "headline": self.headline,
            "body": self.body,
            "text_to_embed": self.text_to_embed or self.headline + "\n" + self.body[:500],
            "published_at": self.fake_ts,
        }

    def _synth_url(self) -> str:
        base = self.source_url.rstrip("/")
        slug = str(abs(hash((self.source_url, self.headline, self.fake_ts.isoformat()))))[:12]
        return f"{base}/item/{slug}"


@dataclass
class FakeAdapter:
    """Shared implementation for all fake source types (RSS / TG / Reddit)."""

    source_url: str
    items: list[ScenarioItem]
    last_served: datetime | None = None

    async def fetch_posts(self, now: datetime) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for item in self.items:
            if self.last_served is not None and item.fake_ts <= self.last_served:
                continue
            if item.fake_ts > now:
                continue
            out.append(item.to_normalized())
        if out:
            self.last_served = max(i.fake_ts for i in self.items if i.fake_ts <= now)
        return out

    def source_name(self) -> str:
        return self.source_url

    def log_label(self) -> str:
        return f"fake({self.source_url})"
