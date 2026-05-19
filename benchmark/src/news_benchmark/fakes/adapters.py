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

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScenarioItem:
    """One line in a scenario's content_timeline."""

    fake_ts: datetime
    source_url: str
    headline: str
    body: str
    text_to_embed: str = ""

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

def make_scenario_poll_adapter(adapters_by_url: dict[str, FakeAdapter]):
    """Return a class that impersonates ``news_service.tasks.poll_adapters``
    RSS / Telegram / Reddit adapters but draws items from the scenario
    timeline instead of hitting the real internet.

    We need a class rather than an instance because ``_poll_single_source``
    constructs the adapter itself via ``RssAdapter(src)`` etc. Closing
    over the ``adapters_by_url`` map lets the returned class look up the
    scenario items for a given ``Source.url`` at call time. Falls back to
    hostname matching so a source whose URL drifted (e.g. canonical path
    differs but hostname is the same) still resolves to a scenario
    adapter.
    """
    from datetime import UTC
    from urllib.parse import urlparse

    from news_service.tasks.poll_adapters import NormalizedPost

    from news_benchmark.clock import CLOCK

    def _aware(ts: datetime) -> datetime:
        """Force ts-naive scenario fake_ts to UTC-aware for comparisons with CLOCK.now()."""
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    class ScenarioPollAdapter:
        """Adapter used in the benchmark in place of RssAdapter/TelegramAdapter/RedditAdapter."""

        def __init__(self, src, *_ignored_args, **_ignored_kwargs) -> None:  # type: ignore[no-untyped-def]
            self._src = src
            self._url = src.url

        def source_name(self) -> str:
            return getattr(self._src, "title", None) or self._url

        def log_label(self) -> str:
            return f"scenario({self._url})"

        async def fetch_posts(self) -> list[NormalizedPost]:
            adapter = adapters_by_url.get(self._url)
            if adapter is None:
                host = (urlparse(self._url).hostname or "").lower()
                if host:
                    for key, candidate in adapters_by_url.items():
                        if (urlparse(key).hostname or "").lower() == host:
                            adapter = candidate
                            break
            if adapter is None:
                return []
            now = CLOCK.now()
            last_served = _aware(adapter.last_served) if adapter.last_served is not None else None
            out: list[NormalizedPost] = []
            freshest: datetime | None = last_served
            for item in adapter.items:
                published = _aware(item.fake_ts)
                if last_served is not None and published <= last_served:
                    continue
                if published > now:
                    continue
                norm = item.to_normalized()
                out.append(
                    NormalizedPost(
                        url=str(norm["url"]),
                        headline=str(norm["headline"]),
                        body=str(norm["body"]),
                        text_to_embed=str(norm["text_to_embed"]),
                        published_at=published,
                    )
                )
                if freshest is None or published > freshest:
                    freshest = published
            if out:
                adapter.last_served = freshest
            return out

    return ScenarioPollAdapter
