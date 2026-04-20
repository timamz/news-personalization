"""
World wires every fake into news_service's module globals.

The backend modules under test reference their dependencies by name
(`search_web`, `deliver`, `fetch_article_text`, adapter registry lookups).
World.install() monkey-patches those module globals to point at the
supplied fake instances. uninstall() restores them — used for test
isolation but not strictly required in production benchmark runs, since
the harness creates a fresh process per invocation.

Real LLM calls and real embeddings are NOT mocked here — we want them to
hit the configured LiteLLM provider for real, so the benchmark measures
actual model behavior. Only the "outside world" (search, fetch, poll,
deliver) is faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem
from news_benchmark.fakes.article_fetch import FakeArticleFetch
from news_benchmark.fakes.delivery import FakeDelivery
from news_benchmark.fakes.search import FakeSearch, SearchResult


@dataclass
class World:
    """Container for every fake, ready to be installed against news_service."""

    search: FakeSearch = field(default_factory=FakeSearch)
    delivery: FakeDelivery = field(default_factory=FakeDelivery)
    article_fetch: FakeArticleFetch = field(default_factory=FakeArticleFetch)
    adapters: dict[str, FakeAdapter] = field(default_factory=dict)

    _originals: dict[str, object] = field(default_factory=dict, init=False)

    def load_scenario(
        self,
        items: list[ScenarioItem],
        search_corpus: dict[str, list[SearchResult]],
    ) -> None:
        """Populate every fake from a loaded scenario."""
        self.search.corpus.update(search_corpus)
        for item in items:
            synth_url = item.to_normalized()["url"]
            self.article_fetch.bodies[str(synth_url)] = item.body
        by_source: dict[str, list[ScenarioItem]] = {}
        for item in items:
            by_source.setdefault(item.source_url, []).append(item)
        for source_url, src_items in by_source.items():
            self.adapters[source_url] = FakeAdapter(
                source_url=source_url, items=sorted(src_items, key=lambda x: x.fake_ts)
            )

    def install(self) -> None:
        """Install fakes by replacing module-level references in news_service."""
        from news_service.services import article_fetch as article_fetch_mod
        from news_service.services import delivery as delivery_mod
        from news_service.services import search as search_mod

        self._originals["search.search_web"] = search_mod.search_web
        self._originals["delivery.deliver"] = delivery_mod.deliver
        self._originals["article_fetch.fetch_article_text"] = article_fetch_mod.fetch_article_text

        search_mod.search_web = self.search.search_web  # type: ignore[assignment]
        delivery_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        article_fetch_mod.fetch_article_text = (  # type: ignore[assignment]
            self.article_fetch.fetch_article_text
        )

    def uninstall(self) -> None:
        """Restore originals. Safe to call even if install() was never called."""
        if not self._originals:
            return
        from news_service.services import article_fetch as article_fetch_mod
        from news_service.services import delivery as delivery_mod
        from news_service.services import search as search_mod

        search_mod.search_web = self._originals["search.search_web"]  # type: ignore[assignment]
        delivery_mod.deliver = self._originals["delivery.deliver"]  # type: ignore[assignment]
        article_fetch_mod.fetch_article_text = self._originals[  # type: ignore[assignment]
            "article_fetch.fetch_article_text"
        ]
        self._originals.clear()
