"""
World wires every fake into news_service's module globals.

The backend modules under test reference their dependencies in three
different ways, and the harness has to follow every one:

  1. ``module.attr`` lookup at call time
     (e.g. ``news_service.services.search.search_web(...)``).
     One monkey-patch of ``module.attr`` covers every caller.

  2. ``from module import attr`` at module-import time
     (e.g. ``from news_service.services.search import search_web``).
     The importing module now has its OWN local reference, unaffected
     by a patch on the original module. Each such importer has to be
     patched independently.

  3. Celery task dispatch
     (``celery_app.send_task(name, args=...)`` / ``task.delay(...)``).
     No worker runs in the benchmark, so every enqueue was sitting in
     Redis forever. CeleryShim routes each dispatch back to the
     underlying async function on the current event loop.

Real LLM calls and real embeddings are NOT mocked here -- we want them
to hit the configured LiteLLM provider for real, so the harness
measures actual model behavior. Only the "outside world" (search,
HTTP fetch, source polling, webhook delivery, Celery) is faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem, make_scenario_poll_adapter
from news_benchmark.fakes.article_fetch import FakeArticleFetch
from news_benchmark.fakes.celery_shim import CeleryShim
from news_benchmark.fakes.delivery import FakeDelivery
from news_benchmark.fakes.search import FakeSearch, SearchResult


@dataclass
class World:
    """Container for every fake, ready to be installed against news_service."""

    search: FakeSearch = field(default_factory=FakeSearch)
    delivery: FakeDelivery = field(default_factory=FakeDelivery)
    article_fetch: FakeArticleFetch = field(default_factory=FakeArticleFetch)
    adapters: dict[str, FakeAdapter] = field(default_factory=dict)
    celery: CeleryShim = field(default_factory=CeleryShim)

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

    async def fake_fetch_source_posts(self, url: str, source_kind: str) -> list[object]:
        """Stand-in for ``news_service.services.relevance.fetch_source_posts``.

        The production implementation hits the live internet via httpx.
        In the harness we return ``DatedPost`` entries drawn from the
        fake adapter for this ``url`` (ignoring ``source_kind``).

        Matching is exact first, then falls back to hostname: LLMs like
        to validate URL variants. Unknown hostnames return an empty
        list, which is what the real implementation does for fetch
        failures.
        """
        from datetime import UTC
        from urllib.parse import urlparse

        from news_service.services.relevance import DatedPost

        adapter = self.adapters.get(url)
        if adapter is None:
            host = (urlparse(url).hostname or "").lower()
            if host:
                for key, candidate in self.adapters.items():
                    if (urlparse(key).hostname or "").lower() == host:
                        adapter = candidate
                        break
        if adapter is None:
            return []
        posts: list[DatedPost] = []
        for item in adapter.items:
            text = (item.headline + "\n\n" + (item.body or "")).strip()
            if not text:
                continue
            published = item.fake_ts
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            posts.append(DatedPost(text=text, published_at=published))
        return posts

    async def fake_validate_source_url(self, url: str, *, source_kind: str = "rss") -> bool:
        """Stand-in for ``news_service.agents.discovery.validate_source_url``.

        The production check hits the live internet (httpx GET for RSS,
        telegram scraping for channels, reddit JSON for subreddits). In
        the harness we accept the URL iff it (or its hostname) has a
        registered adapter. Unknown URLs are rejected, matching the
        user-facing semantics of the ``add_source`` tool.
        """
        from urllib.parse import urlparse

        if url in self.adapters:
            return True
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        return any((urlparse(key).hostname or "").lower() == host for key in self.adapters)

    def install(self) -> None:
        """Install fakes by replacing module-level references in news_service.

        Patches every ``from X import Y`` alias along with the canonical
        ``module.attr`` so the faked function names are consistently
        routed through the fakes. Also installs the Celery dispatch shim
        so ``send_task`` and ``.delay`` run inline on the current event
        loop.
        """
        from news_service.agents import discovery as discovery_mod
        from news_service.agents import web_tools as web_tools_mod
        from news_service.agents.conversational import tools as conv_tools_mod
        from news_service.agents.digest import writer as digest_writer_mod
        from news_service.agents.event import verifier as event_verifier_mod
        from news_service.agents.source_discovery import finder as finder_mod
        from news_service.agents.source_discovery import pipeline as discovery_pipeline_mod
        from news_service.services import article_fetch as article_fetch_mod
        from news_service.services import delivery as delivery_mod
        from news_service.services import relevance as relevance_mod
        from news_service.services import search as search_mod
        from news_service.tasks import deliver_digest as deliver_digest_mod
        from news_service.tasks import deliver_events as deliver_events_mod
        from news_service.tasks import poll_adapters as poll_adapters_mod
        from news_service.tasks import poll_feeds as poll_feeds_mod
        from news_service.tasks import reflect_events as reflect_events_mod

        self._originals["search.search_web"] = search_mod.search_web
        self._originals["finder.search_web"] = finder_mod.search_web
        self._originals["event_verifier.search_web"] = event_verifier_mod.search_web
        self._originals["digest_writer._search_web"] = digest_writer_mod._search_web

        search_mod.search_web = self.search.search_web  # type: ignore[assignment]
        finder_mod.search_web = self.search.search_web  # type: ignore[assignment]
        event_verifier_mod.search_web = self.search.search_web  # type: ignore[assignment]
        digest_writer_mod._search_web = self.search.search_web  # type: ignore[assignment]

        self._originals["delivery.deliver"] = delivery_mod.deliver
        self._originals["deliver_digest.deliver"] = deliver_digest_mod.deliver
        self._originals["deliver_events.deliver"] = deliver_events_mod.deliver
        self._originals["reflect_events.deliver"] = reflect_events_mod.deliver

        delivery_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        deliver_digest_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        deliver_events_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        reflect_events_mod.deliver = self.delivery.deliver  # type: ignore[assignment]

        self._originals["article_fetch.fetch_article_text"] = article_fetch_mod.fetch_article_text
        self._originals["relevance.fetch_article_text"] = relevance_mod.fetch_article_text
        self._originals["poll_adapters.fetch_article_text"] = poll_adapters_mod.fetch_article_text
        self._originals["web_tools.fetch_article_text"] = web_tools_mod.fetch_article_text

        article_fetch_mod.fetch_article_text = (  # type: ignore[assignment]
            self.article_fetch.fetch_article_text
        )
        relevance_mod.fetch_article_text = (  # type: ignore[assignment]
            self.article_fetch.fetch_article_text
        )
        poll_adapters_mod.fetch_article_text = (  # type: ignore[assignment]
            self.article_fetch.fetch_article_text
        )
        web_tools_mod.fetch_article_text = (  # type: ignore[assignment]
            self.article_fetch.fetch_article_text
        )

        self._originals["relevance.fetch_source_posts"] = relevance_mod.fetch_source_posts
        self._originals["discovery_pipeline.fetch_source_posts"] = (
            discovery_pipeline_mod.fetch_source_posts
        )
        relevance_mod.fetch_source_posts = self.fake_fetch_source_posts  # type: ignore[assignment]
        discovery_pipeline_mod.fetch_source_posts = (  # type: ignore[assignment]
            self.fake_fetch_source_posts
        )

        self._originals["discovery.validate_source_url"] = discovery_mod.validate_source_url
        self._originals["conv_tools._validate_source_url"] = conv_tools_mod._validate_source_url
        discovery_mod.validate_source_url = self.fake_validate_source_url  # type: ignore[assignment]
        conv_tools_mod._validate_source_url = self.fake_validate_source_url  # type: ignore[assignment]

        scenario_cls = make_scenario_poll_adapter(self.adapters)
        self._originals["poll_adapters.RssAdapter"] = poll_adapters_mod.RssAdapter
        self._originals["poll_adapters.TelegramAdapter"] = poll_adapters_mod.TelegramAdapter
        self._originals["poll_adapters.RedditAdapter"] = poll_adapters_mod.RedditAdapter
        self._originals["poll_feeds.RssAdapter"] = poll_feeds_mod.RssAdapter
        self._originals["poll_feeds.TelegramAdapter"] = poll_feeds_mod.TelegramAdapter
        self._originals["poll_feeds.RedditAdapter"] = poll_feeds_mod.RedditAdapter
        poll_adapters_mod.RssAdapter = scenario_cls  # type: ignore[assignment]
        poll_adapters_mod.TelegramAdapter = scenario_cls  # type: ignore[assignment]
        poll_adapters_mod.RedditAdapter = scenario_cls  # type: ignore[assignment]
        poll_feeds_mod.RssAdapter = scenario_cls  # type: ignore[assignment]
        poll_feeds_mod.TelegramAdapter = scenario_cls  # type: ignore[assignment]
        poll_feeds_mod.RedditAdapter = scenario_cls  # type: ignore[assignment]

        self.celery.install()

    def uninstall(self) -> None:
        """Restore originals. Safe to call even if install() was never called."""
        if not self._originals:
            return
        from news_service.agents import discovery as discovery_mod
        from news_service.agents import web_tools as web_tools_mod
        from news_service.agents.conversational import tools as conv_tools_mod
        from news_service.agents.digest import writer as digest_writer_mod
        from news_service.agents.event import verifier as event_verifier_mod
        from news_service.agents.source_discovery import finder as finder_mod
        from news_service.agents.source_discovery import pipeline as discovery_pipeline_mod
        from news_service.services import article_fetch as article_fetch_mod
        from news_service.services import delivery as delivery_mod
        from news_service.services import relevance as relevance_mod
        from news_service.services import search as search_mod
        from news_service.tasks import deliver_digest as deliver_digest_mod
        from news_service.tasks import deliver_events as deliver_events_mod
        from news_service.tasks import poll_adapters as poll_adapters_mod
        from news_service.tasks import poll_feeds as poll_feeds_mod
        from news_service.tasks import reflect_events as reflect_events_mod

        discovery_mod.validate_source_url = self._originals[  # type: ignore[assignment]
            "discovery.validate_source_url"
        ]
        conv_tools_mod._validate_source_url = self._originals[  # type: ignore[assignment]
            "conv_tools._validate_source_url"
        ]

        search_mod.search_web = self._originals["search.search_web"]  # type: ignore[assignment]
        finder_mod.search_web = self._originals["finder.search_web"]  # type: ignore[assignment]
        event_verifier_mod.search_web = self._originals[  # type: ignore[assignment]
            "event_verifier.search_web"
        ]
        digest_writer_mod._search_web = self._originals[  # type: ignore[assignment]
            "digest_writer._search_web"
        ]

        delivery_mod.deliver = self._originals["delivery.deliver"]  # type: ignore[assignment]
        deliver_digest_mod.deliver = self._originals[  # type: ignore[assignment]
            "deliver_digest.deliver"
        ]
        deliver_events_mod.deliver = self._originals[  # type: ignore[assignment]
            "deliver_events.deliver"
        ]
        reflect_events_mod.deliver = self._originals[  # type: ignore[assignment]
            "reflect_events.deliver"
        ]

        article_fetch_mod.fetch_article_text = self._originals[  # type: ignore[assignment]
            "article_fetch.fetch_article_text"
        ]
        relevance_mod.fetch_article_text = self._originals[  # type: ignore[assignment]
            "relevance.fetch_article_text"
        ]
        poll_adapters_mod.fetch_article_text = self._originals[  # type: ignore[assignment]
            "poll_adapters.fetch_article_text"
        ]
        web_tools_mod.fetch_article_text = self._originals[  # type: ignore[assignment]
            "web_tools.fetch_article_text"
        ]

        relevance_mod.fetch_source_posts = self._originals[  # type: ignore[assignment]
            "relevance.fetch_source_posts"
        ]
        discovery_pipeline_mod.fetch_source_posts = self._originals[  # type: ignore[assignment]
            "discovery_pipeline.fetch_source_posts"
        ]

        poll_adapters_mod.RssAdapter = self._originals[  # type: ignore[assignment]
            "poll_adapters.RssAdapter"
        ]
        poll_adapters_mod.TelegramAdapter = self._originals[  # type: ignore[assignment]
            "poll_adapters.TelegramAdapter"
        ]
        poll_adapters_mod.RedditAdapter = self._originals[  # type: ignore[assignment]
            "poll_adapters.RedditAdapter"
        ]
        poll_feeds_mod.RssAdapter = self._originals[  # type: ignore[assignment]
            "poll_feeds.RssAdapter"
        ]
        poll_feeds_mod.TelegramAdapter = self._originals[  # type: ignore[assignment]
            "poll_feeds.TelegramAdapter"
        ]
        poll_feeds_mod.RedditAdapter = self._originals[  # type: ignore[assignment]
            "poll_feeds.RedditAdapter"
        ]

        self.celery.uninstall()
        self._originals.clear()
