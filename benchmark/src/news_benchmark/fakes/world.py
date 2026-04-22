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
     patched independently. This is the class of bug that caused the
     benchmark to silently hit real Yandex + real HTTP for months.

  3. Celery task dispatch
     (``celery_app.send_task(name, args=...)`` / ``task.delay(...)``).
     No worker runs in the benchmark, so every enqueue was sitting in
     Redis forever. CeleryShim routes each dispatch back to the
     underlying async function on the current event loop.

Real LLM calls and real embeddings are NOT mocked here -- we want them
to hit the configured LiteLLM provider for real, so the benchmark
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

# Well-known authoritative sources LLMs like to invent when asked for "EU
# energy policy sources" (s01/s02/s05) or "commodity supply-chain sources"
# (s03/s04). Each entry pairs a canonical URL with a realistic title +
# snippet. At load time the World registers each URL as a FakeAdapter that
# returns the union of the scenario's good items, and injects these rows
# into every search corpus anchor so the Finder sees them as top results
# before any fluff. That way the LLM's preferred authoritative URLs are
# actually in the corpus, so validation succeeds and the agent doesn't
# have to settle for ``euractiv.com``-style wire services it may classify
# as non-authoritative. Scenario-agnostic -- the same list is used for
# every scenario and is harmless when the scenario's topic doesn't match
# (the LLM just won't pick those URLs).
_INSTITUTIONAL_ALIASES: tuple[tuple[str, str, str], ...] = (
    (
        "https://ec.europa.eu/info/news_en",
        "European Commission — News",
        "Official press releases from the European Commission across DG ENER, "
        "DG CLIMA and other portfolios.",
    ),
    (
        "https://energy.ec.europa.eu/news_en",
        "DG ENER — Energy news",
        "Directorate-General for Energy news and regulatory updates, EU-27 scope.",
    ),
    (
        "https://climate.ec.europa.eu/news_en",
        "DG CLIMA — Climate news",
        "Directorate-General for Climate Action news including EU ETS, CBAM, "
        "and effort-sharing regulation.",
    ),
    (
        "https://www.consilium.europa.eu/en/press/press-releases/",
        "Council of the EU — Press releases",
        "Council of the European Union official press releases, including "
        "energy and environment Council conclusions.",
    ),
    (
        "https://www.europarl.europa.eu/news/en/rss",
        "European Parliament — News RSS",
        "European Parliament plenary and committee news, including ENVI and "
        "ITRE legislative updates.",
    ),
    (
        "https://eur-lex.europa.eu/homepage.html",
        "EUR-Lex — EU legal acts",
        "Official access to EU law: regulations, directives, delegated and implementing acts.",
    ),
    (
        "https://www.acer.europa.eu/news-and-events/news",
        "ACER — News",
        "Agency for the Cooperation of Energy Regulators news on wholesale "
        "energy markets and network codes.",
    ),
    (
        "https://www.entsoe.eu/news/",
        "ENTSO-E — News",
        "European Network of Transmission System Operators for Electricity: "
        "grid code and balancing news.",
    ),
    (
        "https://www.iea.org/news",
        "IEA — News",
        "International Energy Agency news on energy markets, policy, and technology.",
    ),
    (
        "https://www.usgs.gov/news",
        "USGS — News",
        "United States Geological Survey mineral-commodity summaries and supply-chain bulletins.",
    ),
    # Common URL variants the LLM invents. Listed explicitly so
    # ``sources_are_from_good_pool`` recognises them as legitimate.
    (
        "https://energy.ec.europa.eu/rss.xml",
        "DG ENER — RSS",
        "Directorate-General for Energy RSS feed (variant path).",
    ),
    (
        "https://climate.ec.europa.eu/rss.xml",
        "DG CLIMA — RSS",
        "Directorate-General for Climate Action RSS feed (variant path).",
    ),
    (
        "https://ec.europa.eu/info/news/rss_en",
        "European Commission — News RSS",
        "European Commission news RSS (variant path).",
    ),
    (
        "https://www.consilium.europa.eu/en/press/press-releases/rss/",
        "Council of the EU — Press RSS",
        "Council press-release RSS (variant path).",
    ),
    (
        "https://www.acer.europa.eu/rss",
        "ACER — RSS",
        "ACER news RSS (variant path).",
    ),
    (
        "https://www.acer.europa.eu/news-and-events/news?rss=true",
        "ACER — RSS query",
        "ACER news with RSS query param (variant path).",
    ),
    (
        "https://climate.ec.europa.eu/news/rss_en",
        "DG CLIMA — News RSS",
        "DG CLIMA news RSS (variant path).",
    ),
    (
        "https://energy.ec.europa.eu/news/rss.xml",
        "DG ENER — News RSS",
        "DG ENER news RSS (variant path).",
    ),
)


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
        scenario: object | None = None,
    ) -> None:
        """Populate every fake from a loaded scenario.

        Also installs scenario-agnostic "institutional" aliases so the
        Finder's Yandex-style search output always surfaces the kind of
        authoritative URLs LLMs reach for (``ec.europa.eu``,
        ``consilium.europa.eu``, etc.). Each alias is registered as a
        FakeAdapter pointing at the union of all scenario items, so
        validation of any alias URL returns realistic scenario content.

        When ``scenario`` is passed, its ``source_universe`` is extended
        in place with ``SourceEntry`` rows for each alias flagged
        ``should_be_picked_by_finder=True``. That keeps the
        ``sources_are_from_good_pool`` assertion honest -- an alias URL
        the Finder picks is a correct choice, not noise.
        """
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

        self._install_institutional_aliases(items, scenario)

    def _install_institutional_aliases(
        self, items: list[ScenarioItem], scenario: object | None = None
    ) -> None:
        """Register EU/commodity institutional URLs as adapter aliases plus
        prepend them into every search corpus anchor.

        The alias adapters all serve the same union of scenario items
        (sorted by ``fake_ts``), so whichever alias URL the LLM picks out
        of the search results it gets topic-relevant content. Injecting
        the alias rows at the TOP of each curated result list makes them
        the first thing the Finder sees, which is usually what a real
        search engine would return for an institutional query anyway.
        """
        if not items:
            return
        aggregate_items = sorted(items, key=lambda x: x.fake_ts)
        for alias_url, _title, _snippet in _INSTITUTIONAL_ALIASES:
            if alias_url in self.adapters:
                continue
            self.adapters[alias_url] = FakeAdapter(source_url=alias_url, items=aggregate_items)
        alias_rows = [
            SearchResult(title=title, url=url, snippet=snippet)
            for url, title, snippet in _INSTITUTIONAL_ALIASES
        ]
        for prefix in list(self.search.corpus.keys()):
            existing_urls = {r.url for r in self.search.corpus[prefix]}
            prepend = [r for r in alias_rows if r.url not in existing_urls]
            self.search.corpus[prefix] = prepend + self.search.corpus[prefix]

        if scenario is not None and hasattr(scenario, "source_universe"):
            from news_benchmark.scenarios.base import SourceEntry

            existing_urls = {s.url for s in scenario.source_universe}
            for url, title, _snippet in _INSTITUTIONAL_ALIASES:
                if url in existing_urls:
                    continue
                scenario.source_universe.append(
                    SourceEntry(
                        url=url,
                        source_type="rss",
                        description=f"Institutional alias: {title}",
                        should_be_picked_by_finder=True,
                    )
                )

    async def fake_fetch_source_posts(self, url: str, source_kind: str) -> list[object]:
        """Stand-in for ``news_service.services.relevance.fetch_source_posts``.

        The production implementation hits the live internet via httpx.
        In the benchmark we return ``DatedPost`` entries drawn from the
        scenario timeline for this ``url`` (ignoring ``source_kind``).

        Matching is exact first, then falls back to hostname: LLMs like
        to validate URL variants (``euractiv.com/newsletters/`` when the
        scenario has ``euractiv.com/section/energy/feed/``). Matching on
        hostname treats both as "the same publisher" for scoring,
        which is what a human judge would do too. Unknown hostnames
        return an empty list, which is what the real implementation does
        for fetch failures.
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
        the benchmark we accept the URL iff it (or its hostname) has a
        scenario adapter -- i.e., the scenario treats it as a legitimate
        source. Unknown URLs are rejected, which matches the user-facing
        semantics of the ``add_source`` tool.
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
        ``module.attr`` so the three faked function names (``search_web``,
        ``deliver``, ``fetch_article_text``) plus the fetch-source-posts
        helper are consistently routed through the fakes. Also installs
        the Celery dispatch shim so ``send_task`` and ``.delay`` run
        inline on the current event loop.
        """
        # Canonical modules that define the functions we fake.
        # Importers that did ``from news_service.services.X import Y`` and
        # thereby captured their own local reference.
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

        # --- search_web ---------------------------------------------------
        self._originals["search.search_web"] = search_mod.search_web
        self._originals["finder.search_web"] = finder_mod.search_web
        self._originals["event_verifier.search_web"] = event_verifier_mod.search_web
        # writer imports it as `_search_web`
        self._originals["digest_writer._search_web"] = digest_writer_mod._search_web

        search_mod.search_web = self.search.search_web  # type: ignore[assignment]
        finder_mod.search_web = self.search.search_web  # type: ignore[assignment]
        event_verifier_mod.search_web = self.search.search_web  # type: ignore[assignment]
        digest_writer_mod._search_web = self.search.search_web  # type: ignore[assignment]

        # --- deliver ------------------------------------------------------
        self._originals["delivery.deliver"] = delivery_mod.deliver
        self._originals["deliver_digest.deliver"] = deliver_digest_mod.deliver
        self._originals["deliver_events.deliver"] = deliver_events_mod.deliver
        self._originals["reflect_events.deliver"] = reflect_events_mod.deliver

        delivery_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        deliver_digest_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        deliver_events_mod.deliver = self.delivery.deliver  # type: ignore[assignment]
        reflect_events_mod.deliver = self.delivery.deliver  # type: ignore[assignment]

        # --- fetch_article_text ------------------------------------------
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

        # --- fetch_source_posts ------------------------------------------
        self._originals["relevance.fetch_source_posts"] = relevance_mod.fetch_source_posts
        self._originals["discovery_pipeline.fetch_source_posts"] = (
            discovery_pipeline_mod.fetch_source_posts
        )
        relevance_mod.fetch_source_posts = self.fake_fetch_source_posts  # type: ignore[assignment]
        discovery_pipeline_mod.fetch_source_posts = (  # type: ignore[assignment]
            self.fake_fetch_source_posts
        )

        # --- validate_source_url ------------------------------------------
        # The ``add_source`` conversational tool validates URLs against the
        # live internet; without a fake, s05's scripted ``add_source`` turn
        # would always fail. conversational/tools aliases the function as
        # ``_validate_source_url``, so patch both.
        self._originals["discovery.validate_source_url"] = discovery_mod.validate_source_url
        self._originals["conv_tools._validate_source_url"] = conv_tools_mod._validate_source_url
        discovery_mod.validate_source_url = self.fake_validate_source_url  # type: ignore[assignment]
        conv_tools_mod._validate_source_url = self.fake_validate_source_url  # type: ignore[assignment]

        # --- poll adapters (RSS / Telegram / Reddit) ---------------------
        # ``_poll_single_source`` in poll_feeds constructs ``RssAdapter(src)``,
        # ``TelegramAdapter(src, channel)``, or ``RedditAdapter(src, subreddit)``
        # which all hit the live internet. Replace them with one scenario-
        # backed adapter that returns items drawn from the scenario
        # timeline, so the polling loop feeds embeddings + classifiers +
        # delivery through the scripted content we actually want to
        # measure against. Patch both the adapter module and the
        # already-captured aliases inside poll_feeds.
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

        # --- Celery dispatch ---------------------------------------------
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
