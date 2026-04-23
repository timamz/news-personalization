"""
FakeSearch replaces news_service.services.search.search_web.

The scenario declares a `web_search_corpus`: a dict mapping query-prefix
strings to an ordered list of result rows `(title, url, snippet)`. The
fake matches the longest prefix it finds and returns up to ten results
formatted exactly like the production service:

    - Title: URL
      Snippet
    - Title2: URL2
      Snippet2

If no prefix matches, returns an empty string (which the real service
also does for zero-hit queries). Every call is recorded so the report
can show which queries the agent ran.

Installation replaces the module-level `search_web` reference used by
news_service; direct callers that imported the function by name into
their own module need to be patched on those modules too. The scenario
loader is aware of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class FakeSearch:
    """Scripted web-search backend, keyed by longest matching query prefix."""

    corpus: dict[str, list[SearchResult]] = field(default_factory=dict)
    call_log: list[str] = field(default_factory=list)

    async def search_web(self, query: str) -> str:
        """Return up to 10 scripted results for `query` in production format."""
        self.call_log.append(query)
        matched = self._match(query)
        if not matched:
            return ""
        lines = []
        for r in matched[:10]:
            lines.append(f"- {r.title}: {r.url}")
            lines.append(f"  {r.snippet}")
        return "\n".join(lines)

    def _match(self, query: str) -> list[SearchResult]:
        best_prefix = ""
        best: list[SearchResult] = []
        q_lower = query.lower().strip()
        for prefix, rows in self.corpus.items():
            p_lower = prefix.lower().strip()
            if q_lower.startswith(p_lower) and len(p_lower) > len(best_prefix):
                best_prefix = p_lower
                best = rows
        if not best:
            for prefix, rows in self.corpus.items():
                if any(tok in q_lower for tok in prefix.lower().split()):
                    return rows
        return best
