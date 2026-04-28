"""
FakeArticleFetch replaces news_service.services.article_fetch.fetch_article_text.

Maps URL -> body string from the scenario's content_timeline. Returns None
(mirroring real behavior on fetch failure) for URLs not in the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeArticleFetch:
    """URL -> body string lookup used for content enrichment at ingest time."""

    bodies: dict[str, str] = field(default_factory=dict)
    call_log: list[str] = field(default_factory=list)

    async def fetch_article_text(
        self,
        url: str,
        *,
        max_chars: int,
        **_kwargs: object,
    ) -> str | None:
        self.call_log.append(url)
        body = self.bodies.get(url)
        if body is None:
            return None
        return body[:max_chars]
