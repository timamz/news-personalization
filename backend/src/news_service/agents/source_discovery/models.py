"""Shared data models for the source discovery pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit", "twitter_account"]


class ScoredSource(BaseModel):
    url: str = Field(..., description="Canonical source URL")
    title: str = Field(default="", description="Human-readable source title")
    source_kind: SourceKind = Field(..., description="Source type")
    relevance_score: float = Field(
        ..., description="0.0-1.0 content relevance score (higher is better)"
    )


class SourceDiscoveryResult(BaseModel):
    sources: list[ScoredSource] = Field(..., description="Selected sources ranked by relevance")


class DiscoveryPlan(BaseModel):
    strategies: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Independent search strategies to execute in parallel",
    )
