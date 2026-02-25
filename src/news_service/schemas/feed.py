from pydantic import BaseModel, Field


class DiscoveredFeed(BaseModel):
    """Structured output from the Discovery Agent."""

    url: str = Field(..., description="RSS feed URL")
    topic_tags: list[str] = Field(..., description="Topics this feed covers")
    title: str = Field(default="", description="Feed title")
