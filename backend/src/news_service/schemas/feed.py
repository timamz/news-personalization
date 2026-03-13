from pydantic import BaseModel, Field


class DiscoveredFeed(BaseModel):
    """Structured output from the Discovery Agent."""

    url: str = Field(..., description="RSS feed URL")
    title: str = Field(default="", description="Feed title")
    source_description: str = Field(default="", description="Human-readable source description")
