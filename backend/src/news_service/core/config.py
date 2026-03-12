from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str = "https://api.vsellm.ru/v1"
    llm_timeout_seconds: float = 300.0
    database_url: str = "postgresql+asyncpg://news:news@localhost:5432/news"
    redis_url: str = "redis://localhost:6379/0"
    http_timeout_seconds: float = 120.0
    reddit_fetch_timeout_seconds: float = 45.0
    reddit_fetch_attempts: int = 2
    reddit_listing_limit: int = 25

    log_level: str = "DEBUG"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-5-mini"
    recent_event_match_concurrency: int = 8

    rss_poll_interval_minutes: int = 30
    topic_similarity_threshold: float = 0.85


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
