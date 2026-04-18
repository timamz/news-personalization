from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    llm_timeout_seconds: float = 300.0
    database_url: str = "postgresql+asyncpg://news:news@localhost:5432/news"
    redis_url: str = "redis://localhost:6379/0"
    http_timeout_seconds: float = 120.0
    reddit_fetch_timeout_seconds: float = 45.0
    reddit_fetch_attempts: int = 2
    reddit_listing_limit: int = 25
    twitter_fetch_timeout_seconds: float = 20.0
    twitter_fetch_attempts: int = 3
    twitter_listing_limit: int = 20
    twitter_fetch_retry_backoff_seconds: float = 1.0
    twitter_fetch_max_rate_limit_wait_seconds: float = 30.0
    news_item_max_age_days: int = 7

    proxy_url: str | None = None

    log_level: str = "DEBUG"

    litellm_model: str = "openai/gpt-5.4-nano"
    litellm_embedding_model: str = "openai/text-embedding-3-small"
    litellm_judge_model: str = "openai/gpt-5.4-nano"
    embedding_dimensions: int = 1536
    recent_event_match_concurrency: int = 8

    llm_retry_max_attempts: int = 3
    llm_retry_base_delay_seconds: float = 1.0

    rss_poll_interval_minutes: int = 30
    topic_similarity_threshold: float = 0.85

    source_target_count: int = 8
    content_sample_size: int = 10
    content_sample_window_days: int = 30
    content_relevance_top_k: int = 3
    content_db_candidate_threshold: float = 0.80
    source_embedding_smoothing: float = 0.9

    conversation_ttl_seconds: int = 30 * 24 * 3600
    conversation_hot_max_bytes: int = 20000

    max_concurrent_discoveries: int = 3

    llm_max_context_chars: int = 1_200_000

    reflector_drift_similarity_threshold: float = 0.3
    reflector_source_staleness_days: int = 30
    reflector_max_interval_days: int = 30
    reflector_fetch_source_items_max_limit: int = 50

    article_fetch_timeout_seconds: float = 15.0
    article_body_max_chars: int = 50_000
    article_fetch_concurrency: int = 10

    searxng_url: str = "http://searxng:8080"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
