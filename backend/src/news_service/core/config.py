from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str = "https://api.vsellm.ru/v1"
    database_url: str = "postgresql+asyncpg://news:news@localhost:5432/news"
    redis_url: str = "redis://localhost:6379/0"

    log_level: str = "DEBUG"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"

    rss_poll_interval_minutes: int = 30
    topic_similarity_threshold: float = 0.85


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
