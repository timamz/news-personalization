"""
BenchmarkConfig collects every tunable the harness needs at startup.

Values cascade: explicit constructor argument > environment variable >
field default. Pydantic-settings handles the env loading.

Example usage:

    from news_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig()
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkConfig(BaseSettings):
    """Harness-wide configuration resolved from environment."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Devbox Postgres -- throwaway benchmark DBs are created and dropped here.
    benchmark_pg_host: str = "100.73.138.67"
    benchmark_pg_port: int = 5432
    benchmark_pg_admin_user: str = "news"
    benchmark_pg_admin_password: str = "news"
    benchmark_pg_admin_db: str = "news"

    # Devbox Redis. Keys are additionally prefixed per run.
    benchmark_redis_url: str = "redis://100.73.138.67:6379/5"

    # Primary model under test -- read from the backend's settings by default.
    litellm_model: str = "openai/gpt-5.4-nano"
    litellm_embedding_model: str = "openai/text-embedding-3-small"
    litellm_judge_model: str = "openai/gpt-5.4-nano"

    # Keep the throwaway DB behind on failure so humans can post-mortem.
    keep_db_on_failure: bool = False

    def bench_db_url(self, run_id: str) -> str:
        """Return the async SQLAlchemy URL for this run's throwaway database."""
        return (
            f"postgresql+asyncpg://{self.benchmark_pg_admin_user}:"
            f"{self.benchmark_pg_admin_password}@{self.benchmark_pg_host}:"
            f"{self.benchmark_pg_port}/news_bench_{run_id}"
        )

    def admin_db_url(self) -> str:
        """Return the admin database URL used to CREATE/DROP per-run databases."""
        return (
            f"postgresql+asyncpg://{self.benchmark_pg_admin_user}:"
            f"{self.benchmark_pg_admin_password}@{self.benchmark_pg_host}:"
            f"{self.benchmark_pg_port}/{self.benchmark_pg_admin_db}"
        )
