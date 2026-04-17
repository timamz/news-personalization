from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bot_token: str
    proxy_url: str | None = None
    backend_url: str = "http://app:8000"
    backend_request_timeout_seconds: float = 300.0
    backend_slow_request_timeout_seconds: float = 600.0
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8001
    webhook_public_host: str = "tgbot"
    bot_storage_path: str = "/home/appuser/bot_storage.db"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
