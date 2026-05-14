import json

from pydantic import field_validator, model_validator
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
    news_item_max_age_days: int = 7

    proxy_url: str | None = None

    log_level: str = "DEBUG"

    litellm_model: str = "openai/gpt-5.4-nano"
    litellm_embedding_model: str = "openai/text-embedding-3-small"
    litellm_judge_model: str = "openai/gpt-5.4-nano"
    embedding_dimensions: int = 1536
    recent_event_match_concurrency: int = 8
    event_judge_max_revisions: int = 2
    event_reflector_interval_days: int = 3
    event_verifier_lookback_days: int = 7
    event_verifier_max_searches: int = 5
    digest_writer_max_search_calls: int = 6
    digest_writer_max_fetch_calls: int = 3
    digest_writer_max_llm_calls: int = 25

    llm_retry_max_attempts: int = 3
    llm_retry_base_delay_seconds: float = 1.0

    admin_alert_webhook_url: str | None = None
    admin_alert_throttle_seconds: int = 1800

    provider_failure_retry_countdown_seconds: int = 1800
    provider_failure_retry_max_attempts: int = 48

    rss_poll_interval_minutes: int = 30
    poll_max_concurrency: int = 5
    poll_per_source_timeout_seconds: float = 90.0
    topic_similarity_threshold: float = 0.85

    digest_lead_time_minutes: int = 10
    """Start digest generation this many minutes before the user-specified
    cron time so the rendered digest reaches the user at (not after) the
    requested moment. The scheduler matches the cron against an instant
    shifted forward by this amount; the recorded ``last_digest_scheduled_at``
    is the matched wall time, not the launch instant.
    """

    source_soft_cap: int = 10
    source_hard_cap: int = 20
    content_sample_size: int = 10
    content_sample_window_days: int = 30
    content_relevance_top_k: int = 3
    content_db_candidate_threshold: float = 0.80
    source_embedding_smoothing: float = 0.9

    conversation_ttl_seconds: int = 30 * 24 * 3600
    conversation_hot_max_bytes: int = 20000
    conversation_hot_max_tokens: int = 30_000

    max_active_subscriptions_per_user: int = 5

    max_user_message_chars: int = 10_000
    max_llm_external_text_chars: int = 50_000

    rate_limit_conversation_per_hour: int = 120
    rate_limit_discovery_per_day: int = 20
    rate_limit_digest_now_per_day: int = 60

    tool_call_budget_conversational: int = 50
    tool_call_budget_discovery_pipeline: int = 200
    tool_call_budget_finder: int = 30
    tool_call_budget_reflector: int = 30
    tool_call_budget_verifier: int = 30

    ssrf_block_private_ips: bool = True
    ssrf_allowed_schemes: tuple[str, ...] = ("http", "https")
    ssrf_max_redirects: int = 5

    injection_classifier_enabled: bool = False
    """When True, an extra ML classifier scan runs alongside the regex layer.

    Disabled by default because it requires the model weights to be
    downloaded once at first call (~300 MB) and an extra ~150 ms per
    scan. Flip to True after running ``huggingface-cli login`` and
    accepting Meta's Llama Prompt Guard license once on the host.
    """
    injection_classifier_model: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    injection_classifier_threshold: float = 0.5

    output_safety_classifier_enabled: bool = False
    """When True, runs a multilingual DistilBERT toxicity classifier.

    Off by default: one-time weight download (~540 MB), extra torch
    dep, and ~150 ms per call on CPU. Covers Russian, English, and the
    other 100+ languages DistilBERT-multilingual was pre-trained on
    via Wikipedia. Flip to True after ``uv sync --extra classifier``
    and the first warm-up download.
    """
    output_safety_classifier_model: str = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    output_safety_classifier_threshold: float = 0.5

    max_concurrent_discoveries: int = 3
    max_concurrent_web_searches: int = 2
    source_validation_timeout_seconds: float = 30.0
    discovery_removal_lockout_days: int = 30

    llm_max_context_chars: int = 1_200_000

    reflector_drift_similarity_threshold: float = 0.3
    reflector_source_staleness_days: int = 30
    reflector_contribution_streak_threshold: int = 10
    reflector_fetch_source_items_max_limit: int = 50

    article_fetch_timeout_seconds: float = 15.0
    article_body_max_chars: int = 50_000
    article_fetch_concurrency: int = 10

    yandex_search_api_key: str
    yandex_search_type: str = "COM"

    llm_model_pricing_usd_per_1m: dict[str, dict[str, float]] = {}
    yandex_search_price_usd_per_call: float = 0.005
    """USD cost attributed to each Yandex Search API dispatch.

    Official rate is 480 RUB per 1000 daytime requests (360 RUB at night).
    At ~95 RUB/USD that is ~$0.00505 daytime / ~$0.00379 nighttime per call.
    We stamp the daytime rate for accounting simplicity; override via
    ``YANDEX_SEARCH_PRICE_USD_PER_CALL`` env when running off-peak.
    """

    @field_validator("llm_model_pricing_usd_per_1m", mode="before")
    def _parse_json_pricing(value: object) -> object:
        """Accept the pricing table either as a dict or a JSON string.

        pydantic-settings loads env-var strings verbatim; when the setting
        is configured via .env as a JSON blob, we parse it here so the
        downstream validator sees a real dict.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return {}
            return json.loads(stripped)
        return value

    @model_validator(mode="after")
    def _require_pricing_for_configured_models(self) -> "Settings":
        """Fail fast when a configured model has no pricing entry.

        Unit-economics accounting is mandatory: every model this service
        can dispatch to must have input/output prices declared, otherwise
        a call would silently contribute $0 to the ledger. Raising at
        startup surfaces the misconfiguration before the first LLM call.
        """
        required = {self.litellm_model, self.litellm_judge_model, self.litellm_embedding_model}
        missing = sorted(m for m in required if m not in self.llm_model_pricing_usd_per_1m)
        if missing:
            raise ValueError(
                "LLM_MODEL_PRICING_USD_PER_1M is missing entries for configured models: "
                f"{missing}. Configure every model with its input/output price per 1M tokens."
            )
        for name, entry in self.llm_model_pricing_usd_per_1m.items():
            for key in ("input", "output"):
                if key not in entry:
                    raise ValueError(
                        f"LLM_MODEL_PRICING_USD_PER_1M['{name}'] is missing '{key}' price."
                    )
                if not isinstance(entry[key], int | float) or entry[key] < 0:
                    raise ValueError(
                        f"LLM_MODEL_PRICING_USD_PER_1M['{name}']['{key}'] must be a "
                        f"non-negative number, got {entry[key]!r}."
                    )
        return self


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
