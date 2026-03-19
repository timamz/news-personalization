import pytest
from pydantic import ValidationError

from news_service.schemas.subscription import (
    SubscriptionCreate,
    SubscriptionUpdate,
)


def test_subscription_create_valid():
    payload = SubscriptionCreate(
        prompt="I want AI news every morning",
        digest_language_override="ru",
    )
    assert payload.prompt == "I want AI news every morning"
    assert payload.delivery_mode is None
    assert payload.digest_language_override == "ru"


def test_subscription_create_too_short():
    with pytest.raises(ValidationError):
        SubscriptionCreate(prompt="hi")


def test_subscription_create_optional_fields_default_to_none():
    payload = SubscriptionCreate(prompt="I want AI news every morning")
    assert payload.prompt_summary is None
    assert payload.short_label is None
    assert payload.format_instructions is None


def test_subscription_create_accepts_all_optional_fields():
    payload = SubscriptionCreate(
        prompt="I want AI news every morning",
        prompt_summary="AI news every morning",
        short_label="AI News",
        format_instructions="detailed analysis",
        delivery_mode="event",
    )
    assert payload.prompt_summary == "AI news every morning"
    assert payload.short_label == "AI News"
    assert payload.format_instructions == "detailed analysis"


def test_subscription_update_accepts_partial_fields():
    payload = SubscriptionUpdate(
        schedule_cron="0 9 * * 1-5",
        format_instructions="detailed analysis",
        digest_language="ru",
    )
    assert payload.schedule_cron == "0 9 * * 1-5"
    assert payload.format_instructions == "detailed analysis"
    assert payload.digest_language == "ru"


def test_subscription_update_rejects_empty_format_string():
    with pytest.raises(ValidationError):
        SubscriptionUpdate(format_instructions="")
