import pytest
from pydantic import ValidationError

from news_service.schemas.subscription import SubscriptionConfig, SubscriptionCreate


def test_subscription_create_valid():
    payload = SubscriptionCreate(prompt="I want AI news every morning")
    assert payload.prompt == "I want AI news every morning"


def test_subscription_create_too_short():
    with pytest.raises(ValidationError):
        SubscriptionCreate(prompt="hi")


def test_subscription_config_valid():
    config = SubscriptionConfig(
        topics=["AI"],
        schedule_cron="0 8 * * *",
        format_instructions="detailed analysis",
    )
    assert config.topics == ["AI"]
    assert config.schedule_cron == "0 8 * * *"


def test_subscription_config_empty_topics():
    with pytest.raises(ValidationError):
        SubscriptionConfig(
            topics=[],
            schedule_cron="0 8 * * *",
        )


def test_subscription_config_default_format():
    config = SubscriptionConfig(
        topics=["politics"],
        schedule_cron="0 21 * * *",
    )
    assert config.format_instructions == "brief summary"
