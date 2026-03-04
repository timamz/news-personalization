import pytest
from pydantic import ValidationError

from news_service.schemas.subscription import (
    EventConstraint,
    SubscriptionConfig,
    SubscriptionCreate,
    SubscriptionUpdate,
)


def test_subscription_create_valid():
    payload = SubscriptionCreate(prompt="I want AI news every morning")
    assert payload.prompt == "I want AI news every morning"
    assert payload.delivery_mode is None


def test_subscription_create_too_short():
    with pytest.raises(ValidationError):
        SubscriptionCreate(prompt="hi")


def test_subscription_config_valid():
    config = SubscriptionConfig(
        topics=["AI"],
        delivery_mode="digest",
        event_matching_mode="basic",
        schedule_cron="0 8 * * *",
        schedule_was_explicit=True,
        format_instructions="detailed analysis",
        digest_language="en",
    )
    assert config.topics == ["AI"]
    assert config.delivery_mode == "digest"
    assert config.schedule_cron == "0 8 * * *"


def test_subscription_config_empty_topics():
    with pytest.raises(ValidationError):
        SubscriptionConfig(
            topics=[],
            delivery_mode="digest",
            event_matching_mode="basic",
            schedule_cron="0 8 * * *",
            schedule_was_explicit=True,
            digest_language="en",
        )


def test_subscription_config_default_format():
    config = SubscriptionConfig(
        topics=["politics"],
        delivery_mode="digest",
        event_matching_mode="basic",
        schedule_cron="0 21 * * *",
        schedule_was_explicit=True,
        digest_language="en",
    )
    assert config.format_instructions == "brief summary"


def test_subscription_config_supports_manual_mode():
    config = SubscriptionConfig(
        topics=["politics"],
        schedule_cron=None,
        schedule_was_explicit=False,
        digest_language="en",
    )
    assert config.delivery_mode == "digest"
    assert config.schedule_cron is None
    assert config.event_matching_mode == "basic"
    assert config.event_constraints == []


def test_subscription_update_accepts_partial_fields():
    payload = SubscriptionUpdate(
        schedule_cron="0 9 * * 1-5",
        format_instructions="detailed analysis",
    )
    assert payload.schedule_cron == "0 9 * * 1-5"
    assert payload.format_instructions == "detailed analysis"


def test_subscription_update_rejects_empty_format_string():
    with pytest.raises(ValidationError):
        SubscriptionUpdate(format_instructions="")


def test_strict_event_subscription_requires_constraints():
    with pytest.raises(ValidationError):
        SubscriptionConfig(
            topics=["lectures"],
            delivery_mode="event",
            event_matching_mode="strict_with_prefilter",
            schedule_cron=None,
            schedule_was_explicit=False,
            digest_language="ru",
        )


def test_event_constraint_validates_value_type_and_match_mode():
    constraint = EventConstraint(
        key="speaker_must_be_drobyshevsky",
        description="Primary speaker must be Stanislav Drobyshevsky",
        value_type="string",
        match_mode="exact",
        required_string="станислав владимирович дробышевский",
        prefilter_terms=["станислав", "дробышевский"],
    )
    assert constraint.required_string == "станислав владимирович дробышевский"


def test_event_constraint_rejects_invalid_match_mode():
    with pytest.raises(ValidationError):
        EventConstraint(
            key="speaker_must_be_drobyshevsky",
            description="Primary speaker must be Stanislav Drobyshevsky",
            value_type="boolean",
            match_mode="contains",
            required_boolean=True,
        )
