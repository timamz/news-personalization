import pytest

from news_service.services.delivery import LogChannel


@pytest.mark.asyncio
async def test_log_channel_does_not_raise():
    channel = LogChannel()
    await channel.send("Test Subject", "Test body content")
