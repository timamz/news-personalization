from celery import Celery

from news_service.core.config import get_settings

settings = get_settings()

celery_app = Celery("news_service", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "poll-all-feeds": {
            "task": "news_service.tasks.poll_feeds.poll_all_feeds",
            "schedule": settings.rss_poll_interval_minutes * 60,
        },
    },
)

import news_service.tasks.deliver_digest  # noqa: E402, F401
import news_service.tasks.poll_feeds  # noqa: E402, F401
