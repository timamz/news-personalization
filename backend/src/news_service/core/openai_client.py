from openai import AsyncOpenAI

from news_service.core.config import get_settings

settings = get_settings()

openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=settings.llm_timeout_seconds,
)
