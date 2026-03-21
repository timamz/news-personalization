from agents import OpenAIChatCompletionsModel, set_default_openai_client
from openai import AsyncOpenAI

from news_service.core.config import get_settings

settings = get_settings()

openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    timeout=settings.llm_timeout_seconds,
)

set_default_openai_client(openai_client)

agents_model = OpenAIChatCompletionsModel(
    model=settings.llm_model,
    openai_client=openai_client,
)
