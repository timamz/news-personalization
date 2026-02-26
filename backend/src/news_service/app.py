from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from news_service.api.routes_debug import router as debug_router
from news_service.api.routes_health import router as health_router
from news_service.api.routes_subscriptions import router as subscriptions_router
from news_service.api.routes_users import router as users_router
from news_service.core.config import get_settings
from news_service.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    settings = get_settings()
    setup_logging(settings.log_level)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="News Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(users_router)
    app.include_router(subscriptions_router)

    if settings.log_level == "DEBUG":
        app.include_router(debug_router)

    return app


app = create_app()
