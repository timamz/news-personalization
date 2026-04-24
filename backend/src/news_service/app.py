from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from news_service.api.routes_conversations import router as conversations_router
from news_service.api.routes_debug import router as debug_router
from news_service.api.routes_health import router as health_router
from news_service.api.routes_users import router as users_router
from news_service.core.config import get_settings
from news_service.core.llm_usage import current_run_id, install_usage_callback
from news_service.core.logging import setup_logging
from news_service.core.redis import close_redis_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    settings = get_settings()
    setup_logging(settings.log_level)
    install_usage_callback()
    try:
        yield
    finally:
        await close_redis_client()


async def _run_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Pull the X-Run-Id header into the request's ContextVar scope.

    The header is only set by the benchmark / economics harness to correlate
    LLM usage rows with a specific benchmark run. Normal production clients
    omit it and the ledger row simply carries a NULL run_id.
    """
    run_id = request.headers.get("x-run-id") or request.headers.get("X-Run-Id")
    token = current_run_id.set(run_id or None)
    try:
        return await call_next(request)
    finally:
        current_run_id.reset(token)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="News Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.middleware("http")(_run_id_middleware)

    app.include_router(health_router)
    app.include_router(users_router)
    app.include_router(conversations_router)

    if settings.log_level == "DEBUG":
        app.include_router(debug_router)

    return app


app = create_app()
