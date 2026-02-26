import asyncio
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from news_service.app import create_app
from news_service.core.config import get_settings
from news_service.db.session import engine
from news_service.models import Base


@pytest_asyncio.fixture(scope="session", loop_scope="session")
def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


async def _ensure_database_exists() -> None:
    settings = get_settings()
    database_url = make_url(settings.database_url)
    database_name = database_url.database
    if database_name is None or not database_name.endswith("_test"):
        raise RuntimeError(
            "Integration tests require DATABASE_URL to point to a *_test database, "
            f"got: {settings.database_url}"
        )

    last_error: Exception | None = None

    target_engine = create_async_engine(database_url.render_as_string(hide_password=False))
    try:
        async with target_engine.connect() as conn:
            await conn.scalar(text("SELECT 1"))
            return
    except Exception as exc:
        last_error = exc
    finally:
        await target_engine.dispose()

    raise RuntimeError(
        "Failed to connect to integration DATABASE_URL. Ensure the *_test database exists "
        f"and credentials are correct. URL: {settings.database_url}. Error: {last_error}"
    )


@pytest_asyncio.fixture(scope="session", autouse=True, loop_scope="session")
async def enforce_test_database() -> AsyncGenerator[None]:
    await _ensure_database_exists()
    yield


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def reset_database(enforce_test_database) -> AsyncGenerator[None]:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture(loop_scope="session")
async def api_client() -> AsyncGenerator[AsyncClient]:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
