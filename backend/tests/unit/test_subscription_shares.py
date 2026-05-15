"""Tests for share_subscription / import_shared_subscription conversational tools.

Covers the share-token plumbing in ``core.subscription_shares`` together
with the two new tools that mint and redeem the tokens. The fake Redis
client is shared between both tools by patching the module-level
``get_redis_client`` symbol; tools talk to it via the indirection in
``core/subscription_shares.py``.
"""

import logging
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.conversational import create_conversational_agent

logging.disable(logging.CRITICAL)

SHARES_MODULE = "news_service.core.subscription_shares"


def _fake_redis(initial: dict[str, str] | None = None) -> MagicMock:
    """Minimal in-memory fake of the async Redis client used by shares."""
    storage = dict(initial or {})

    async def mock_set(key, value, ex=None, nx=False):
        del ex
        if nx and key in storage:
            return None
        storage[key] = value
        return True

    async def mock_getdel(key):
        return storage.pop(key, None)

    async def mock_get(key):
        return storage.get(key)

    async def mock_delete(key):
        storage.pop(key, None)

    fake = MagicMock()
    fake.set = AsyncMock(side_effect=mock_set)
    fake.getdel = AsyncMock(side_effect=mock_getdel)
    fake.get = AsyncMock(side_effect=mock_get)
    fake.delete = AsyncMock(side_effect=mock_delete)
    fake._storage = storage
    return fake


def _fake_user(*, has_onboarded: bool = True, user_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=user_id or uuid.uuid4(),
        timezone="Europe/Moscow",
        language="en",
        delivery_webhook_url=f"https://importer-{uuid.uuid4().hex[:6]}.test/deliver",
        conversation_summary="",
        has_onboarded=has_onboarded,
    )


class _SessionFactory:
    def __init__(self, session: Any) -> None:
        self._session = session

    def __call__(self) -> Any:
        mgr = AsyncMock()
        mgr.__aenter__ = AsyncMock(return_value=self._session)
        mgr.__aexit__ = AsyncMock(return_value=None)
        return mgr


def _db_session_returning(*results: Any) -> AsyncMock:
    """Build a top-level db_session whose ``execute`` returns each result in turn."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=list(results))
    return session


def _scalar_one_or_none(value: Any) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _scalar_one(value: Any) -> MagicMock:
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _scalar_iter(values: list[Any]) -> MagicMock:
    """Result whose ``.scalars().all()`` returns the given list."""
    scalars = MagicMock()
    scalars.all.return_value = list(values)
    result = MagicMock()
    result.scalars.return_value = scalars
    return result


def _get_tool(agent: Any, name: str):
    return next(t for t in agent.tools if callable(t) and t.__name__ == name)


def _build_agent(
    *,
    user: SimpleNamespace,
    db_session: Any,
    scoped: Any,
) -> tuple[Any, dict[str, Any]]:
    return create_conversational_agent(
        db_session=db_session,
        user=user,
        conversation_summary="",
        session_factory=_SessionFactory(scoped),
    )


# ---------- core.subscription_shares ----------


@pytest.mark.asyncio
async def test_create_share_persists_payload_under_share_sub_key_prefix(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    from news_service.core.subscription_shares import create as create_share

    owner = str(uuid.uuid4())
    subscription_id = str(uuid.uuid4())
    token = await create_share(owner_user_id=owner, subscription_id=subscription_id, ttl_seconds=60)

    stored_keys = [k for k in fake._storage if k.startswith("share:sub:")]
    assert (
        len(stored_keys) == 1
        and stored_keys[0] == f"share:sub:{token}"
        and owner in fake._storage[stored_keys[0]]
        and subscription_id in fake._storage[stored_keys[0]]
    ), "share token was not stored under the share:sub:<token> key with the right payload"


@pytest.mark.asyncio
async def test_consume_share_deletes_record_so_second_redeem_returns_none(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    from news_service.core.subscription_shares import (
        consume as consume_share,
    )
    from news_service.core.subscription_shares import (
        create as create_share,
    )

    token = await create_share(
        owner_user_id=str(uuid.uuid4()),
        subscription_id=str(uuid.uuid4()),
        ttl_seconds=60,
    )
    first = await consume_share(token)
    second = await consume_share(token)
    assert first is not None and second is None, (
        "consume must be one-shot; a second redeem of the same token must not succeed"
    )


@pytest.mark.asyncio
async def test_consume_share_returns_none_for_unknown_token(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    from news_service.core.subscription_shares import consume as consume_share

    assert await consume_share("does-not-exist") is None, (
        "consume must return None when the token has never been stored"
    )


# ---------- share_subscription tool ----------


@pytest.mark.asyncio
async def test_share_subscription_mints_a_token_for_the_owners_active_subscription(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.is_active = True

    db_session = _db_session_returning(_scalar_one_or_none(sub))
    agent, _ = _build_agent(user=user, db_session=db_session, scoped=AsyncMock())

    result = await _get_tool(agent, "share_subscription")(str(sub.id))

    stored = [k for k in fake._storage if k.startswith("share:sub:")]
    assert (
        "SHARE_TOKEN=" in result
        and len(stored) == 1
        and stored[0].split("share:sub:", 1)[1] in result
    ), "share_subscription did not surface the freshly minted token verbatim"


@pytest.mark.asyncio
async def test_share_subscription_refuses_when_subscription_belongs_to_another_user(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    user = _fake_user()
    db_session = _db_session_returning(_scalar_one_or_none(None))
    agent, _ = _build_agent(user=user, db_session=db_session, scoped=AsyncMock())

    result = await _get_tool(agent, "share_subscription")(str(uuid.uuid4()))
    assert "not found" in result and not fake._storage, (
        "sharing a subscription owned by someone else must refuse and must not mint a token"
    )


@pytest.mark.asyncio
async def test_share_subscription_refuses_soft_deleted_subscription(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    user = _fake_user()
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = user.id
    sub.is_active = False

    db_session = _db_session_returning(_scalar_one_or_none(sub))
    agent, _ = _build_agent(user=user, db_session=db_session, scoped=AsyncMock())

    result = await _get_tool(agent, "share_subscription")(str(sub.id))
    assert "cannot share a deleted subscription" in result and not fake._storage, (
        "sharing a soft-deleted subscription must refuse and must not mint a token"
    )


# ---------- import_shared_subscription tool ----------


def _source_subscription(*, user_id: uuid.UUID) -> MagicMock:
    source_sub = MagicMock()
    source_sub.id = uuid.uuid4()
    source_sub.user_id = user_id
    source_sub.is_active = True
    source_sub.topic_embedding = [0.1] * 4
    source_sub.user_spec = f"shared spec {uuid.uuid4().hex[:6]}"
    source_sub.delivery_mode = "digest"
    source_sub.schedule_cron = "0 8 * * *"
    source_sub.digest_language = "ru"
    source_sub.delivery_webhook_url = f"https://owner-{uuid.uuid4().hex[:4]}.test/deliver"
    return source_sub


def _capture_scoped(
    *, source_sub: MagicMock | None, source_links: list[Any]
) -> tuple[AsyncMock, list[Any]]:
    captured: list[Any] = []
    scoped = AsyncMock()
    scoped.add = MagicMock(side_effect=captured.append)
    scoped.flush = AsyncMock()
    scoped.commit = AsyncMock()
    scoped.rollback = AsyncMock()
    scoped.execute = AsyncMock(
        side_effect=[
            _scalar_one_or_none(source_sub),
            _scalar_iter(source_links),
        ]
    )
    return scoped, captured


@pytest.mark.asyncio
async def test_import_shared_subscription_copies_spec_sources_and_schedule(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    owner_id = uuid.uuid4()
    importer = _fake_user()
    source_sub = _source_subscription(user_id=owner_id)

    link_a = MagicMock()
    link_a.source_id = uuid.uuid4()
    link_a.is_user_specified = True
    link_b = MagicMock()
    link_b.source_id = uuid.uuid4()
    link_b.is_user_specified = False

    from news_service.core.subscription_shares import create as create_share

    token = await create_share(
        owner_user_id=str(owner_id),
        subscription_id=str(source_sub.id),
        ttl_seconds=60,
    )

    db_session = _db_session_returning(_scalar_one(0))
    scoped, captured = _capture_scoped(source_sub=source_sub, source_links=[link_a, link_b])
    agent, shared_state = _build_agent(user=importer, db_session=db_session, scoped=scoped)

    result = await _get_tool(agent, "import_shared_subscription")(token)

    new_sub = next(
        obj for obj in captured if getattr(obj, "__class__", None).__name__ == "Subscription"
    )
    new_links = [
        obj for obj in captured if getattr(obj, "__class__", None).__name__ == "SubscriptionSource"
    ]
    copied_ids = {link.source_id for link in new_links}
    assert (
        "shared_subscription_imported" in result
        and new_sub.user_id == importer.id
        and new_sub.user_spec == source_sub.user_spec
        and new_sub.schedule_cron == source_sub.schedule_cron
        and new_sub.digest_language == source_sub.digest_language
        and new_sub.delivery_mode == source_sub.delivery_mode
        and new_sub.topic_embedding == source_sub.topic_embedding
        and new_sub.delivery_webhook_url == importer.delivery_webhook_url
        and copied_ids == {link_a.source_id, link_b.source_id}
        and shared_state["created_subscription_id"] is not None
    ), (
        "imported subscription did not faithfully clone the owner's "
        "spec/sources/schedule onto a row owned by the importer with the "
        "importer's webhook"
    )


@pytest.mark.asyncio
async def test_import_shared_subscription_refuses_unknown_or_expired_token(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    importer = _fake_user()
    db_session = AsyncMock()
    scoped = AsyncMock()
    agent, shared_state = _build_agent(user=importer, db_session=db_session, scoped=scoped)

    result = await _get_tool(agent, "import_shared_subscription")("never-existed")
    assert (
        "share_invalid" in result
        and db_session.execute.await_count == 0
        and scoped.commit.await_count == 0
        and shared_state["created_subscription_id"] is None
    ), "an unknown share token must short-circuit without touching the DB"


@pytest.mark.asyncio
async def test_import_shared_subscription_refuses_self_import(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    importer = _fake_user()
    from news_service.core.subscription_shares import create as create_share

    token = await create_share(
        owner_user_id=str(importer.id),
        subscription_id=str(uuid.uuid4()),
        ttl_seconds=60,
    )

    db_session = AsyncMock()
    scoped = AsyncMock()
    agent, shared_state = _build_agent(user=importer, db_session=db_session, scoped=scoped)
    result = await _get_tool(agent, "import_shared_subscription")(token)
    assert (
        "share_self_import" in result
        and db_session.execute.await_count == 0
        and scoped.commit.await_count == 0
        and shared_state["created_subscription_id"] is None
    ), "importing your own share token must be rejected without DB work"


@pytest.mark.asyncio
async def test_import_shared_subscription_refuses_when_importer_at_active_limit(mocker) -> None:
    fake = _fake_redis()
    mocker.patch(f"{SHARES_MODULE}.get_redis_client", return_value=fake)

    importer = _fake_user()
    owner_id = uuid.uuid4()
    source_sub_id = uuid.uuid4()

    from news_service.core.subscription_shares import create as create_share

    token = await create_share(
        owner_user_id=str(owner_id),
        subscription_id=str(source_sub_id),
        ttl_seconds=60,
    )

    db_session = _db_session_returning(_scalar_one(5))
    scoped = AsyncMock()
    agent, shared_state = _build_agent(user=importer, db_session=db_session, scoped=scoped)
    result = await _get_tool(agent, "import_shared_subscription")(token)
    assert (
        "subscription limit reached" in result
        and "5" in result
        and "delete_subscription" in result
        and scoped.commit.await_count == 0
        and shared_state["created_subscription_id"] is None
    ), "an importer at the active cap must be refused and no new subscription must be persisted"
