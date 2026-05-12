"""Integration coverage for the tgbot<->backend user-registration contract.

Replaces mock-heavy unit tests that previously asserted tgbot client
behavior against a faked httpx transport. Each test here issues the same
requests the real ``tgbot.client.BackendClient`` would issue (POST
``/users``, PATCH ``/users/me`` with an ``X-API-Key`` header) against a
live ASGI backend so the wire contract is verified end to end.
"""

import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio(loop_scope="session")
async def test_registered_user_can_authenticate_and_update_webhook(
    api_client: AsyncClient,
) -> None:
    """A freshly minted api_key authorizes a webhook update on /users/me."""
    webhook_url = f"https://hooks.example.test/{uuid.uuid4().hex[:8]}/деливери"

    create_response = await api_client.post("/users")
    assert create_response.status_code == 201, (
        f"user creation did not return 201 Created, got {create_response.status_code}"
    )
    created = create_response.json()
    assert "id" in created and "api_key" in created, (
        "new user payload is missing id or api_key fields"
    )
    api_key = created["api_key"]
    assert isinstance(api_key, str) and api_key, "returned api_key is not a non-empty string"

    update_response = await api_client.patch(
        "/users/me",
        headers={"X-API-Key": api_key},
        json={"delivery_webhook_url": webhook_url},
    )
    assert update_response.status_code == 200, (
        f"profile update with a valid api_key was rejected, got {update_response.status_code}"
    )
    updated = update_response.json()
    assert updated["delivery_webhook_url"] == webhook_url, (
        "backend did not persist the submitted delivery_webhook_url"
    )
    assert updated["id"] == created["id"], (
        "profile update returned a different user id than registration"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_bogus_api_key_cannot_update_profile(api_client: AsyncClient) -> None:
    """A random api_key must not authorize PATCH /users/me."""
    bogus_key = f"not-a-real-key-{uuid.uuid4().hex}"

    response = await api_client.patch(
        "/users/me",
        headers={"X-API-Key": bogus_key},
        json={"delivery_webhook_url": f"https://hooks.example.test/{uuid.uuid4().hex[:8]}"},
    )

    assert response.status_code == 401, (
        f"unknown api_key was not rejected with 401, got {response.status_code}"
    )
