from httpx import AsyncClient


async def create_user(api_client: AsyncClient, *, timezone: str | None = None) -> dict[str, object]:
    response = await api_client.post("/users")
    assert response.status_code == 201
    payload = response.json()
    if timezone is not None:
        update_response = await api_client.patch(
            "/users/me",
            headers={"X-API-Key": payload["api_key"]},
            json={"timezone": timezone},
        )
        assert update_response.status_code == 200
        payload = update_response.json()
    return payload
