import json

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


async def create_subscription_via_stream(
    api_client: AsyncClient,
    api_key: str,
    payload: dict,
) -> dict:
    """Create a subscription via the streaming endpoint and return the subscription dict."""
    response = await api_client.post(
        "/subscriptions/stream",
        headers={"X-API-Key": api_key},
        json=payload,
    )
    assert response.status_code == 200
    result: dict | None = None
    for line in response.text.strip().splitlines():
        event = json.loads(line)
        if event.get("event") == "done":
            result = event["subscription"]
        elif event.get("event") == "error":
            raise AssertionError(f"Subscription stream error: {event.get('detail')}")
    assert result is not None, "Subscription stream produced no 'done' event"
    return result


def parse_ndjson_done_event(response_text: str) -> dict | None:
    """Extract the 'done' event payload from an NDJSON streaming response."""
    for line in response_text.strip().splitlines():
        event = json.loads(line)
        if event.get("event") == "done":
            return event
    return None
