"""SSRF protection for outbound URL fetches driven by user / LLM input.

When the conversational agent or the discovery pipeline accepts a URL
that originated from external input (user message, LLM tool argument,
web-search result), the host the URL points at must not be resolvable
to a private network. Otherwise an attacker can pivot the backend into
the internal network -- AWS metadata, internal admin panels, postgres,
Redis, neighbouring containers.

This module performs the standard defence:

1. Reject schemes outside ``ssrf_allowed_schemes`` (default http/https).
2. Reject hostnames that are bare IPs in a private / loopback /
   link-local / multicast / reserved range.
3. Resolve the hostname via DNS and reject if any A/AAAA points at a
   private range.
4. When walking redirects, validate the next hop the same way before
   following it.

It is intentionally strict-by-default. Callers that need an exception
(e.g. running inside a container where outbound traffic must traverse a
deliberately-private proxy) should pass ``allow_private=True`` rather
than disabling the validator wholesale.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)


class UnsafeUrlError(ValueError):
    """The URL targets a network range the application is forbidden to reach."""


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"DNS resolution failed for {host!r}: {exc}") from exc
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            out.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    return out


def assert_safe_url(url: str, *, allow_private: bool = False) -> None:
    """Raise ``UnsafeUrlError`` if the URL is unsafe for outbound fetch.

    Set ``allow_private=True`` only for trusted callers that legitimately
    target a private range (e.g. the internal Telegram/Reddit adapters
    routed through a known proxy).
    """
    settings = get_settings()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in settings.ssrf_allowed_schemes:
        raise UnsafeUrlError(
            f"scheme {parsed.scheme!r} is not in allowed set {settings.ssrf_allowed_schemes}"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError(f"URL has no hostname: {url!r}")

    if not settings.ssrf_block_private_ips or allow_private:
        return

    try:
        as_ip = ipaddress.ip_address(host)
    except ValueError:
        as_ip = None

    if as_ip is not None:
        if _is_disallowed_ip(as_ip):
            raise UnsafeUrlError(f"URL targets disallowed IP literal {as_ip}")
        return

    for ip in _resolve(host):
        if _is_disallowed_ip(ip):
            raise UnsafeUrlError(
                f"hostname {host!r} resolves to disallowed IP {ip}; refusing to fetch"
            )


async def safe_get(
    url: str,
    *,
    timeout: float,
    proxy: str | None = None,
    max_redirects: int | None = None,
) -> httpx.Response:
    """Fetch ``url`` while validating every hop against SSRF rules.

    Disables httpx auto-redirects so each hop is checked. Refuses to
    follow a redirect to a private host even if the initial URL was
    public.
    """
    settings = get_settings()
    cap = max_redirects if max_redirects is not None else settings.ssrf_max_redirects
    current = url
    client_kwargs: dict[str, object] = {"timeout": timeout, "follow_redirects": False}
    if proxy:
        client_kwargs["proxy"] = proxy
    async with httpx.AsyncClient(**client_kwargs) as client:  # type: ignore[arg-type]
        for _ in range(cap + 1):
            assert_safe_url(current)
            response = await client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    return response
                current = str(httpx.URL(current).join(location))
                continue
            return response
        raise UnsafeUrlError(f"redirect chain exceeded {cap} hops starting at {url!r}")
