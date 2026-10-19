"""Loopback enforcement for outbound model traffic.

This is the single choke point that keeps document text on this machine. It is
checked both at configuration load and again immediately before every request.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from ..errors import ModelUnavailableError

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def host_is_local(hostname: str) -> bool:
    if not hostname:
        return False
    if hostname in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    # A hostname we cannot parse as an IP is only acceptable if it resolves to
    # loopback. Anything public is refused.
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return False
    return all(
        ipaddress.ip_address(info[4][0]).is_loopback for info in infos if info[4]
    )


def assert_local_target(base_url: str, *, allow_non_loopback: bool) -> None:
    """Refuse to send content anywhere that is not this machine."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ModelUnavailableError(
            base_url, reason=f"unsupported scheme {parsed.scheme!r}"
        )
    hostname = parsed.hostname or ""
    if not hostname:
        raise ModelUnavailableError(base_url, reason="no host in URL")
    if allow_non_loopback:
        return
    if not host_is_local(hostname):
        raise ModelUnavailableError(
            base_url,
            reason=(
                f"host {hostname!r} is not a loopback address and "
                "LLM_ALLOW_NON_LOOPBACK is false"
            ),
        )
