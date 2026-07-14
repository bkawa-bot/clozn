"""Small HTTP safety policy shared by JSON and streaming responses.

The product binds loopback by default, but a browser on an unrelated website can still
attempt requests to localhost.  CORS is therefore loopback-only unless the operator opts
additional exact origins in through ``CLOZN_ORIGINS``.  Request bodies are bounded before
they are read so a malformed or hostile client cannot make the gateway allocate without
limit.
"""
from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit


DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_ALLOW_HEADERS = (
    "Accept, Authorization, Content-Type, OpenAI-Organization, OpenAI-Project, "
    "X-Requested-With"
)


def max_request_bytes() -> int:
    """Configured request-body ceiling, falling back safely on invalid input."""
    raw = os.environ.get("CLOZN_MAX_REQUEST_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES
    return value if value > 0 else DEFAULT_MAX_REQUEST_BYTES


def _configured_origins() -> set[str]:
    return {
        value.strip().rstrip("/")
        for value in os.environ.get("CLOZN_ORIGINS", "").split(",")
        if value.strip()
    }


def origin_allowed(origin: str | None) -> bool:
    """Allow loopback browser origins plus explicit exact operator opt-ins."""
    if not origin:
        return True
    if any(ord(char) < 0x20 or ord(char) == 0x7f for char in origin):
        return False
    origin = origin.strip().rstrip("/")
    configured = _configured_origins()
    if "*" in configured or origin in configured:
        return True
    try:
        parsed = urlsplit(origin)
        parsed.port  # validate a present port rather than accepting malformed values
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            return False
        host = parsed.hostname.lower()
        if host == "localhost" or host.endswith(".localhost"):
            return True
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def request_origin(handler) -> str | None:
    headers = getattr(handler, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    value = getter("Origin")
    return str(value) if value else None


def send_cors_headers(handler) -> bool:
    """Write CORS response headers when this request's origin is allowed."""
    origin = request_origin(handler)
    if not origin or not origin_allowed(origin):
        return not origin
    handler.send_header("Access-Control-Allow-Origin", origin)
    handler.send_header("Vary", "Origin")
    handler.send_header("Access-Control-Expose-Headers", "X-Clozn-Run-Id")
    return True
