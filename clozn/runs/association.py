"""Opaque client/session/project association for the local run side-channel.

Third-party clients often discard response extensions and streaming headers are committed before a run
exists.  Clozn therefore records a privacy-preserving session key beside each run and lets a sidecar look
up the newest matching record.  Raw OpenAI ``user`` values and Clozn association headers are never
stored: only stable SHA-256-derived opaque keys are journaled.
"""
from __future__ import annotations

import hashlib
import hmac
import re


SESSION_PREFIX = "session_"
CLIENT_PREFIX = "client_"
PROJECT_PREFIX = "project_"
_SESSION_KEY_RE = re.compile(r"^session_[0-9a-f]{24}$")
_CLIENT_KEY_RE = re.compile(r"^client_[0-9a-f]{24}$")
_PROJECT_KEY_RE = re.compile(r"^project_[0-9a-f]{24}$")


class AssociationValueError(ValueError):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def validate_selector(value, field: str) -> str:
    text = str(value)
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError:
        raise AssociationValueError(field, f"{field} must contain 1-128 visible ASCII characters") from None
    if not (1 <= len(encoded) <= 128) or any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise AssociationValueError(field, f"{field} must contain 1-128 visible ASCII characters")
    return text


def validate_request_headers(headers) -> None:
    for header in ("X-Clozn-Client-Id", "X-Clozn-Session-Id", "X-Clozn-Project-Id"):
        try:
            value = headers.get(header) if headers is not None else None
        except Exception:
            value = None
        if value is not None:
            validate_selector(value, header)


def _digest(prefix: str, text: str) -> str:
    from .store import association_secret
    return prefix + hmac.new(association_secret(), text.encode("utf-8", "surrogatepass"),
                             hashlib.sha256).hexdigest()[:24]


def session_key(value, *, accept_key: bool = True) -> str | None:
    """Normalize a caller-known session identifier to the opaque journal key."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if accept_key and _SESSION_KEY_RE.fullmatch(text.lower()):
        return text.lower()
    return _digest(SESSION_PREFIX, text)


def request_session(headers, explicit=None) -> str | None:
    """Resolve explicit API identity first, then the opt-in cross-protocol session header."""
    value = explicit
    if value is None and headers is not None:
        try:
            value = headers.get("X-Clozn-Session-Id")
        except Exception:
            value = None
    return session_key(value, accept_key=False)


def client_key(value, *, accept_key: bool = True) -> str | None:
    """Opaque key for an explicit client id or raw User-Agent fingerprint."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if accept_key and _CLIENT_KEY_RE.fullmatch(text.lower()):
        return text.lower()
    return _digest(CLIENT_PREFIX, text)


def request_client(headers) -> tuple[str | None, str | None]:
    """Return ``(opaque_key, source)``; explicit header wins over coarse User-Agent fallback."""
    try:
        explicit = headers.get("X-Clozn-Client-Id") if headers is not None else None
        user_agent = headers.get("User-Agent") if headers is not None else None
    except Exception:
        explicit = user_agent = None
    if explicit is not None and str(explicit).strip():
        return client_key(explicit, accept_key=False), "header"
    key = client_key(user_agent, accept_key=False)
    return key, "user_agent" if key else None


def request_explicit_client(headers) -> str | None:
    """Return the opaque explicit client key without a User-Agent fallback."""
    try:
        value = headers.get("X-Clozn-Client-Id") if headers is not None else None
    except Exception:
        value = None
    return client_key(value, accept_key=False)


def project_key(value, *, accept_key: bool = True) -> str | None:
    """Normalize a caller-known project identifier to the opaque journal key."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if accept_key and _PROJECT_KEY_RE.fullmatch(text.lower()):
        return text.lower()
    return _digest(PROJECT_PREFIX, text)


def request_project(headers) -> str | None:
    """Return the opaque key for the explicit project association header."""
    try:
        value = headers.get("X-Clozn-Project-Id") if headers is not None else None
    except Exception:
        value = None
    return project_key(value, accept_key=False)
