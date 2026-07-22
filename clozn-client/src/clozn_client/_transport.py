"""Small stdlib HTTP transport shared by the public gateway and private engine clients."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any


class CloznClientError(RuntimeError):
    """Base exception for the installable researcher client."""


class CloznConnectionError(CloznClientError):
    """The requested Clozn process could not be reached."""


class CloznProtocolError(CloznClientError):
    """A successful response did not satisfy the JSON protocol contract."""


class CloznHTTPError(CloznClientError):
    """A Clozn endpoint returned a non-success HTTP status."""

    def __init__(self, method: str, path: str, status: int, body: Any):
        self.method = method
        self.path = path
        self.status = int(status)
        self.body = body
        super().__init__(f"{method} {path} -> {status}: {_error_message(body)}")


def _error_message(body: Any) -> str:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message") or error.get("code") or error)
        if error is not None:
            return str(error)
    return str(body)


class JsonTransport:
    """Synchronous JSON-over-HTTP transport with typed failures and no third-party dependency."""

    def __init__(self, base_url: str, *, timeout: float = 120.0,
                 headers: Mapping[str, str] | None = None):
        base = str(base_url).strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base
        self.timeout = float(timeout)
        self.headers = {str(k): str(v) for k, v in (headers or {}).items()}

    def request_json(self, method: str, path: str, *, body: Mapping[str, Any] | None = None,
                     query: Mapping[str, Any] | None = None) -> Any:
        method = method.upper()
        url = self._url(path, query)
        data = None if body is None else json.dumps(dict(body), separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json", **self.headers}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            raise CloznHTTPError(method, self._path(path, query), exc.code,
                                 self._decode_error(payload)) from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise CloznConnectionError(f"{method} {url}: {reason}") from None
        if not payload:
            return None
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloznProtocolError(
                f"{method} {self._path(path, query)} returned invalid JSON: {exc}") from None

    def request_text(self, method: str, path: str, *, query: Mapping[str, Any] | None = None) -> str:
        method = method.upper()
        url = self._url(path, query)
        request = urllib.request.Request(
            url, method=method, headers={"Accept": "text/plain", **self.headers})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            raise CloznHTTPError(method, self._path(path, query), exc.code,
                                 self._decode_error(payload)) from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise CloznConnectionError(f"{method} {url}: {reason}") from None

    def _url(self, path: str, query: Mapping[str, Any] | None) -> str:
        return self.base_url + self._path(path, query)

    @staticmethod
    def _path(path: str, query: Mapping[str, Any] | None) -> str:
        normalized = "/" + str(path).lstrip("/")
        if not query:
            return normalized
        pairs: list[tuple[str, str]] = []
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            pairs.append((str(key), str(value)))
        encoded = urllib.parse.urlencode(pairs)
        return normalized if not encoded else f"{normalized}?{encoded}"

    @staticmethod
    def _decode_error(payload: bytes) -> Any:
        text = payload.decode("utf-8", "replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
