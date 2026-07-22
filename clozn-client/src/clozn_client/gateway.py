"""Typed client for Clozn's public product gateway evidence/run API."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._transport import JsonTransport
from .models import LatestRun, ReceiptBundle, Run, RunPage, Timeline, require_object

_USER_AGENT = "clozn-client/0.2.0"


class CloznClient:
    """Public gateway client.

    This class intentionally has no ``score`` or attention-knockout methods. Those are native
    worker capabilities and live on :class:`clozn_client.EngineClient`.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", *, timeout: float = 120.0,
                 client_id: str | None = None, session_id: str | None = None):
        headers = {"User-Agent": _USER_AGENT}
        if client_id is not None:
            headers["X-Clozn-Client-Id"] = self._selector(client_id, "client_id")
        if session_id is not None:
            headers["X-Clozn-Session-Id"] = self._selector(session_id, "session_id")
        self._transport = JsonTransport(base_url, timeout=timeout, headers=headers)

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    def ready(self) -> dict[str, Any]:
        return require_object(self._transport.request_json("GET", "/readyz"), "ready response")

    def runs(self) -> tuple[Run, ...]:
        obj = require_object(self._transport.request_json("GET", "/runs"), "runs response")
        rows = obj.get("runs", [])
        if not isinstance(rows, list):
            from ._transport import CloznProtocolError
            raise CloznProtocolError("runs response.runs must be an array")
        return tuple(Run.from_json(row) for row in rows)

    def run(self, run_id: str) -> Run:
        rid = self._run_id(run_id)
        return Run.from_json(self._transport.request_json("GET", f"/runs/{rid}"))

    def latest_run(self, *, client_id: str | None = None, session_id: str | None = None,
                   client: str | None = None, model: str | None = None,
                   include_derived: bool = False) -> LatestRun:
        if all(value is None for value in (client_id, session_id, client)):
            # Constructor association headers also satisfy the gateway contract.
            headers = self._transport.headers
            if "X-Clozn-Client-Id" not in headers and "X-Clozn-Session-Id" not in headers:
                raise ValueError("latest_run needs client_id, session_id, client, or constructor association headers")
        query = {
            "client_id": None if client_id is None else self._selector(client_id, "client_id"),
            "session": None if session_id is None else self._selector(session_id, "session_id"),
            "client": None if client is None else self._selector(client, "client"),
            "model": None if model is None else self._selector(model, "model"),
            "include_derived": include_derived,
        }
        return LatestRun.from_json(self._transport.request_json("GET", "/runs/latest", query=query))

    def watch_runs(self, *, after: str | None = None, limit: int = 100,
                   client_id: str | None = None, session_id: str | None = None,
                   client: str | None = None, model: str | None = None,
                   include_derived: bool = False) -> RunPage:
        if isinstance(limit, bool) or not 1 <= int(limit) <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = {
            "after": after,
            "limit": int(limit),
            "client_id": None if client_id is None else self._selector(client_id, "client_id"),
            "session": None if session_id is None else self._selector(session_id, "session_id"),
            "client": None if client is None else self._selector(client, "client"),
            "model": None if model is None else self._selector(model, "model"),
            "include_derived": include_derived,
        }
        return RunPage.from_json(self._transport.request_json("GET", "/runs/watch", query=query))

    def timeline(self, run_id: str) -> Timeline:
        rid = self._run_id(run_id)
        return Timeline.from_json(self._transport.request_json("GET", f"/runs/{rid}/timeline"))

    def diagnosis(self, run_id: str) -> dict[str, Any]:
        return self._evidence_get(run_id, "diagnosis")

    def lineage(self, run_id: str) -> dict[str, Any]:
        return self._evidence_get(run_id, "lineage")

    def family(self, run_id: str) -> tuple[Run, ...]:
        obj = self._evidence_get(run_id, "family")
        rows = obj.get("runs", [])
        if not isinstance(rows, list):
            from ._transport import CloznProtocolError
            raise CloznProtocolError("family.runs must be an array")
        return tuple(Run.from_json(row) for row in rows)

    def spans(self, run_id: str) -> dict[str, Any]:
        return self._evidence_get(run_id, "spans")

    def export_receipt(self, run_id: str) -> ReceiptBundle:
        rid = self._run_id(run_id)
        return ReceiptBundle.from_json(self._transport.request_json("GET", f"/runs/{rid}/export"))

    def export_receipt_markdown(self, run_id: str) -> str:
        rid = self._run_id(run_id)
        return self._transport.request_text("GET", f"/runs/{rid}/export", query={"format": "md"})

    def explain(self, run_id: str) -> dict[str, Any]:
        rid = self._run_id(run_id)
        return require_object(self._transport.request_json("POST", f"/runs/{rid}/explain", body={}),
                              "explanation")

    def receipt(self, run_id: str, influence: Mapping[str, Any], *, mode: str = "forced") -> dict[str, Any]:
        rid = self._run_id(run_id)
        if mode not in {"regen", "forced", "both"}:
            raise ValueError("mode must be one of regen, forced, or both")
        if not isinstance(influence, Mapping) or not influence:
            raise ValueError("influence must be a non-empty object")
        body = {"mode": mode, "influence": dict(influence)}
        return require_object(self._transport.request_json("POST", f"/runs/{rid}/receipt", body=body),
                              "causal receipt")

    def _evidence_get(self, run_id: str, suffix: str) -> dict[str, Any]:
        rid = self._run_id(run_id)
        value = self._transport.request_json("GET", f"/runs/{rid}/{suffix}")
        return require_object(value, suffix)

    @staticmethod
    def _run_id(value: str) -> str:
        rid = str(value).strip()
        if not rid or "/" in rid or "?" in rid or "#" in rid:
            raise ValueError("run_id must be one non-empty path segment")
        return rid

    @staticmethod
    def _selector(value: str, label: str) -> str:
        text = str(value).strip()
        if not text or len(text) > 256 or any(ord(ch) < 32 for ch in text):
            raise ValueError(f"{label} must be a non-empty printable string up to 256 characters")
        return text
