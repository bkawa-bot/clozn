"""Bounded serialization for stateful product operations.

The current product adapter keeps generation metadata, steering, and memory state on one
shared object.  Until those become request-local, concurrent POST dispatch would let two
runs overwrite each other's evidence.  This gate admits a bounded queue and executes one
POST at a time; GET health, Studio assets, and run inspection remain concurrent.
"""
from __future__ import annotations

import os
import threading
import time


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


class RequestGate:
    def __init__(self, capacity: int = 32, wait_timeout: float = 600.0):
        self.capacity = max(1, int(capacity))
        self.wait_timeout = max(0.001, float(wait_timeout))
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._turn = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = 0
        self._waiting = 0

    @classmethod
    def from_env(cls):
        return cls(
            capacity=_positive_int("CLOZN_MAX_PENDING_REQUESTS", 32),
            wait_timeout=_positive_float("CLOZN_QUEUE_TIMEOUT", 600.0),
        )

    def acquire(self, cancel_check=None, poll_interval: float = 0.2) -> str | None:
        """Admit one request, or explain why not. Returns ``None`` on admission, else one of ``"full"`` |
        ``"timeout"`` | ``"cancelled"``.

        `cancel_check`, when given, is polled every `poll_interval` seconds while this request is QUEUED
        waiting for its turn (never while merely checking the bounded `_slots` semaphore just below, which
        is already non-blocking) -- a callable returning True means the caller's own liveness signal (in
        production: "has the requesting client's TCP connection already closed", see http_policy.
        client_gone) says to abandon the wait rather than occupy a queue slot for the full `wait_timeout`
        (600s by default) after nobody is left to serve. `cancel_check=None` preserves the exact original
        single blocking wait -- every existing caller/test that has no liveness probe to offer keeps the
        pre-cancellation behavior byte-for-byte (one `Lock.acquire(timeout=...)` call, not a poll loop)."""
        if not self._slots.acquire(blocking=False):
            return "full"
        with self._state_lock:
            self._waiting += 1
        acquired = False
        cancelled = False
        try:
            if cancel_check is None:
                acquired = self._turn.acquire(timeout=self.wait_timeout)
            else:
                deadline = time.monotonic() + self.wait_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    acquired = self._turn.acquire(timeout=min(poll_interval, remaining))
                    if acquired:
                        break
                    if cancel_check():
                        cancelled = True
                        break
        finally:
            with self._state_lock:
                self._waiting -= 1
            if not acquired:
                self._slots.release()
        if not acquired:
            return "cancelled" if cancelled else "timeout"
        with self._state_lock:
            self._active = 1
        return None

    def release(self) -> None:
        with self._state_lock:
            self._active = 0
        self._turn.release()
        self._slots.release()

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "capacity": self.capacity,
                "wait_timeout_seconds": self.wait_timeout,
            }
