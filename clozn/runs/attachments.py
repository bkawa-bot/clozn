"""Atomic updates to fields attached to an existing SQLite run document."""
from __future__ import annotations

from . import store


def update_tiny_tests(rid: str, tiny_tests: list) -> bool:
    rec = store.get_run(rid)
    if rec is None:
        return False
    rec["tiny_tests"] = list(tiny_tests) if isinstance(tiny_tests, list) else []
    return store.replace_run(rec)
