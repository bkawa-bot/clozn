"""Sidecar-style updates for persisted run records."""
from __future__ import annotations

import json
import os

from . import store


def update_tiny_tests(rid: str, tiny_tests: list) -> bool:
    """Attach tiny-test harness results to a stored run's `tiny_tests` field in place.

    `rid` is reachable from `clozn test --attach` with an arbitrary string on the command line, so it
    gets the same path-traversal guard as the read side (store._safe_run_path) -- an unsafe id must never
    let this WRITE a file outside RUNS_DIR."""
    p = store._safe_run_path(rid)
    if not p or not os.path.isfile(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        if not isinstance(rec, dict):
            return False
        rec["tiny_tests"] = list(tiny_tests) if isinstance(tiny_tests, list) else []
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        return True
    except Exception:
        return False
