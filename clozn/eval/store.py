"""Persist the latest OUTCOME-grounded calibration report to ~/.clozn, so a served route can surface the
TRUTH tier beside the PROXY curve (GET /journal/actuary) WITHOUT re-running live generation on every
request. The truth tier needs the model (probes must actually be answered), so it can't be computed
on-demand in a GET the way actuary's proxy curve can -- `clozn eval --save` writes it, the route reads it.

Deliberately dependency-light: mirrors clozn.runs.store's `~/.clozn` convention with a plain atomic write,
importing no server code (eval stays a pure analysis package). `save`/`load` take an optional path so tests
point at a tmp dir.
"""
from __future__ import annotations

import json
import os
import time

_PATH = os.path.join(os.path.expanduser("~/.clozn"), "eval_report.json")


def save(payload: dict, path: str | None = None) -> str:
    """Atomically write the report payload (report + policy + provenance), stamping `saved_ts` if absent.
    Returns the path written."""
    path = path or _PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = dict(payload)
    out.setdefault("saved_ts", time.time())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    return path


def load(path: str | None = None) -> dict | None:
    """The last saved report, or None if none exists / the file is unreadable. Never raises."""
    try:
        with open(path or _PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
