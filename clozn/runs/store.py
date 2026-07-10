"""Run storage for persisted Clozn model interactions.

Every product path writes completed interactions to a JSON-file journal. This module owns file IO and record
lifecycle; trace normalization, compact summaries, attachments, and lineage live in sibling modules.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import uuid

from .summaries import SUMMARY_FIELDS, _flags, _summ, _summary
from .trace import (
    TRACE_KEYS,
    _clean_alt,
    _clean_alts,
    _clean_step,
    _entropy_from_probs,
    _float_or_none,
    _int_or_none,
    _logprob,
    _normalize_workspace_readouts,
    _norm_trace,
    _rounded_prob,
    _steps_from_parallel,
    _with_workspace_readouts,
    accumulate_ar_events,
    finish_reason_from_frames,
    steps_to_trace,
)


RUNS_DIR = os.path.join(os.path.expanduser("~/.clozn"), "runs")
KEEP = 1000

# A run id is generated as `run_<hex-ts>_<hex-uuid6>` (see record() below); this also covers the plain
# alnum/underscore/hyphen ids test fixtures and legacy records use. No dots, no slashes, no backslashes --
# so "..", "/", "\\", and an absolute path (which on POSIX starts with "/" and on Windows needs either a
# backslash or a "C:" drive letter, both excluded by the charset) can never be constructed from it.
_SAFE_RID = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_run_path(rid) -> str | None:
    """`rid` (attacker-influenced: it arrives straight from a URL path segment on every `/runs/<rid>/...`
    route, and from a CLI arg on `clozn test --attach`) -> the on-disk path inside RUNS_DIR, or None if
    `rid` isn't safe. Two independent layers, both must pass:

      1. a strict allow-list character check (no path separators, no '..', never absolute) BEFORE `rid`
         is allowed anywhere near os.path.join -- this alone already makes traversal impossible;
      2. resolve the joined path with os.path.realpath and confirm its parent is RUNS_DIR itself --
         belt-and-suspenders in case the charset above is ever loosened.

    Never raises: an unsafe id is a normal, expected input (a hostile request, or just a typo'd id), not
    an error -- callers treat None exactly like "file not found"."""
    if not isinstance(rid, str) or not rid or not _SAFE_RID.match(rid) or os.path.isabs(rid):
        return None
    base = os.path.realpath(RUNS_DIR)
    path = os.path.realpath(os.path.join(RUNS_DIR, rid + ".json"))
    if os.path.dirname(path) != base:
        return None
    return path


def _ensure():
    os.makedirs(RUNS_DIR, exist_ok=True)


def _files():
    return glob.glob(os.path.join(RUNS_DIR, "run_*.json"))


def record(*, source: str, client: str = "unknown", model: str = "", substrate: str = "",
           messages=None, response: str = "", memory: dict | None = None, behavior: dict | None = None,
           trace: dict | None = None, started: float | None = None, ended: float | None = None,
           parent_run_id: str | None = None, changes_applied: dict | None = None,
           error: str | None = None, finish_reason: str | None = None,
           meta: dict | None = None, assembled_messages=None, final_prompt: str | None = None,
           workspace_provider=None) -> str | None:
    """Persist a completed run; return its id, or None if logging fails."""
    try:
        _ensure()
        started = started if started is not None else time.time()
        ended = ended if ended is not None else time.time()
        rid = f"run_{int(started * 1000):013x}_{uuid.uuid4().hex[:6]}"
        msgs = messages or []
        prompt = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        norm_trace = _with_workspace_readouts(rid, _norm_trace(trace), workspace_provider)
        rec = {
            "id": rid,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "created_ts": started,
            "source": source,
            "client": client or "unknown",
            "model": model,
            "substrate": substrate,
            "prompt_summary": _summ(prompt),
            "response_summary": _summ(response),
            "messages": msgs,
            "response": response,
            "assembled_messages": assembled_messages if assembled_messages is not None else None,
            "final_prompt": final_prompt,
            "memory": memory or {},
            "behavior": behavior or {},
            "trace": norm_trace,
            "timing": {"started_at": started, "ended_at": ended, "duration_ms": int((ended - started) * 1000)},
            "parent_run_id": parent_run_id,
            "changes_applied": changes_applied,
            "error": error,
            "finish_reason": finish_reason,
            "meta": meta or {},
        }
        rec["flags"] = _flags(rec)
        with open(os.path.join(RUNS_DIR, rid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)
        _prune()
        return rid
    except Exception:
        return None


def _prune():
    files = sorted(_files())
    for old in files[:-KEEP]:
        try:
            os.remove(old)
        except Exception:
            pass


def list_runs(limit: int = 50) -> list[dict]:
    _ensure()
    out = []
    for f in sorted(_files(), reverse=True)[:limit]:
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            out.append(_summary(r))
        except Exception:
            pass
    return out


def get_run(rid: str) -> dict | None:
    p = _safe_run_path(rid)
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def update_tiny_tests(rid: str, tiny_tests: list) -> bool:
    from .attachments import update_tiny_tests as _update_tiny_tests

    return _update_tiny_tests(rid, tiny_tests)


def _load_runs() -> list[dict]:
    from .lineage import _load_runs as _load

    return _load()


def _change_label(run: dict) -> str | None:
    from .lineage import _change_label as _label

    return _label(run)


def _lineage_summary(run: dict, current_id: str | None = None) -> dict:
    from .lineage import _lineage_summary as _summary_for_lineage

    return _summary_for_lineage(run, current_id)


def lineage(rid: str, limit: int = 500) -> dict | None:
    from .lineage import lineage as _lineage

    return _lineage(rid, limit)


def lineage_family(rid: str, limit: int = 2000) -> list[dict] | None:
    from .lineage import lineage_family as _lineage_family

    return _lineage_family(rid, limit)
