"""runlog -- the product spine: every model interaction becomes an inspectable run.

Per the v1 roadmap (notes/STUDIO_PRODUCT_ROADMAP.md, Milestone 2): a JSON-file journal at ~/.clozn/runs/
that every path writes to (the OpenAI endpoint, studio chat, engine chat, the CLI). The Studio "Runs" page
and the Run Inspector read it back. Schema is intentionally normalized so the UI/backend contract is stable.

Stdlib only; a flat file-per-run store is plenty for v1 (don't over-architect).
"""
from __future__ import annotations

import glob
import json
import os
import time
import uuid

RUNS_DIR = os.path.join(os.path.expanduser("~/.clozn"), "runs")
KEEP = 1000                                              # prune to the most recent N runs

# the slim fields returned by list_runs() (the Runs page doesn't need full messages/trace)
SUMMARY_FIELDS = ("id", "created_at", "source", "client", "model", "substrate",
                  "prompt_summary", "response_summary", "memory", "behavior", "timing",
                  "parent_run_id", "flags")


def _ensure():
    os.makedirs(RUNS_DIR, exist_ok=True)


def _files():
    return glob.glob(os.path.join(RUNS_DIR, "run_*.json"))


def _summ(text: str, n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def _flags(rec: dict) -> list[str]:
    """Cheap UI flags derived from the record (the Runs page filters on these)."""
    f = []
    mem = rec.get("memory") or {}
    if mem.get("cards_applied"):
        f.append("memory")
    if mem.get("proposed_cards"):
        f.append("pending-memory")
    if (rec.get("behavior") or {}).get("active_dials"):
        f.append("steered")
    if rec.get("parent_run_id"):
        f.append("replayed")
    if rec.get("error"):
        f.append("error")
    conf = (rec.get("trace") or {}).get("confidence") or []
    if conf and min(conf) < 0.3:
        f.append("low-confidence")
    if len((rec.get("response") or "").split()) > 220:
        f.append("long")
    return f


def record(*, source: str, client: str = "unknown", model: str = "", substrate: str = "",
           messages=None, response: str = "", memory: dict | None = None, behavior: dict | None = None,
           trace: dict | None = None, started: float | None = None, ended: float | None = None,
           parent_run_id: str | None = None, changes_applied: dict | None = None,
           error: str | None = None) -> str | None:
    """Persist a completed run; return its id (or None on failure -- logging must never break a request)."""
    try:
        _ensure()
        started = started if started is not None else time.time()
        ended = ended if ended is not None else time.time()
        rid = f"run_{int(started * 1000):013x}_{uuid.uuid4().hex[:6]}"
        msgs = messages or []
        prompt = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        rec = {
            "id": rid,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "created_ts": started,
            "source": source, "client": client or "unknown", "model": model, "substrate": substrate,
            "prompt_summary": _summ(prompt), "response_summary": _summ(response),
            "messages": msgs, "response": response,
            "memory": memory or {}, "behavior": behavior or {}, "trace": trace or {},
            "timing": {"started_at": started, "ended_at": ended, "duration_ms": int((ended - started) * 1000)},
            "parent_run_id": parent_run_id, "changes_applied": changes_applied, "error": error,
        }
        rec["flags"] = _flags(rec)
        with open(os.path.join(RUNS_DIR, rid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)
        _prune()
        return rid
    except Exception:
        return None


def _prune():
    files = sorted(_files())                            # ids embed a zero-padded ms timestamp -> chronological
    for old in files[:-KEEP]:
        try:
            os.remove(old)
        except Exception:
            pass


def list_runs(limit: int = 50) -> list[dict]:
    _ensure()
    out = []
    for f in sorted(_files(), reverse=True)[:limit]:    # newest first
        try:
            r = json.load(open(f, encoding="utf-8"))
            out.append({k: r.get(k) for k in SUMMARY_FIELDS})
        except Exception:
            pass
    return out


def get_run(rid: str) -> dict | None:
    p = os.path.join(RUNS_DIR, rid + ".json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
