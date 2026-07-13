"""Compact run summaries and list flags."""
from __future__ import annotations


# The slim fields returned by list_runs() (the Runs page doesn't need full messages/trace).
SUMMARY_FIELDS = (
    "id",
    "created_at",
    "source",
    "client",
    "model",
    "substrate",
    "prompt_summary",
    "response_summary",
    "memory",
    "behavior",
    "timing",
    "finish_reason",
    "parent_run_id",
    "flags",
)


def _summ(text: str, n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def _flags(rec: dict) -> list[str]:
    """Cheap UI flags derived from the record (the Runs page filters on these)."""
    f = []
    mem = rec.get("memory") or {}
    if mem.get("cards_applied") or mem.get("anchored"):
        f.append("memory")
    if mem.get("anchored"):
        f.append("anchored-memory")
    if mem.get("proposed_cards"):
        f.append("pending-memory")
    if (rec.get("behavior") or {}).get("active_dials"):
        f.append("steered")
    if rec.get("parent_run_id"):
        f.append("replayed")
    if rec.get("error"):
        f.append("error")
    if rec.get("finish_reason") == "length":
        f.append("truncated")
    conf = (rec.get("trace") or {}).get("confidence") or []
    if conf and min(conf) < 0.3:
        f.append("low-confidence")
    if len((rec.get("response") or "").split()) > 220:
        f.append("long")
    return f


def _summary(r: dict) -> dict:
    """One run dict -> the compact SUMMARY_FIELDS view."""
    return {k: r.get(k) for k in SUMMARY_FIELDS}
