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
                  "finish_reason", "parent_run_id", "flags")


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
    if rec.get("finish_reason") == "length":
        f.append("truncated")            # hit the token cap -- the reply was cut off, not a natural stop
    conf = (rec.get("trace") or {}).get("confidence") or []
    if conf and min(conf) < 0.3:
        f.append("low-confidence")
    if len((rec.get("response") or "").split()) > 220:
        f.append("long")
    return f


# --------------------------------------------------------------------------- trace (per-token timeline)
# The Run Inspector's timeline (issue B3) wants, per generated token: what was committed, how sure the
# model was (confidence 0..1), and what it nearly said instead (alternatives). Two code paths already
# carry that: the CLI's stream_ar and the engine chat (when it captures the SSE event stream). Both hand
# us the SAME per-token "step" shape -- keep the mapping in ONE pure, testable place so the on-disk trace
# schema stays a single contract. The three keys below are load-bearing (read by _flags and the UI); do
# not rename them.
TRACE_KEYS = ("tokens", "confidence", "alternatives")


def _clean_alts(alts) -> list[dict]:
    """Normalize a step's alternatives to [{piece, prob}] (floats rounded, junk dropped)."""
    out = []
    for a in alts or []:
        if not isinstance(a, dict):
            continue
        try:
            out.append({"piece": str(a.get("piece", "")), "prob": round(float(a.get("prob", 0.0)), 4)})
        except (TypeError, ValueError):
            continue
    return out


def steps_to_trace(steps) -> dict:
    """Map a list of per-token steps -> the run's `trace` dict {tokens, confidence, alternatives}.

    A step is a dict as produced by stream_ar / the engine-chat capture:
      {"piece": str, "conf": float, "alts": [{"piece": str, "prob": float}, ...]}
    (legacy keys "token"/"confidence"/"alternatives" are accepted too). Empty/None in -> a clean empty
    {} (no half-populated trace): the caller stores that as-is and the timeline simply shows nothing.
    `alternatives` is only included when at least one step actually recorded some (the non-streaming engine
    path, for instance, has confidence but no alts -> we omit the key rather than store a wall of []).
    Pure: no I/O, no model -- unit-testable with fabricated steps.
    """
    steps = [s for s in (steps or []) if isinstance(s, dict)]
    if not steps:
        return {}
    tokens, confidence, alternatives = [], [], []
    for s in steps:
        tokens.append(str(s.get("piece", s.get("token", ""))))
        try:
            confidence.append(round(float(s.get("conf", s.get("confidence", 0.0))), 4))
        except (TypeError, ValueError):
            confidence.append(0.0)
        alternatives.append(_clean_alts(s.get("alts", s.get("alternatives"))))
    trace = {"tokens": tokens, "confidence": confidence}
    if any(alternatives):                                   # only carry alts if some token actually had them
        trace["alternatives"] = alternatives
    return trace


def accumulate_ar_events(events) -> list[dict]:
    """Fold the engine's autoregressive SSE frames into ordered per-token steps (the raw trace material).

    The engine streams, per committed token, a `tokens_committed` frame (the chosen piece + its confidence)
    and a `step_lens` frame (the top-k it weighed at that position). We pair them by position -- exactly as
    the CLI's stream_ar does -- into a replayable step list: where was it uncertain, and what did it almost
    say. `events` is an iterable of already-parsed frame dicts; unknown/malformed frames are skipped. Pure
    (no network) so the same accumulation is unit-tested here and reused by both the CLI and the server.
    """
    by_pos: dict = {}
    order: list = []                                        # preserve first-seen order for positions
    for obj in events or []:
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        if typ == "tokens_committed":
            for it in obj.get("items", []):
                pos = it.get("pos")
                if pos not in by_pos:
                    order.append(pos)
                try:
                    conf = round(float(it.get("conf", 0.0)), 4)
                except (TypeError, ValueError):
                    conf = 0.0
                by_pos[pos] = {"pos": pos, "piece": str(it.get("piece", "")), "conf": conf, "alts": []}
        elif typ == "step_lens":                            # the top-k the model weighed at this position
            pos = (obj.get("positions") or [None])[0]
            step = by_pos.get(pos)
            if step:
                chosen = step["piece"]
                pieces, probs = obj.get("pieces", []), obj.get("probs", [])
                step["alts"] = [{"piece": str(p), "prob": round(float(pr), 4)}
                                for p, pr in zip(pieces, probs) if str(p) != chosen][:3]
    return [by_pos[p] for p in sorted(order, key=lambda x: (x is None, x))]


def finish_reason_from_frames(frames) -> str | None:
    """Pluck the generation's stop cause from the engine's SSE frames -- WHY it stopped, which the engine
    already computes (eos -> "stop", length | steps_exhausted -> "length"; cloze_server.cpp) and drops on
    the floor here. It rides the FINAL frame two ways: as choices[0].finish_reason on the OpenAI-style AR
    frame, or as a top-level finish_reason on the state-stream 'final' frame -- accept either. Returns the
    last one seen ("stop"|"length"|...) or None when no frame carried it (a stream that errored before it
    finished). Pure (no network): the same fold is unit-tested here and reused by both engine paths."""
    reason = None
    for obj in frames or []:
        if not isinstance(obj, dict):
            continue
        # the AR `gen_finished` event carries the raw stop cause (eos | length | steps_exhausted) and is
        # emitted on EVERY generation -- map it like the engine's finish_reason() so the reason is captured
        # even when the trailing OpenAI `choices` frame isn't in this frame slice.
        if obj.get("type") == "gen_finished" and isinstance(obj.get("reason"), str):
            reason = "stop" if obj["reason"] == "eos" else "length"
        if isinstance(obj.get("finish_reason"), str):          # state-stream 'final' frame (top-level)
            reason = obj["finish_reason"]
        ch = obj.get("choices")                                # OpenAI-style final frame: choices[0]
        if isinstance(ch, list) and ch and isinstance(ch[0], dict) \
                and isinstance(ch[0].get("finish_reason"), str):
            reason = ch[0]["finish_reason"]
    return reason


def _norm_trace(trace) -> dict:
    """Coerce whatever a caller passes for `trace` into the stored shape. A ready trace dict is kept as-is
    (only the known keys); a raw list of steps is run through steps_to_trace; anything else -> {}."""
    if isinstance(trace, list):
        return steps_to_trace(trace)
    if isinstance(trace, dict):
        return {k: trace[k] for k in TRACE_KEYS if k in trace}
    return {}


def record(*, source: str, client: str = "unknown", model: str = "", substrate: str = "",
           messages=None, response: str = "", memory: dict | None = None, behavior: dict | None = None,
           trace: dict | None = None, started: float | None = None, ended: float | None = None,
           parent_run_id: str | None = None, changes_applied: dict | None = None,
           error: str | None = None, finish_reason: str | None = None,
           meta: dict | None = None) -> str | None:
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
            "memory": memory or {}, "behavior": behavior or {}, "trace": _norm_trace(trace),
            "timing": {"started_at": started, "ended_at": ended, "duration_ms": int((ended - started) * 1000)},
            "parent_run_id": parent_run_id, "changes_applied": changes_applied, "error": error,
            "finish_reason": finish_reason, "meta": meta or {},
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
