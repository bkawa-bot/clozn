"""runlog -- the product spine: every model interaction becomes an inspectable run.

Per the v1 roadmap (notes/STUDIO_PRODUCT_ROADMAP.md, Milestone 2): a JSON-file journal at ~/.clozn/runs/
that every path writes to (the OpenAI endpoint, studio chat, engine chat, the CLI). The Studio "Runs" page
and the Run Inspector read it back. Schema is intentionally normalized so the UI/backend contract is stable.

Stdlib only; a flat file-per-run store is plenty for v1 (don't over-architect).
"""
from __future__ import annotations

import glob
import json
import math
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
# schema stays a single contract. The v1 parallel-array keys are load-bearing (read by _flags, explain,
# and older UI code); do not rename them. `steps` is the richer v2 per-token schema stored alongside them.
TRACE_KEYS = ("tokens", "confidence", "alternatives", "steps", "workspace_readouts")


def _float_or_none(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _rounded_prob(x):
    v = _float_or_none(x)
    return round(v, 4) if v is not None else None


def _logprob(prob):
    p = _float_or_none(prob)
    if p is None or p <= 0:
        return None
    return round(math.log(p), 6)


def _entropy_from_probs(probs):
    vals = [_float_or_none(p) for p in (probs or [])]
    vals = [p for p in vals if p is not None and p > 0]
    if not vals:
        return None
    return round(-sum(p * math.log(p) for p in vals), 6)


def _clean_alt(a) -> dict | None:
    """Normalize one alternative, preserving token id/text/prob/logprob when they are real."""
    if not isinstance(a, dict):
        return None
    piece = str(a.get("piece", a.get("text", "")))
    prob = _rounded_prob(a.get("prob", a.get("confidence", a.get("conf"))))
    item = {"piece": piece, "text": piece}
    token_id = _int_or_none(a.get("token_id", a.get("id")))
    if token_id is not None:
        item["token_id"] = token_id
    if prob is not None:
        item["prob"] = prob
        lp = _logprob(prob)
        if lp is not None:
            item["logprob"] = lp
    elif _float_or_none(a.get("logprob")) is not None:
        item["logprob"] = round(float(a["logprob"]), 6)
    return item


def _clean_alts(alts) -> list[dict]:
    """Normalize a step's alternatives to rich alt dicts; junk entries are dropped."""
    out = []
    for a in alts or []:
        item = _clean_alt(a)
        if item is not None:
            out.append(item)
    return out


def _clean_step(s, fallback_index: int) -> dict | None:
    """Normalize one raw token step into the v2 schema while keeping v1 aliases readable."""
    if not isinstance(s, dict):
        return None
    piece = str(s.get("piece", s.get("token", s.get("text", ""))))
    index = _int_or_none(s.get("index", s.get("pos")))
    if index is None:
        index = int(fallback_index)
    prob = _rounded_prob(s.get("prob", s.get("conf", s.get("confidence"))))
    step = {"index": index, "piece": piece, "text": piece}
    token_id = _int_or_none(s.get("token_id", s.get("id")))
    if token_id is not None:
        step["token_id"] = token_id
    if prob is not None:
        step["prob"] = prob
        step["confidence"] = prob
        lp = _logprob(prob)
        if lp is not None:
            step["logprob"] = lp
    elif _float_or_none(s.get("logprob")) is not None:
        step["logprob"] = round(float(s["logprob"]), 6)
    alts = _clean_alts(s.get("alts", s.get("alternatives")))
    step["alternatives"] = alts
    for k in ("entropy", "wall_ms", "dt_ms"):
        v = _float_or_none(s.get(k))
        if v is not None:
            step[k] = round(v, 6 if k == "entropy" else 3)
    return step


def _steps_from_parallel(trace: dict) -> list[dict]:
    tokens = trace.get("tokens") if isinstance(trace, dict) else None
    if not isinstance(tokens, list):
        return []
    confidence = trace.get("confidence") if isinstance(trace.get("confidence"), list) else []
    alternatives = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    out = []
    for i, piece in enumerate(tokens):
        raw = {"index": i, "piece": piece, "alts": alternatives[i] if i < len(alternatives) else []}
        if i < len(confidence):
            raw["conf"] = confidence[i]
        if i < len(token_ids):
            raw["token_id"] = token_ids[i]
        step = _clean_step(raw, i)
        if step is not None:
            out.append(step)
    return out


def steps_to_trace(steps) -> dict:
    """Map per-token steps -> the run's trace dict with v1 arrays plus rich v2 `steps`.

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
    rich = []
    for i, s in enumerate(steps):
        step = _clean_step(s, i)
        if step is not None:
            rich.append(step)
    if not rich:
        return {}
    tokens = [s.get("piece", "") for s in rich]
    confidence = [s.get("prob", 0.0) for s in rich]
    alternatives = [s.get("alternatives", []) for s in rich]
    trace = {"tokens": tokens, "confidence": confidence, "steps": rich}
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
                step = {"pos": pos, "index": _int_or_none(pos), "id": it.get("id"),
                        "piece": str(it.get("piece", "")), "conf": conf, "alts": []}
                for k in ("wall_ms", "dt_ms"):
                    if it.get(k) is not None:
                        step[k] = it.get(k)
                by_pos[pos] = step
        elif typ == "step_lens":                            # the top-k the model weighed at this position
            positions = obj.get("positions") or [None]
            pieces, probs = obj.get("pieces", []), obj.get("probs", [])
            ids = obj.get("ids") or [None] * len(pieces)
            try:
                k = int(obj.get("k") or (len(probs) // max(1, len(positions))))
            except (TypeError, ValueError):
                k = len(probs)
            entropies = obj.get("entropy", obj.get("entropies"))
            for row, pos in enumerate(positions):
                step = by_pos.get(pos)
                if not step:
                    continue
                start, end = row * k, row * k + k
                chosen_piece = step.get("piece")
                chosen_id = _int_or_none(step.get("id"))
                alts = []
                for tid, piece, prob in zip(ids[start:end], pieces[start:end], probs[start:end]):
                    token_id = _int_or_none(tid)
                    if (chosen_id is not None and token_id == chosen_id) or str(piece) == str(chosen_piece):
                        continue
                    alts.append({"token_id": token_id, "piece": str(piece), "prob": prob})
                    if len(alts) >= 3:
                        break
                step["alts"] = alts
                if isinstance(entropies, list) and row < len(entropies):
                    step["entropy"] = entropies[row]
                elif _float_or_none(entropies) is not None:
                    step["entropy"] = entropies
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
        if isinstance(trace.get("steps"), list):
            norm = steps_to_trace(trace["steps"])
        else:
            norm = {k: trace[k] for k in ("tokens", "confidence", "alternatives") if k in trace}
            steps = _steps_from_parallel(norm)
            if steps:
                norm["steps"] = steps
        if "workspace_readouts" in trace:
            norm["workspace_readouts"] = trace["workspace_readouts"]
        return {k: norm[k] for k in TRACE_KEYS if k in norm}
    return {}


def _normalize_workspace_readouts(rid: str, readouts) -> list[dict]:
    """Keep explicit readouts, filling the run id when a provider leaves it blank."""
    out = []
    for r in readouts or []:
        if not isinstance(r, dict):
            continue
        item = dict(r)
        item.setdefault("type", "workspace_readout")
        if not item.get("run_id"):
            item["run_id"] = rid
        if not item.get("provider_type") or not item.get("readout_kind"):
            try:
                import workspace_lens
                fields = workspace_lens.taxonomy_fields(item.get("provider"), item.get("readout_kind"))
                if fields.get("provider_type") and not item.get("provider_type"):
                    item["provider_type"] = fields["provider_type"]
                if fields.get("readout_kind") and not item.get("readout_kind"):
                    item["readout_kind"] = fields["readout_kind"]
            except Exception:
                pass
        out.append(item)
    return out


def _with_workspace_readouts(rid: str, trace: dict, workspace_provider=None) -> dict:
    """Attach explicit/provider Workspace Lens readouts to token traces.

    Mock readouts are intentionally not auto-generated here. Real logging paths
    pass a provider callback from the server when engine/SAE concepts are live;
    fixtures may pass ready `trace.workspace_readouts` directly.
    """
    if not isinstance(trace, dict) or not trace.get("tokens"):
        return trace
    if trace.get("workspace_readouts"):
        trace = dict(trace)
        trace["workspace_readouts"] = _normalize_workspace_readouts(rid, trace["workspace_readouts"])
        return trace
    if workspace_provider is None:
        return trace
    try:
        readouts = workspace_provider(rid, trace)
        readouts = _normalize_workspace_readouts(rid, readouts)
        if readouts:
            trace = dict(trace)
            trace["workspace_readouts"] = readouts
    except Exception:
        pass
    return trace


def record(*, source: str, client: str = "unknown", model: str = "", substrate: str = "",
           messages=None, response: str = "", memory: dict | None = None, behavior: dict | None = None,
           trace: dict | None = None, started: float | None = None, ended: float | None = None,
           parent_run_id: str | None = None, changes_applied: dict | None = None,
           error: str | None = None, finish_reason: str | None = None,
           meta: dict | None = None, assembled_messages=None, workspace_provider=None) -> str | None:
    """Persist a completed run; return its id (or None on failure -- logging must never break a request)."""
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
            "source": source, "client": client or "unknown", "model": model, "substrate": substrate,
            "prompt_summary": _summ(prompt), "response_summary": _summ(response),
            "messages": msgs, "response": response,
            "assembled_messages": assembled_messages if assembled_messages is not None else None,
            "memory": memory or {}, "behavior": behavior or {}, "trace": norm_trace,
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


def _load_runs() -> list[dict]:
    """Best-effort load of every persisted run; corrupt files are ignored like list_runs()."""
    _ensure()
    out = []
    for f in _files():
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            if isinstance(r, dict) and r.get("id"):
                out.append(r)
        except Exception:
            pass
    return out


def _change_label(run: dict) -> str | None:
    """Short lineage label for known replay/branch change specs."""
    changes = run.get("changes_applied") or {}
    if not isinstance(changes, dict) or not changes:
        return None
    if isinstance(changes.get("label"), str) and changes["label"].strip():
        return changes["label"].strip()

    parts = []
    if changes.get("memory_off"):
        parts.append("memory off")
    disabled = changes.get("disabled_memory_ids")
    if isinstance(disabled, list) and disabled:
        parts.append(f"memory disabled ({len(disabled)})")
    if changes.get("behavior_off"):
        parts.append("dials neutralized")

    overrides = changes.get("behavior_overrides")
    if isinstance(overrides, dict):
        eff = ((run.get("behavior") or {}).get("active_dials") or {})
        for k in sorted(overrides):
            v = eff.get(k, overrides.get(k))
            try:
                parts.append(f"{k} {float(v):.2f}")
            except (TypeError, ValueError):
                parts.append(str(k))

    if changes.get("nudge"):
        name = str(changes.get("nudge"))
        eff = ((run.get("behavior") or {}).get("active_dials") or {})
        suffix = ""
        try:
            if eff.get(name) is not None:
                suffix = f" -> {float(eff[name]):.2f}"
        except (TypeError, ValueError):
            suffix = ""
        parts.append(f"{name} up{suffix}")

    if changes.get("branch_turn") is not None:
        branch = f"branched from turn {changes.get('branch_turn')}"
        if changes.get("edited_user"):
            branch += " (edited question)"
        if changes.get("kv_snapshot"):
            branch += " + KV snapshot"
        parts.append(branch)

    if changes.get("plain"):
        parts.append("re-roll")
    if parts:
        return ", ".join(parts)

    keys = [str(k) for k in sorted(changes)[:3]]
    return "changed " + ", ".join(keys) if keys else None


def _lineage_summary(run: dict, current_id: str | None = None) -> dict:
    timing = run.get("timing") or {}
    return {
        "id": run.get("id"),
        "parent_run_id": run.get("parent_run_id"),
        "created_at": run.get("created_at"),
        "created_ts": run.get("created_ts"),
        "source": run.get("source"),
        "client": run.get("client"),
        "model": run.get("model"),
        "substrate": run.get("substrate"),
        "prompt_summary": run.get("prompt_summary"),
        "response_summary": run.get("response_summary"),
        "finish_reason": run.get("finish_reason"),
        "duration_ms": timing.get("duration_ms"),
        "changes_applied": run.get("changes_applied") or {},
        "change_label": _change_label(run),
        "flags": run.get("flags") or [],
        "is_current": run.get("id") == current_id,
    }


def lineage(rid: str, limit: int = 500) -> dict | None:
    """Return ancestors, siblings, children, and a simple descendant tree for a run."""
    runs = _load_runs()
    by_id = {r.get("id"): r for r in runs if r.get("id")}
    current = by_id.get(rid)
    if not current:
        return None

    children_by_parent: dict[str, list[dict]] = {}
    for r in runs:
        parent = r.get("parent_run_id")
        if parent:
            children_by_parent.setdefault(parent, []).append(r)
    for children in children_by_parent.values():
        children.sort(key=lambda r: (r.get("created_ts") or 0, r.get("id") or ""))

    ancestors = []
    seen = {rid}
    parent_id = current.get("parent_run_id")
    while parent_id and parent_id in by_id and parent_id not in seen and len(ancestors) < limit:
        parent = by_id[parent_id]
        ancestors.append(parent)
        seen.add(parent_id)
        parent_id = parent.get("parent_run_id")
    ancestors.reverse()

    root = ancestors[0] if ancestors else current
    tree_count = 0

    def build_tree(run: dict, seen_tree: set[str]) -> dict:
        nonlocal tree_count
        tree_count += 1
        node = _lineage_summary(run, rid)
        node_id = run.get("id")
        kids = []
        if tree_count < limit and node_id not in seen_tree:
            next_seen = set(seen_tree)
            if node_id:
                next_seen.add(node_id)
            for child in children_by_parent.get(node_id, []):
                if tree_count >= limit:
                    break
                if child.get("id") in next_seen:
                    continue
                kids.append(build_tree(child, next_seen))
        node["children"] = kids
        return node

    parent_for_siblings = current.get("parent_run_id")
    siblings = []
    if parent_for_siblings:
        siblings = [r for r in children_by_parent.get(parent_for_siblings, []) if r.get("id") != rid]

    return {
        "run_id": rid,
        "root_id": root.get("id"),
        "original": _lineage_summary(root, rid),
        "current": _lineage_summary(current, rid),
        "ancestors": [_lineage_summary(r, rid) for r in ancestors],
        "children": [_lineage_summary(r, rid) for r in children_by_parent.get(rid, [])],
        "siblings": [_lineage_summary(r, rid) for r in siblings],
        "tree": build_tree(root, set()),
    }
