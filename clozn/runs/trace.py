"""Run trace normalization and engine event folding."""
from __future__ import annotations

import math


# --------------------------------------------------------------------------- trace (per-token timeline)
# The Run Inspector's timeline wants, per generated token: what was committed, how sure the model was
# (confidence 0..1), and what it nearly said instead (alternatives). Two code paths already carry that:
# the CLI's stream_ar and the engine chat capture. Both hand us the same per-token "step" shape; keep the
# mapping in one pure place so the on-disk trace schema stays a single contract.
TRACE_KEYS = (
    "tokens",
    "confidence",
    "alternatives",
    "token_ids",
    "logprobs",
    "topk_entropy",
    "steps",
    "workspace_readouts",
)


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
    for k in ("entropy", "topk_entropy", "wall_ms", "dt_ms"):
        v = _float_or_none(s.get(k))
        if v is not None:
            step[k] = round(v, 6 if k in ("entropy", "topk_entropy") else 3)
    return step


def _steps_from_parallel(trace: dict) -> list[dict]:
    """Reconstruct v2 `steps` from a ready trace dict's parallel arrays."""
    tokens = trace.get("tokens") if isinstance(trace, dict) else None
    if not isinstance(tokens, list):
        return []
    confidence = trace.get("confidence") if isinstance(trace.get("confidence"), list) else []
    alternatives = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    topk_entropy = trace.get("topk_entropy") if isinstance(trace.get("topk_entropy"), list) else []
    out = []
    for i, piece in enumerate(tokens):
        raw = {"index": i, "piece": piece, "alts": alternatives[i] if i < len(alternatives) else []}
        if i < len(confidence):
            raw["conf"] = confidence[i]
        if i < len(token_ids):
            raw["token_id"] = token_ids[i]
        if i < len(topk_entropy) and topk_entropy[i] is not None:
            raw["topk_entropy"] = topk_entropy[i]
        step = _clean_step(raw, i)
        if step is not None:
            out.append(step)
    return out


def steps_to_trace(steps) -> dict:
    """Map per-token steps -> the run's trace dict with v1 arrays plus rich v2 `steps`."""
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
    token_ids = [s.get("token_id") for s in rich]
    logprobs = [s.get("logprob") for s in rich]
    topk_entropy = [s.get("topk_entropy") for s in rich]
    trace = {"tokens": tokens, "confidence": confidence, "steps": rich}
    if any(alternatives):
        trace["alternatives"] = alternatives
    if any(v is not None for v in token_ids):
        trace["token_ids"] = token_ids
    if any(v is not None for v in logprobs):
        trace["logprobs"] = logprobs
    if any(v is not None for v in topk_entropy):
        trace["topk_entropy"] = topk_entropy
    return trace


def accumulate_ar_events(events) -> list[dict]:
    """Fold the engine's autoregressive SSE frames into ordered per-token steps."""
    by_pos: dict = {}
    order: list = []
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
                step = {
                    "pos": pos,
                    "index": _int_or_none(pos),
                    "id": it.get("id"),
                    "piece": str(it.get("piece", "")),
                    "conf": conf,
                    "alts": [],
                }
                for k in ("wall_ms", "dt_ms"):
                    if it.get(k) is not None:
                        step[k] = it.get(k)
                by_pos[pos] = step
        elif typ == "step_lens":
            positions = obj.get("positions") or [None]
            pieces, probs = obj.get("pieces", []), obj.get("probs", [])
            ids = obj.get("ids") or [None] * len(pieces)
            try:
                k = int(obj.get("k") or (len(probs) // max(1, len(positions))))
            except (TypeError, ValueError):
                k = len(probs)
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
                topk_entropy = _entropy_from_probs(probs[start:end])
                if topk_entropy is not None:
                    step["topk_entropy"] = topk_entropy
    return [by_pos[p] for p in sorted(order, key=lambda x: (x is None, x))]


def finish_reason_from_frames(frames) -> str | None:
    """Pluck the generation's stop cause from the engine's SSE frames."""
    reason = None
    for obj in frames or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "gen_finished" and isinstance(obj.get("reason"), str):
            reason = "stop" if obj["reason"] == "eos" else "length"
        if isinstance(obj.get("finish_reason"), str):
            reason = obj["finish_reason"]
        ch = obj.get("choices")
        if (
            isinstance(ch, list)
            and ch
            and isinstance(ch[0], dict)
            and isinstance(ch[0].get("finish_reason"), str)
        ):
            reason = ch[0]["finish_reason"]
    return reason


def _norm_trace(trace) -> dict:
    """Coerce whatever a caller passes for `trace` into the stored shape."""
    if isinstance(trace, list):
        return steps_to_trace(trace)
    if isinstance(trace, dict):
        if isinstance(trace.get("steps"), list):
            norm = steps_to_trace(trace["steps"])
        else:
            norm = {
                k: trace[k]
                for k in ("tokens", "confidence", "alternatives", "token_ids", "logprobs", "topk_entropy")
                if k in trace
            }
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
                from clozn.readouts import workspace_lens

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
    """Attach explicit/provider Workspace Lens readouts to token traces."""
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
