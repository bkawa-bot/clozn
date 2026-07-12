"""analysis/model_diff.py -- MODEL-DIFF: where do two recorded runs (same prompt, different model/quant/
config) diverge, token by token. The data layer of the model-diff atlas: records in, dict out -- pure,
model-free, GPU-free, never raises. Served by clozn/server/routes/diff.py (POST /diff/runs).

WHAT THIS IS (and honestly is not), relative to clozn/receipts/quant_receipts.py:

  * quant_receipts diffs two TEACHER-FORCED /score arrays of the SAME recorded continuation -- aligned by
    construction, so its per-position logprob-delta / argmax-flip math is a real one-step counterfactual.
    That math is deliberately NOT re-implemented here, because it does not fit this input: two runs that
    each generated FREELY. After their first disagreement the two models were continuing DIFFERENT
    prefixes, so a per-position logprob comparison past that point would be comparing apples to oranges,
    and this module refuses to imply otherwise (see _DIFF_CAVEAT -- the same never-let-a-bare-number-
    overclaim discipline as quant_receipts' _QUANT_CAVEAT, reused as framing rather than as formulas).
  * What model_diff DOES claim is observational and exact: the common prefix, the first split (both
    pieces, both recorded confidences), the per-position same/differs map, and the "did a almost say b's
    token" signal read from a's own RECORDED alternatives at the split (a capture-time top-k list -- see
    _ALT_RANK_NOTE for what its rank does and does not mean; an empty list is UNKNOWN, never "no",
    mirroring quant_receipts' _TOPK_CAVEAT bucket discipline).
  * Text similarity ships as difflib ratio labeled SURFACE_SIMILARITY_LABEL verbatim -- wording, not
    meaning. No semantic-equivalence claim is made anywhere in this module.
  * The rigorous upgrade path for the same question stays quant_receipts.quant_receipt_for_run: replay
    ONE run's own answer under both model files via /score and diff the forced steps. This module is the
    cheap, generation-free companion for runs you already have, not a replacement for that receipt.

Input shape: clozn.runs.store run records -- {id, model, substrate, messages, response,
meta: {quant, model_file, ...}, trace: {tokens, confidence, alternatives, token_ids?, ...}} (see
clozn/runs/trace.py for the trace contract). Degrades cleanly: a run with no usable trace turns the diff
text-only ("trace_available": false); a missing/malformed run yields {"ok": false, "error": ...}.
"""
from __future__ import annotations

import difflib

MAX_POSITIONS = 200      # per-position detail cap; prefix/divergence math always runs on the FULL traces

# The task-mandated label, verbatim: char similarity is about the strings, never the semantics.
SURFACE_SIMILARITY_LABEL = "surface similarity — wording, not meaning"

_DIFF_CAVEAT = (
    "this is an OBSERVATIONAL diff of two independently-generated runs: after the first divergence the "
    "two models were continuing DIFFERENT prefixes, so every later disagreement is a downstream "
    "consequence of that first split, never independent evidence that the models disagree at that "
    "position. 'b_was_alternative_in_a' reads a's RECORDED capture-time alternatives at the split -- "
    "absence from that short list is NOT absence from the model's distribution. 'char_similarity' is "
    + SURFACE_SIMILARITY_LABEL + ". For the rigorous one-step counterfactual (teacher-force the SAME "
    "continuation under both model files and diff argmax/logprob per forced step), use "
    "clozn.receipts.quant_receipts -- its flip-vs-dependence-shift receipt is the honest upgrade path; "
    "its math is intentionally not re-applied here to unaligned free-running traces."
)

_ALT_RANK_NOTE = (
    "rank is 0-based within a's RECORDED alternatives at the divergence point (the capture-time top-k "
    "list, which excludes a's own committed token) -- NOT a full-vocabulary rank; 'found: false' means "
    "'not in the recorded list', which is weaker than 'not close'."
)

_ALT_UNKNOWN_NOTE = (
    "a recorded no alternatives at the divergence point, so whether b's token was nearly said is "
    "UNKNOWN -- counted as unknown, never folded into 'no' (mirrors quant_receipts' topk discipline)."
)


# ------------------------------------------------------------------------------------------ tiny coercers

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


# ------------------------------------------------------------------------------------- record extractors

def _user_prompts(run: dict) -> list[str]:
    """Every user message's content, in order -- the 'same prompt?' guard compares the full user side of
    the conversation, not just the last turn (two runs that share a final question but arrived there via
    different context are still not a controlled comparison)."""
    msgs = run.get("messages")
    if not isinstance(msgs, list):
        return []
    return [str(m.get("content", "")) for m in msgs if isinstance(m, dict) and m.get("role") == "user"]


def _response_text(run: dict) -> str:
    return str(run.get("response") or "")


def _trace(run: dict) -> dict:
    trace = run.get("trace")
    return trace if isinstance(trace, dict) else {}


def _tokens(run: dict) -> list[str] | None:
    """The committed token pieces, or None when the run has no usable per-token trace (old/light-tier
    run) -- the trigger for the text-only degrade."""
    toks = _trace(run).get("tokens")
    if not isinstance(toks, list) or not toks:
        return None
    return [str(t) for t in toks]


def _confidences(run: dict) -> list:
    conf = _trace(run).get("confidence")
    return conf if isinstance(conf, list) else []


def _conf_at(conf: list, i: int):
    return _float_or_none(conf[i]) if 0 <= i < len(conf) else None


def _alts_at(run: dict, i: int) -> list:
    alts = _trace(run).get("alternatives")
    if not isinstance(alts, list) or not (0 <= i < len(alts)):
        return []
    return alts[i] if isinstance(alts[i], list) else []


def _token_id_at(run: dict, i: int):
    ids = _trace(run).get("token_ids")
    if not isinstance(ids, list) or not (0 <= i < len(ids)):
        return None
    return _int_or_none(ids[i])


def _mean_confidence(run: dict):
    """Mean of the parseable trace confidences, None when there are none -- the same tolerant per-value
    parse testkit/ci.py's _min_confidence uses."""
    vals = [v for v in (_float_or_none(c) for c in _confidences(run)) if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _identity(run: dict) -> dict:
    """Who produced this run -- model + quant tag (run.meta's own fields, stamped by the engine's
    run_meta; see testkit/runner.py's REPRO_META_KEYS) so the caller can label the diff's two arms."""
    meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}
    return {
        "run_id": run.get("id"),
        "model": run.get("model") or meta.get("model_id") or None,
        "quant": meta.get("quant"),
        "model_file": meta.get("model_file"),
        "substrate": run.get("substrate") or None,
    }


# ---------------------------------------------------------------------------------- the almost-said signal

def _b_in_a_alternatives(run_a: dict, run_b: dict, index: int) -> dict:
    """At the first token divergence, was b's committed token among a's RECORDED alternatives -- the
    'did a almost say that' signal. Matches by token id when both sides carry one (exact), else by piece
    string. Three honest outcomes, never conflated: found (with its recorded rank/prob -- _ALT_RANK_NOTE),
    not-in-the-recorded-list, and UNKNOWN (a recorded no alternatives at all -- _ALT_UNKNOWN_NOTE)."""
    b_tokens = _tokens(run_b) or []
    if not (0 <= index < len(b_tokens)):
        return {"checked": False, "found": None, "rank": None, "prob": None,
                "note": "b committed no token at the divergence point (a length divergence) -- "
                        "nothing to look up in a's alternatives"}
    b_piece, b_id = b_tokens[index], _token_id_at(run_b, index)
    alts = _alts_at(run_a, index)
    if not alts:
        return {"checked": False, "found": None, "rank": None, "prob": None, "note": _ALT_UNKNOWN_NOTE}
    for rank, alt in enumerate(alts):
        if not isinstance(alt, dict):
            continue
        alt_id = _int_or_none(alt.get("token_id"))
        id_hit = b_id is not None and alt_id is not None and alt_id == b_id
        if id_hit or str(alt.get("piece", alt.get("text", ""))) == b_piece:
            return {"checked": True, "found": True, "rank": rank,
                    "prob": _float_or_none(alt.get("prob")),
                    "matched_by": "token_id" if id_hit else "piece", "note": _ALT_RANK_NOTE}
    return {"checked": True, "found": False, "rank": None, "prob": None, "note": _ALT_RANK_NOTE}


# ==================================================================================== the public surface

def diff_runs(run_a: dict, run_b: dict, *, max_positions: int = MAX_POSITIONS) -> dict:
    """Compare two recorded runs token by token. Pure (records in, dict out) and never raises: a
    missing/malformed run yields {"ok": false, "error", "missing"} (the clean error shape); runs without
    usable traces degrade to a text-only diff ("trace_available": false). See the module docstring for
    the full wire shape and the honesty contract."""
    missing = [name for name, r in (("a", run_a), ("b", run_b)) if not isinstance(r, dict) or not r]
    if missing:
        return {"mode": "model_diff", "ok": False, "missing": missing,
                "error": "run " + " and ".join(missing) + " missing/unreadable -- nothing to diff"}
    try:
        return _diff(run_a, run_b, max_positions=max_positions)
    except Exception as e:      # the never-raise discipline (mirrors quant_receipts), but with a reason
        return {"mode": "model_diff", "ok": False, "missing": [],
                "error": f"diff failed: {type(e).__name__}: {e}"}


def _diff(run_a: dict, run_b: dict, *, max_positions: int) -> dict:
    prompts_match = _user_prompts(run_a) == _user_prompts(run_b)
    text_a, text_b = _response_text(run_a), _response_text(run_b)

    out = {
        "mode": "model_diff",
        "ok": True,
        "a": _identity(run_a),
        "b": _identity(run_b),
        "prompts_match": prompts_match,
        "caveat": _DIFF_CAVEAT,
    }
    if not prompts_match:
        out["warn"] = ("the two runs were given different prompts -- this is NOT a controlled comparison "
                       "(a same-prompt pair isolates the model/quant/config change; different prompts "
                       "confound it with the prompt change itself)")

    summary = {
        "a_reply_chars": len(text_a),
        "b_reply_chars": len(text_b),
        "a_mean_confidence": _mean_confidence(run_a),
        "b_mean_confidence": _mean_confidence(run_b),
        "char_similarity": round(difflib.SequenceMatcher(a=text_a, b=text_b).ratio(), 4),
        "char_similarity_label": SURFACE_SIMILARITY_LABEL,
    }

    toks_a, toks_b = _tokens(run_a), _tokens(run_b)
    trace_missing = [name for name, t in (("a", toks_a), ("b", toks_b)) if t is None]
    if trace_missing:
        # ------- text-only degrade: no per-token claims at all, just the surface diff -----------------
        summary.update({"a_reply_tokens": None if toks_a is None else len(toks_a),
                        "b_reply_tokens": None if toks_b is None else len(toks_b),
                        "identical": text_a == text_b,
                        "b_was_alternative_in_a": None})
        out.update({
            "trace_available": False,
            "trace_missing": trace_missing,
            "note": ("run(s) " + " and ".join(trace_missing) + " carry no usable per-token trace "
                     "(an old or light-capture-tier run) -- text-only diff: no token positions, no "
                     "divergence index, no almost-said signal"),
            "common_prefix_len": None,
            "first_divergence": None,
            "positions": [],
            "positions_truncated": False,
            "summary": summary,
        })
        return out

    # ------- token-level diff (full traces; the positions LIST alone is capped) ----------------------
    len_a, len_b = len(toks_a), len(toks_b)
    conf_a, conf_b = _confidences(run_a), _confidences(run_b)

    common = 0
    for ta, tb in zip(toks_a, toks_b):
        if ta != tb:
            break
        common += 1

    if common == len_a == len_b:
        first_divergence = None                          # identical token streams -- no divergence at all
    elif common < min(len_a, len_b):
        first_divergence = {"index": common, "kind": "token_mismatch",
                            "a_piece": toks_a[common], "b_piece": toks_b[common],
                            "a_conf": _conf_at(conf_a, common), "b_conf": _conf_at(conf_b, common)}
    else:                                                # one reply is a strict prefix of the other
        first_divergence = {"index": common, "kind": "length_mismatch",
                            "a_piece": toks_a[common] if common < len_a else None,
                            "b_piece": toks_b[common] if common < len_b else None,
                            "a_conf": _conf_at(conf_a, common) if common < len_a else None,
                            "b_conf": _conf_at(conf_b, common) if common < len_b else None}

    n_pos = min(max(0, _int_or_none(max_positions) or 0), max(len_a, len_b))
    positions = []
    for i in range(n_pos):
        a_piece = toks_a[i] if i < len_a else None
        b_piece = toks_b[i] if i < len_b else None
        positions.append({"i": i, "a_piece": a_piece, "b_piece": b_piece,
                          "same": a_piece is not None and a_piece == b_piece,
                          "a_conf": _conf_at(conf_a, i) if a_piece is not None else None,
                          "b_conf": _conf_at(conf_b, i) if b_piece is not None else None})

    if first_divergence is not None:
        almost = _b_in_a_alternatives(run_a, run_b, first_divergence["index"])
    else:
        almost = {"checked": False, "found": None, "rank": None, "prob": None,
                  "note": "the runs never diverged -- nothing to look up"}

    summary.update({"a_reply_tokens": len_a, "b_reply_tokens": len_b,
                    "identical": first_divergence is None,
                    "b_was_alternative_in_a": almost})
    out.update({
        "trace_available": True,
        "common_prefix_len": common,
        "first_divergence": first_divergence,
        "positions": positions,
        "positions_truncated": max(len_a, len_b) > n_pos,
        "summary": summary,
    })
    return out
