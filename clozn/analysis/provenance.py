"""provenance.py -- did this answer come from the CONTEXT or from the model's own weights?

The question every RAG product needs answered and none can currently answer honestly: the model
cited a document, but did it actually *use* it? Today that is settled with attention heatmaps
(correlational -- attention weight is not influence) or by asking the model (which confabulates).

This measures it. Attention KNOCKOUT severs "the answering position may read position p" at every
layer and re-scores the answer token teacher-forced. If the answer survives having its context
access cut, it did not come from the context.

    context_dependence = 1 - P_cut(answer) / P_baseline(answer)

~1.0  the answer is carried by the context (cut it and the answer is gone)
~0.0  the answer is parametric (the model already knew it; the context was decoration)

Measured on Qwen2.5-7B (notes/CIRCUIT_TRACER_DESIGN.md section 5h) -- the dissociation is sharp:
an in-context key/value lookup and a name retrieval score ~1.0, while "the modern capital of Japan
is Tokyo" scores low: knock out every context position the search can find and the model STILL
says Tokyo, because it knows it parametrically.

WHY THIS AND NOT THE RESIDUAL TRACER: residual-site path patching cannot measure cross-position
influence at all on this stack (section 5f -- a flat 0.0% routed at every depth, because patching
one site lets the source re-supply downstream and the last layer is unpatchable). Cutting the
attention edge sidesteps both problems.

TWO THINGS THAT WILL GIVE YOU A WRONG ANSWER IF YOU SKIP THEM, both measured the hard way:
  1. RENORMALIZE. Position 0 is an attention sink holding a large share of the mass; zeroing it
     without rescaling shrinks the whole attention output -- a generic amplitude perturbation that
     looks like a routing result (it was the top-ranked position at +0.717 before renormalising,
     and vanishes from every span after). `renormalize=True` is the default here for that reason.
  2. GREEDY, NOT TOP-K. Single-position knockout loses to redundancy: cutting one token of a
     multi-token entity leaves its siblings to re-supply it, so every single position scores ~0.
     Greedy accumulation finds sets that are jointly decisive and individually invisible (measured:
     best single +0.11 vs the greedy span's +7.60 on the same prompt).

Requires a cloze-server started with --no-flash-attn (flash attention fuses the softmax so the
weights never materialize); `available()` checks, and the engine refuses rather than silently
no-opping. House style: the product-facing call never raises -- it returns a labeled
{"ok": False, "blocked": ...} dict.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Optional

DEFAULT_ENGINE = "http://127.0.0.1:8080"


@dataclass
class ProvenanceBudget:
    max_span: int = 6         # greedy steps (each = a few arms)
    candidates: int = 12      # strongest singles reconsidered per greedy step
    min_gain: float = 0.05    # stop when the best addition adds less than this
    n_controls: int = 3       # matched random-set draws per step
    renormalize: bool = True  # see the module docstring -- do not turn this off casually


# ============================================================== pure helpers (fixture-tested)

def context_dependence(base_logprob: float, cut_logprob: float) -> float:
    """1 - P_cut/P_base, clamped to [0, 1]. Both inputs are natural-log probabilities.
    1.0 = the context carried the answer; 0.0 = the answer survived losing its context."""
    import math
    if cut_logprob >= base_logprob:
        return 0.0
    ratio = math.exp(cut_logprob - base_logprob)     # P_cut / P_base, stable in log space
    return max(0.0, min(1.0, 1.0 - ratio))


def verdict(dependence: float, best_ratio: float) -> str:
    """CONTEXT_CARRIED / MIXED / PARAMETRIC / INCONCLUSIVE. `best_ratio` is the strongest
    span-vs-matched-random-control ratio: without separation from control, no verdict is earned
    no matter how large the raw effect."""
    if best_ratio < 3.0:
        return "INCONCLUSIVE"
    if dependence >= 0.8:
        return "CONTEXT_CARRIED"
    if dependence >= 0.3:
        return "MIXED"
    return "PARAMETRIC"


def _post(engine: str, path: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(engine.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def available(engine_url: str = DEFAULT_ENGINE) -> bool:
    """Is the engine started with --no-flash-attn (so attention weights materialize)?"""
    try:
        h = json.loads(urllib.request.urlopen(engine_url.rstrip("/") + "/health", timeout=10).read())
        return bool(h.get("capabilities", {}).get("attn_knockout"))
    except Exception:
        return False


# ==================================================================================== the API

def trace_provenance(prompt: str, continuation=None, *, engine_url: str = DEFAULT_ENGINE,
                     budget: Optional[ProvenanceBudget] = None, seed: int = 0) -> dict:
    """Which context positions carry this answer, and is it context-carried or parametric?

    `continuation` defaults to whatever the model generates greedily (so we never grade it on an
    answer it did not give). Returns a receipt dict; {"ok": False, "blocked": ...} on failure."""
    import numpy as np
    budget = budget or ProvenanceBudget()
    rng = np.random.default_rng(seed)
    try:
        health = json.loads(urllib.request.urlopen(engine_url.rstrip("/") + "/health",
                                                   timeout=10).read())
        if not health.get("capabilities", {}).get("attn_knockout"):
            return {"ok": False, "blocked": "engine lacks attn_knockout; start cloze-server with "
                                            "--no-flash-attn (flash attention fuses the softmax, so "
                                            "the attention weights never materialize)"}
        n_layer = int(health["n_layer"])

        if continuation is None:
            continuation = _post(engine_url, "/v1/completions",
                                 {"prompt": prompt, "max_tokens": 2,
                                  "temperature": 0})["choices"][0]["text"]
        base = _post(engine_url, "/score", {"prompt": prompt, "continuation": continuation,
                                            "topk": 3})
        n_p = int(base["n_prompt"])
        base_lp = float(base["tokens"][0]["logprob"])
        cont_ids = [int(base["tokens"][0]["id"])]
        answer = base["tokens"][0]["piece"]
        final = n_p - 1
        toks = _post(engine_url, "/harvest", {"text": prompt, "layer": 1})["tokens"]

        def cut(keys, topk=0):
            specs = [{"layer": L, "queries": [final], "keys": sorted(set(int(k) for k in keys)),
                      "renormalize": budget.renormalize} for L in range(n_layer)]
            r = _post(engine_url, "/score", {"prompt": prompt, "continuation_ids": cont_ids,
                                             "topk": topk, "attn_knockout": specs})
            lp = float(r["tokens"][0]["logprob"])
            return base_lp - lp, lp, r

        singles = sorted(((cut([p])[0], p) for p in range(final)), key=lambda x: -x[0])
        span, trace, cur = [], [], 0.0
        remaining = [p for _, p in singles]

        # ADJACENT-BIGRAM SEEDING (found by the R1 battery, case kv_late): greedy accumulation
        # cannot START when no SINGLE position clears min_gain -- and a multi-token entity whose
        # pieces are individually redundant (a passcode split into two digit tokens, a name split
        # across BPE pieces) is exactly that case: cut either piece alone and the sibling re-
        # supplies it, so every single scores ~0 and the old code returned an empty span --
        # "PARAMETRIC" -- for an answer that exists ONLY in the prompt. Entities are contiguous,
        # so before giving up we scan ADJACENT PAIRS (n-2 extra arms) and, if the best pair
        # clears min_gain, seed the span with both positions at once; greedy then continues
        # normally from that seed.
        seeded_bigram = None
        if not singles or singles[0][0] <= budget.min_gain:
            bigrams = sorted((((cut([p, p + 1])[0]), p) for p in range(final - 1)),
                             key=lambda x: -x[0])
            if bigrams and bigrams[0][0] > budget.min_gain:
                d2, p2 = bigrams[0]
                span = [p2, p2 + 1]
                cur = d2
                remaining = [p for p in remaining if p not in span]
                pool = [p for p in range(final) if p not in span]
                ctl = (max(cut(rng.choice(pool, size=2, replace=False).tolist())[0]
                           for _ in range(budget.n_controls))
                       if len(pool) >= 2 else float("nan"))
                seeded_bigram = {"step": 1, "added": [int(p2), int(p2 + 1)],
                                 "token": [toks[p2], toks[p2 + 1]], "joint": cur, "control": ctl,
                                 "ratio": (abs(cur) / abs(ctl)) if ctl and ctl == ctl
                                          and abs(ctl) > 1e-9 else None,
                                 "seeded": "adjacent_bigram"}
                trace.append(seeded_bigram)

        step_base = 1 if seeded_bigram else 0
        for step in range(budget.max_span):
            best = None
            for p in remaining[:budget.candidates]:
                d, _, _ = cut(span + [p])
                if best is None or d > best[0]:
                    best = (d, p)
            if best is None or best[0] <= cur + budget.min_gain:
                break
            cur, add = best
            span.append(add)
            remaining.remove(add)
            pool = [p for p in range(final) if p not in span]
            ctl = (max(cut(rng.choice(pool, size=len(span), replace=False).tolist())[0]
                       for _ in range(budget.n_controls))
                   if len(pool) >= len(span) else float("nan"))
            trace.append({"step": step_base + step + 1, "added": int(add), "token": toks[add],
                          "joint": cur, "control": ctl,
                          "ratio": (abs(cur) / abs(ctl)) if ctl and ctl == ctl and abs(ctl) > 1e-9
                                   else None})

        if not span:
            return {"ok": True, "answer": answer, "span": [], "dependence": 0.0,
                    "verdict": "PARAMETRIC", "note": "no context position changed the answer",
                    "baseline_logprob": base_lp}
        d_final, lp_cut, r_final = cut(span, topk=3)
        dep = context_dependence(base_lp, lp_cut)
        ratios = [t["ratio"] for t in trace if t["ratio"] is not None]
        best_ratio = max(ratios) if ratios else 0.0
        return {
            "ok": True,
            "answer": answer, "baseline_logprob": base_lp, "cut_logprob": lp_cut,
            "span": sorted(int(p) for p in span),
            "span_tokens": [toks[p] for p in sorted(span)],
            "delta": d_final, "dependence": dep,
            "best_control_ratio": best_ratio,
            "verdict": verdict(dep, best_ratio),
            "top3_after_cut": [{"piece": t["piece"], "logprob": t["logprob"]}
                               for t in r_final["tokens"][0]["topk"]],
            "best_single": {"pos": int(singles[0][1]), "token": toks[singles[0][1]],
                            "delta": singles[0][0]},
            "trace": trace,
            "config": {"renormalize": budget.renormalize, "n_layer": n_layer, "seed": seed,
                       "units": "delta = logprob(baseline) - logprob(context access cut)"},
        }
    except Exception as e:
        return {"ok": False, "blocked": f"{type(e).__name__}: {e}"}
