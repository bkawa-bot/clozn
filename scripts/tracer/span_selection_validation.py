"""Automatic span selection for attention knockout -- the §5g gap.

Why the obvious approaches fail:
  * residual causal mass points at the wrong tokens (it peaks where an entity is ASSEMBLED, not
    where the emitted token appeared) -- measured: tail-span +0.077 vs whole-name +2.78.
  * single-position knockout ranking loses to REDUNDANCY: cutting one token of a multi-token
    entity leaves its siblings to re-supply the information, so every single position looks ~0.

So drive selection by the knockout itself, and accumulate GREEDILY -- at each step add the key
position that most increases the JOINT effect. Greedy handles redundancy by construction: two
positions that are individually useless but jointly decisive get found on the second step.

Validation (the point of this script): greedy must REDISCOVER the whole name on the induction
prompt WITHOUT being told what the entity is, and must beat a random-set-of-equal-size control at
every step.
"""
import json
import time
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\span_select.json"
MAX_SPAN = 6
RENORM = True     # see cut(): keeps each attention row summing to 1 so the sink can't fake a result

CASES = [
    ("induction",
     "In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
     "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
     "The name of the wizard who cast the spell is"),
    ("factual", "The capital of France is"),
    ("distractor",
     "Kyoto was the capital of Japan for over a thousand years. Since the Meiji era, however, "
     "the government has been located elsewhere. The modern capital of Japan is"),
    ("in_context_kv",
     "Alice put the key in the blue box. Bob put the coin in the red box. "
     "Later, Alice went back to retrieve her key, opening the box coloured"),
]

def post(path, body, timeout=900):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

H = json.loads(urllib.request.urlopen(ENGINE + "/health", timeout=10).read())
NL = H["n_layer"]
if not H["capabilities"].get("attn_knockout"):
    raise SystemExit("engine must be started with --no-flash-attn")

rng = np.random.default_rng(0)
results = []

for name, prompt in CASES:
    cont = post("/v1/completions", {"prompt": prompt, "max_tokens": 2,
                                    "temperature": 0})["choices"][0]["text"]
    base = post("/score", {"prompt": prompt, "continuation": cont, "topk": 3})
    n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
    cont_ids = [int(base["tokens"][0]["id"])]
    FINAL = n_p - 1
    toks = post("/harvest", {"text": prompt, "layer": 15})["tokens"]
    tgt = base["tokens"][0]["piece"]

    def cut(keys, topk=0, renorm=RENORM):
        """Knock out final -> keys at EVERY layer (single layers are swamped by redundancy).

        renormalize=True rescales each surviving attention row back to sum 1. That matters a lot
        here: position 0 is an attention SINK holding a large share of the mass, so zeroing it
        WITHOUT renormalising shrinks the whole attention output -- a generic amplitude
        perturbation that masquerades as a routing result (it was the top single position at
        +0.717 in the un-renormalised run). Renormalising isolates "who is read" from "how much
        attention flows in total", which is the question we mean to ask."""
        specs = [{"layer": L, "queries": [FINAL], "keys": list(keys), "renormalize": renorm}
                 for L in range(NL)]
        r = post("/score", {"prompt": prompt, "continuation_ids": cont_ids, "topk": topk,
                            "attn_knockout": specs})
        return base_lp - float(r["tokens"][0]["logprob"]), r

    t0 = time.time()
    # ---- step 1: single-position causal scan over every key position ----
    singles = []
    for p in range(FINAL):
        d, _ = cut([p])
        singles.append((d, p))
    singles.sort(key=lambda x: -x[0])

    # ---- greedy accumulation ----
    span, trace = [], []
    remaining = [p for _, p in singles]
    cur = 0.0
    for step in range(MAX_SPAN):
        best = None
        # only consider the 12 strongest remaining singles: full re-scan each step is O(n^2) arms
        for p in remaining[:12]:
            d, _ = cut(span + [p])
            if best is None or d > best[0]:
                best = (d, p)
        if best is None or best[0] <= cur + 0.05:      # stop when nothing adds real mass
            break
        cur = best[0]
        span.append(best[1])
        remaining.remove(best[1])
        # matched control: a RANDOM set of the same size, drawn from non-span positions
        pool = [p for p in range(FINAL) if p not in span]
        if len(pool) < len(span):        # short prompt: no room for a matched control
            ctl = float("nan")
        else:
            ctl = max(cut(sorted(rng.choice(pool, size=len(span), replace=False).tolist()))[0]
                      for _ in range(3))
        trace.append({"step": step + 1, "added": best[1], "token": toks[best[1]],
                      "joint": cur, "control": ctl})
    wall = time.time() - t0

    d_final, r_final = cut(sorted(span), topk=3)
    top3 = ", ".join(f"{t['piece']!r}={np.exp(t['logprob']):.3f}"
                     for t in r_final["tokens"][0]["topk"])
    print(f"\n=== {name}: target {tgt!r} (prompt {n_p} tok, {wall:.0f}s) ===")
    print(f"  best single position: {singles[0][1]} {toks[singles[0][1]]!r} "
          f"delta {singles[0][0]:+.3f}   <- what naive selection would have picked")
    print(f"  {'step':>4} {'+pos':>5} {'token':<12} {'joint':>9} {'control':>9} {'ratio':>7}")
    for t in trace:
        ratio = abs(t["joint"]) / max(abs(t["control"]), 1e-9)
        print(f"  {t['step']:>4} {t['added']:>5} {t['token']!r:<12} {t['joint']:>+9.3f} "
              f"{t['control']:>+9.3f} {ratio:>6.1f}x")
    print(f"  SPAN {sorted(span)} = {[toks[p] for p in sorted(span)]}")
    print(f"  final delta {d_final:+.3f} | top-3 after the cut: {top3}")
    results.append({"case": name, "target": tgt, "span": sorted(span),
                    "span_tokens": [toks[p] for p in sorted(span)], "trace": trace,
                    "best_single": {"pos": singles[0][1], "delta": singles[0][0]},
                    "delta_final": d_final, "wall_s": wall})

print("\nGreedy is validated if it recovers a semantically coherent span WITHOUT being told the")
print("entity, beats its matched control at every step, and beats the best single position.")
json.dump(results, open(OUT, "w"), indent=2)
print(f"-> {OUT}")
