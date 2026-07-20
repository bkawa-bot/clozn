"""Harden the knockout result: the all-layer effect needs a matched RANDOM-KEY control.

"Cut final->name at every layer = +2.78" is only meaningful against "cut final->{4 random
positions} at every layer". The competitor control (+0.018) already argues specificity, but
Pellinore is a semantically special choice; random keys are the honest matched null.

Also: does it generalize beyond the induction prompt?
"""
import json
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\knockout_controls.json"

def post(path, body, timeout=900):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

H = json.loads(urllib.request.urlopen(ENGINE + "/health", timeout=10).read())
NL = H["n_layer"]

CASES = [
    ("induction",
     "In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
     "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
     "The name of the wizard who cast the spell is",
     "name"),          # target span = the name being retrieved
    ("factual", "The capital of France is", "subject"),
    ("distractor",
     "Kyoto was the capital of Japan for over a thousand years. Since the Meiji era, however, "
     "the government has been located elsewhere. The modern capital of Japan is", "subject"),
]
rng = np.random.default_rng(0)
rows = []

for name, prompt, kind in CASES:
    cont = post("/v1/completions", {"prompt": prompt, "max_tokens": 2,
                                    "temperature": 0})["choices"][0]["text"]
    base = post("/score", {"prompt": prompt, "continuation": cont, "topk": 3})
    n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
    cont_ids = [int(base["tokens"][0]["id"])]
    FINAL = n_p - 1
    toks = post("/harvest", {"text": prompt, "layer": 15})["tokens"]

    # the causally-relevant span: use the layer-21 mean-ablation peak +/- its neighbours, which the
    # (layer x position) map showed is where a multi-token entity is represented
    Hl = np.zeros((n_p, 3584), dtype=np.float32)
    for s in range(0, n_p, 48):
        ch = list(range(s, min(s + 48, n_p)))
        r = post("/score", {"prompt": prompt, "continuation_ids": cont_ids,
                            "capture": {"layers": [21], "positions": ch}})
        cap = (r.get("captured") or {}).get("21") or {}
        for p in ch:
            Hl[p] = np.asarray(cap[str(p)], dtype=np.float32)
    mean21 = Hl.mean(axis=0).tolist()
    solo = []
    for p in range(1, FINAL):
        r = post("/score", {"prompt": prompt, "continuation_ids": cont_ids,
                            "write": {"layer": 21, "positions": [p], "values": mean21}})
        solo.append((base_lp - float(r["tokens"][0]["logprob"]), p))
    solo.sort(key=lambda x: -abs(x[0]))
    # SPAN = a CONTIGUOUS run around the causal peak, not the top-k scattered positions. A
    # multi-token entity spreads its information across its own tokens, so cutting attention to
    # only the peak leaves the neighbours to re-supply it (measured: whole-name span +2.78 vs
    # peak-plus-two-stragglers +0.10). Extend while a neighbour still carries >=20% of the peak.
    by_pos = {p: d for d, p in solo}
    peak = solo[0][1]
    thresh = 0.2 * abs(solo[0][0])
    lo = hi = peak
    while lo - 1 >= 1 and abs(by_pos.get(lo - 1, 0.0)) >= thresh:
        lo -= 1
    while hi + 1 < FINAL and abs(by_pos.get(hi + 1, 0.0)) >= thresh:
        hi += 1
    span = list(range(lo, hi + 1))
    pool = [p for p in range(FINAL) if p not in span]
    if len(pool) < len(span):                        # short prompt: not enough room for a control
        print(f"{name:12} SKIPPED -- prompt too short for a matched random-key control "
              f"(span {len(span)}, pool {len(pool)})")
        continue

    def ko_all(keys):
        specs = [{"layer": L, "queries": [FINAL], "keys": keys} for L in range(NL)]
        r = post("/score", {"prompt": prompt, "continuation_ids": cont_ids, "topk": 3,
                            "attn_knockout": specs})
        return base_lp - float(r["tokens"][0]["logprob"]), r

    d_real, r_real = ko_all(span)
    ctrls = []
    for t in range(5):                                # 5 matched random-key draws
        keys = sorted(rng.choice(pool, size=len(span), replace=False).tolist())
        d, _ = ko_all(keys)
        ctrls.append(d)
    cm = float(np.median(ctrls))
    cx = float(np.max(np.abs(ctrls)))
    ratio = abs(d_real) / max(cx, 1e-9)
    top3 = ", ".join(f"{t['piece']!r}={np.exp(t['logprob']):.3f}" for t in r_real["tokens"][0]["topk"])
    print(f"{name:12} target {base['tokens'][0]['piece']!r:10} span {span} "
          f"{[toks[p] for p in span]}")
    print(f"             cut span   delta {d_real:>+8.3f}   | random-key median {cm:>+7.3f} "
          f"max {cx:>+7.3f} | {ratio:>6.1f}x")
    print(f"             top-3 after the cut: {top3}")
    rows.append({"case": name, "span": span, "span_toks": [toks[p] for p in span],
                 "delta_real": d_real, "controls": ctrls, "ratio": ratio,
                 "base_lp": base_lp, "target": base["tokens"][0]["piece"]})

print("\nA large real/control ratio means the final position's ability to READ THOSE POSITIONS is")
print("what carries the answer -- a cross-position causal claim, which residual-site path patching")
print("could not produce (it measured a flat 0.0% at every depth).")
json.dump(rows, open(OUT, "w"), indent=2)
print(f"-> {OUT}")
