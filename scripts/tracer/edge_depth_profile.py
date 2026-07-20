"""Why did wizard@L2 -> final@L25 route 100% while ax@L21 -> final@L25 routed 0%?

Hypothesis: an upstream ablation's effect is only fully readable at the final position once it has
been INTEGRATED there. wizard acts early and is fully absorbed into the final position's state by
L25; ax acts late (L21) and is still in flight through L22-L27, so patching the final position at
L25 cannot reproduce it. If so, capturing the SAME edge at progressively later layers should make
the routed fraction climb toward 100% -- a measurement of WHEN each source is consumed.

This also checks a methodological worry: "routes 100% through the final position at the last layer"
is close to tautological (that state alone determines the output), so a routed fraction is only
informative when read as a DEPTH PROFILE, not as a single number.
"""
import json
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\when_consumed.json"
PROMPT = ("In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
          "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
          "The name of the wizard who cast the spell is")
CONT = " Zorblax"
CAP_LAYERS = [15, 18, 21, 23, 25, 26]   # 27 = last layer: no l_out tensor (engine now 400s on it)

def post(path, body, timeout=600):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

base = post("/score", {"prompt": PROMPT, "continuation": CONT, "topk": 0})
n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
cont_ids = [int(base["tokens"][0]["id"])]
toks = post("/harvest", {"text": PROMPT, "layer": 15})["tokens"]
FINAL = n_p - 1

MEAN = {}
for L in sorted(set(CAP_LAYERS + [2, 21])):
    Hl = np.zeros((n_p, 3584), dtype=np.float32)
    for s in range(0, n_p, 48):
        ch = list(range(s, min(s + 48, n_p)))
        r = post("/score", {"prompt": PROMPT, "continuation_ids": cont_ids,
                            "capture": {"layers": [L], "positions": ch}})
        cap = (r.get("captured") or {}).get(str(L)) or {}
        for p in ch:
            Hl[p] = np.asarray(cap[str(p)], dtype=np.float32)
    MEAN[L] = Hl.mean(axis=0)

def run(write, capture=None):
    body = {"prompt": PROMPT, "continuation_ids": cont_ids, "write": write}
    if capture:
        body["capture"] = capture
    r = post("/score", body)
    return base_lp - float(r["tokens"][0]["logprob"]), r.get("captured")

SOURCES = [("wizard@L2", 2, 5), ("ax@L21", 21, 12), ("bl@L21", 21, 11), ("Pell@L25", 25, 19)]
print(f"routed fraction into the FINAL position ({FINAL}), by capture depth")
print(f"  {'source':<12} {'delta_A':>8} | " + " ".join(f"L{L:<5}" for L in CAP_LAYERS))
print("-" * 78)
rows = []
for name, lA, pA in SOURCES:
    dA, _ = run({"layer": lA, "positions": [pA], "values": MEAN[lA].tolist()})
    cells = []
    for lB in CAP_LAYERS:
        if lB < lA:
            cells.append(None); continue
        _, cap = run({"layer": lA, "positions": [pA], "values": MEAN[lA].tolist()},
                     capture={"layers": [lB], "positions": [FINAL]})
        hB = (cap or {}).get(str(lB), {}).get(str(FINAL))
        if hB is None:
            cells.append(None); continue
        d_edge, _ = run({"layer": lB, "positions": [FINAL], "values": hB})
        cells.append(d_edge / dA if abs(dA) > 1e-9 else float("nan"))
    txt = " ".join("  --  " if c is None else f"{c:>6.1%}" for c in cells)
    print(f"  {name:<12} {dA:>+8.3f} | {txt}")
    rows.append({"source": name, "layer": lA, "pos": pA, "delta_A": dA,
                 "cap_layers": CAP_LAYERS, "routed": cells})

print("\nA source whose routed fraction is already ~100% at a shallow capture depth was consumed")
print("EARLY; one that only reaches ~100% deep in the stack was still propagating. Reading a")
print("single routed fraction without this profile would have been misleading.")
json.dump({"rows": rows, "final": FINAL, "tokens": toks}, open(OUT, "w"), indent=2)
print(f"-> {OUT}")
