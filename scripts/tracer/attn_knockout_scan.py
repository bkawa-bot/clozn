"""Attention knockout: does cutting 'final position reads the name' kill the answer?

This is the cross-position measurement residual patching could not do (section 5f). Instead of
ablating a residual site and hoping the effect survives to the destination, we sever the EDGE:
zero A[head, query=final, key=name-tail] at a layer, so the final position literally cannot read
the name there.

Checks, in order:
  0. sanity  -- knockout with an empty key set / self-key should be ~no-op; the API refuses when
                flash attention is on.
  1. NULL    -- baseline must be unchanged when nothing is knocked out.
  2. effect  -- cut final->name-tail at each layer; where does the answer break?
  3. control -- cut final->a RANDOM position of the same size at the same layer.
  4. heads   -- at the layer that matters, which individual head carries it?
"""
import json
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\knockout.json"
PROMPT = ("In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
          "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
          "The name of the wizard who cast the spell is")
CONT = " Zorblax"
NAME_TAIL = [9, 10, 11, 12]     # ' Z','or','bl','ax'
COMPETITOR = [19, 20, 21]       # ' Pell','in','ore'

def post(path, body, timeout=600):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.read().decode("utf-8", "replace")}

h = post("/health", {}) if False else json.loads(urllib.request.urlopen(ENGINE + "/health", timeout=10).read())
print(f"engine: n_layer={h['n_layer']} attn_knockout capability={h['capabilities'].get('attn_knockout')}")

base = post("/score", {"prompt": PROMPT, "continuation": CONT, "topk": 3})
n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
cont_ids = [int(base["tokens"][0]["id"])]
FINAL = n_p - 1
toks = post("/harvest", {"text": PROMPT, "layer": 15})["tokens"]
print(f"target {base['tokens'][0]['piece']!r} logprob {base_lp:.4f} | final={FINAL} "
      f"name tail={[toks[p] for p in NAME_TAIL]}")

def ko(specs, topk=0):
    r = post("/score", {"prompt": PROMPT, "continuation_ids": cont_ids, "topk": topk,
                        "attn_knockout": specs})
    if "_http_error" in r:
        return None, r["_http_error"]
    return base_lp - float(r["tokens"][0]["logprob"]), r

# --- 1. NULL: knocking out a position's attention TO ITSELF only at a layer -> tiny/no effect ---
d_self, _ = ko([{"layer": 10, "queries": [FINAL], "keys": [FINAL]}])
print(f"\n1. NULL (final reads only-itself cut @L10): delta {d_self:+.4f}"
      f"  {'OK (small)' if abs(d_self) < 0.5 else 'LARGE -- investigate'}")

# --- 2. cut final -> name tail, layer by layer ---
print(f"\n2. cut final -> name tail {NAME_TAIL}, per layer:")
print(f"   {'layer':>5} {'delta':>9}   {'random-key control':>19}")
rng = np.random.default_rng(0)
rows = []
for L in range(0, h["n_layer"]):
    d, _ = ko([{"layer": L, "queries": [FINAL], "keys": NAME_TAIL}])
    ctrl_keys = sorted(rng.choice([p for p in range(FINAL) if p not in NAME_TAIL],
                                  size=len(NAME_TAIL), replace=False).tolist())
    dc, _ = ko([{"layer": L, "queries": [FINAL], "keys": ctrl_keys}])
    rows.append({"layer": L, "delta": d, "control": dc, "control_keys": ctrl_keys})
    if d is None:
        print(f"   {L:>5} refused: {_}"); break
    flag = "  <---" if abs(d) > 1.0 else ""
    print(f"   {L:>5} {d:>+9.4f}   {dc:>+19.4f}{flag}")

live = [r for r in rows if r["delta"] is not None]
if live:
    best = max(live, key=lambda r: abs(r["delta"]))
    print(f"\n   strongest: L{best['layer']} delta {best['delta']:+.4f} "
          f"(control {best['control']:+.4f})")

    # --- 3. cumulative: cut it at ALL layers at once ---
    allspec = [{"layer": L, "queries": [FINAL], "keys": NAME_TAIL} for L in range(h["n_layer"])]
    d_all, r_all = ko(allspec, topk=3)
    print(f"\n3. cut at EVERY layer: delta {d_all:+.4f}")
    if r_all and "tokens" in r_all:
        print("   top-3 now: " + ", ".join(
            f"{t['piece']!r}={np.exp(t['logprob']):.3f}" for t in r_all["tokens"][0]["topk"]))
    # competitor control: cut final -> Pellinore at every layer
    d_comp, _ = ko([{"layer": L, "queries": [FINAL], "keys": COMPETITOR}
                    for L in range(h["n_layer"])])
    print(f"   same, but cutting the COMPETITOR {COMPETITOR}: delta {d_comp:+.4f}")

    # --- 4. per-head at the strongest layer ---
    Lb = best["layer"]
    print(f"\n4. per-head at L{Lb} (n_head={h.get('n_head', '?')}):")
    head_rows = []
    for hd in range(h.get("n_head", 28)):
        d, err = ko([{"layer": Lb, "head": hd, "queries": [FINAL], "keys": NAME_TAIL}])
        if d is None:
            print(f"   head {hd}: refused {err}"); break
        head_rows.append((d, hd))
    head_rows.sort(key=lambda x: -abs(x[0]))
    for d, hd in head_rows[:6]:
        print(f"   head {hd:>3}: delta {d:+.4f}")
    tot = sum(d for d, _ in head_rows)
    print(f"   sum over heads {tot:+.4f} vs all-heads-at-once {best['delta']:+.4f}")

json.dump({"rows": rows, "base_lp": base_lp, "final": FINAL, "tokens": toks}, open(OUT, "w"), indent=2)
print(f"\n-> {OUT}")
