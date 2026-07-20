"""Is Tokyo really PARAMETRIC, or did greedy just miss the position that matters?

My §5h claim was "the model still says Tokyo because it knows Japan's capital from its weights."
But the greedy span never cut ' Japan'. If cutting ' Japan' (and every other content word) also
leaves Tokyo standing, the claim holds. If cutting ' Japan' kills it, the claim was wrong and the
right reading is "greedy's stopping rule quit too early."

Also runs the comparison the question deserves: attention KNOCKOUT vs LEAVE-ONE-OUT (delete the
text), on the same target, so the difference between them is visible rather than asserted.
"""
import json
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
PROMPT = ("Kyoto was the capital of Japan for over a thousand years. Since the Meiji era, however, "
          "the government has been located elsewhere. The modern capital of Japan is")

def post(path, body, timeout=900):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

NL = json.loads(urllib.request.urlopen(ENGINE + "/health", timeout=10).read())["n_layer"]
base = post("/score", {"prompt": PROMPT, "continuation": " Tokyo", "topk": 3})
n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
cont_ids = [int(base["tokens"][0]["id"])]
FINAL = n_p - 1
toks = post("/harvest", {"text": PROMPT, "layer": 1})["tokens"]
print(f"baseline P(' Tokyo') = {np.exp(base_lp):.4f}\n")
print("token map:")
for i, t in enumerate(toks):
    print(f"  {i:>3} {t!r}", end="   " if (i + 1) % 4 else "\n")
print()

def cut(keys, label, renorm=True):
    specs = [{"layer": L, "queries": [FINAL], "keys": sorted(set(keys)), "renormalize": renorm}
             for L in range(NL)]
    r = post("/score", {"prompt": PROMPT, "continuation_ids": cont_ids, "topk": 3,
                        "attn_knockout": specs})
    lp = float(r["tokens"][0]["logprob"])
    top = ", ".join(f"{t['piece']!r}={np.exp(t['logprob']):.3f}" for t in r["tokens"][0]["topk"])
    print(f"  {label:<44} P {np.exp(lp):.4f}  (delta {base_lp-lp:+.3f})   top3: {top}")
    return base_lp - lp

japan = [i for i, t in enumerate(toks) if "Japan" in t]
kyoto = [0, 1]   # 'Ky','oto' -- the entity is split across tokens; a substring match misses it
content = [i for i, t in enumerate(toks) if t.strip() and t.strip()[0].isalpha() and len(t.strip()) > 2]

print("ATTENTION KNOCKOUT (input identical, only the reading is cut):")
cut(japan, f"cut ' Japan' {japan}")
cut(japan + [30], "cut ' Japan' + ' capital'")
cut(content, f"cut EVERY content word ({len(content)} positions)")
cut(list(range(FINAL)), "cut EVERY position (total context blindness)")
cut(kyoto, f"cut ' Kyoto' {kyoto} (the distractor)")

print("\nLEAVE-ONE-OUT (delete the text -> the model sees a DIFFERENT input):")
def loo(drop, label):
    kept = "".join(t for i, t in enumerate(toks) if i not in set(drop))
    r = post("/score", {"prompt": kept, "continuation": " Tokyo", "topk": 3})
    lp = float(r["tokens"][0]["logprob"])
    top = ", ".join(f"{t['piece']!r}={np.exp(t['logprob']):.3f}" for t in r["tokens"][0]["topk"])
    print(f"  {label:<44} P {np.exp(lp):.4f}  (delta {base_lp-lp:+.3f})   top3: {top}")
    print(f"      prompt became: ...{kept[-60:]!r}")

loo(japan, f"delete ' Japan' {japan}")
loo(kyoto, "delete ' Kyoto'")
