"""facts_efficacy_engine.py -- can the facts injection rung run ENGINE-NATIVE, GPU-fast, instead
of the CPU torch store? (Phase-2 facts efficacy tuning.)

The torch SlotMem store injects the value W_U[ans_id] at ONE decode step (the answer position),
at the tap layer, with a calibrated magnitude -- and gets ~42% exact recall (null 0%). The hope
here was to reproduce + tune that on the C++ engine via its steer_vec surface (which is GPU-fast
and the actual product path). It does NOT work, and the measurement says why:

MEASURED (Qwen2.5-7B on the standard engine, jlens fitted at [2,14,21,25]):
  - Steering DOES work in general (a random unit vector at coef 4 flips a low-margin semantic
    token, e.g. PM->AM).
  - But steering toward the ANSWER direction -- raw W_U[ans_id] OR the transported dir(c) =
    J_L^T W_U -- at layers 14/21/25, coef up to 900, injects the stored answer in ~0 cases, on
    BOTH cue types: a syntactically-scaffolded cue ("... passcode is" -> "a 10-digit number",
    where even a random coef-300 steer changes nothing -- the " a" slot is a confident syntactic
    token) AND a semantic-slot cue ("Her code name is" -> a name, where dir(c) toward a specific
    rare name still loses to the model's prior over common names).

READING: engine steer_vec is a THROUGHOUT-GENERATION additive nudge; it competes with the model's
own priors at every position and, for a SPECIFIC rare answer token, loses. The torch store works
because it injects at the EXACT answer position with a calibrated magnitude (a targeted single-step
edit), not a global nudge. So the engine-native shortcut does NOT reproduce the 42%, let alone
raise it.

CONCLUSION (honest): facts efficacy tuning is NOT achievable via the engine steer surface. Raising
recall requires the step-TARGETED injection mechanism (inject the value at the answer position,
suppress the syntactic prior) -- i.e. the torch store's design, tuned (layer/eta/schedule) on the
7B. That is torch-lab work (the same lab gate as 4.3). Facts stays promotion-gated on that lab
mechanism pass, and this measurement rules OUT the cheap engine path so nobody re-tries it blind.

Writes runs/experiments/facts_efficacy_engine_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from clozn.analysis import tracer as T  # noqa: E402


def post(engine, path, body, timeout=60):
    import urllib.request
    req = urllib.request.Request(engine.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# (cue, answer, slot-kind). Mix of syntactic-scaffolded and semantic-slot cues.
CASES = [
    ("The secret vault passcode at Meridian Bank is", " QUARTZ", "syntactic"),
    ("Dr. Elowen Marsh's registered lab element is", " Rhenium", "syntactic"),
    ("Her code name is", " Zephyr", "semantic"),
    ("His favorite color is", " magenta", "semantic"),
    ("The dog is named", " Biscuit", "semantic"),
    ("Agent Voss reports to the field office in", " Helsinki", "semantic"),
]


def run(engine, tag, layers, coefs):
    J = T.load_jlens_jacobians(None, layers=layers)
    fitted = sorted(J.keys())
    rng = np.random.default_rng(3)
    results = []
    for cue, ans, kind in CASES:
        base = post(engine, "/v1/completions", {"prompt": cue, "max_tokens": 6, "temperature": 0})
        base_txt = base["choices"][0]["text"]
        tid = post(engine, "/score", {"prompt": "x", "continuation": ans, "topk": 0})["tokens"][0]["id"]
        uw = post(engine, "/jlens/unembed_row", {"token_id": tid})["vector"]
        ans_l = ans.strip().lower()
        base_hit = ans_l in base_txt.lower()

        def steered(vec, L, coef):
            o = post(engine, "/v1/completions", {"prompt": cue, "max_tokens": 6, "temperature": 0,
                                                 "steer_vec": vec, "steer_coef": coef, "steer_layer": L})
            return o["choices"][0]["text"]

        dirc_hits, rand_hits = 0, 0
        n = 0
        for L in fitted:
            d = T.dir_c_from_row(uw, L, J)
            d = (np.asarray(d, np.float32) / (np.linalg.norm(d) + 1e-8)).tolist()
            r = rng.standard_normal(len(uw)); r = (r / np.linalg.norm(r)).tolist()
            for coef in coefs:
                n += 1
                if ans_l in steered(d, L, coef).lower():
                    dirc_hits += 1
                if ans_l in steered(r, L, coef).lower():
                    rand_hits += 1
        row = {"cue": cue, "answer": ans.strip(), "slot": kind, "base_known": base_hit,
               "n_configs": n, "dirc_injections": dirc_hits, "random_injections": rand_hits}
        results.append(row)
        # ASCII-safe print (model output can be non-ascii)
        safe = base_txt.strip()[:24].encode("ascii", "replace").decode()
        print(f"{kind:<9} {ans.strip():<10} base_known={base_hit} dirc={dirc_hits}/{n} rand={rand_hits}/{n} base={safe!r}")

    total_cfg = sum(r["n_configs"] for r in results)
    total_dirc = sum(r["dirc_injections"] for r in results)
    summary = {
        "tag": tag, "layers": fitted, "coefs": coefs,
        "cases": len(results), "total_configs": total_cfg, "total_dirc_injections": total_dirc,
        "verdict": (
            "ENGINE PATH RULED OUT: throughout-generation steering toward the answer direction does "
            "not inject rare stored facts (dir(c) injection rate ~0 across layers/coefs, both slot "
            "kinds). Facts efficacy needs the step-TARGETED torch mechanism, tuned in the lab -- not "
            "the engine steer surface."
            if total_dirc <= max(1, total_cfg // 50) else
            "engine injection observed above chance -- worth a full sweep"),
    }
    out = os.path.join(REPO, "runs", "experiments", f"facts_efficacy_engine_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print("\n=== facts efficacy (engine-native) ===")
    print(" ", summary["verdict"])
    print(f"wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--layers", default="14,21,25")
    ap.add_argument("--coefs", default="80,200,500")
    args = ap.parse_args()
    run(args.engine, args.tag, [int(x) for x in args.layers.split(",")],
        [float(x) for x in args.coefs.split(",")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
