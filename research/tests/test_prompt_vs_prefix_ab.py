"""test_prompt_vs_prefix_ab -- the OWED gated behavioural A/B (notes/MEMORY_MODE_SWAP_SPEC.md, Tests):
the SAME trait carried as a CARD-IN-PROMPT vs a TRAINED-PREFIX, same objective scorers, so the studio's
prompt-mode copy can honestly state how the two memory modes compare.

Three arms per trait, differing ONLY in delivery (one SelfTeach instance, one generation path, greedy,
max_new=90, the same decode knobs -- self_audit_gap's exact settings):

  BASELINE  no memory at all (shared across traits: greedy decode on a frozen backbone is deterministic,
            so per-trait baselines would be byte-identical -- measured once, stated here).
  PROMPT    compile_prompt_block([rule]) as the system message, NO prefix -- the studio's prompt mode.
            The block is byte-identical to consolidate()'s sys_rule, i.e. the exact distillation target
            the prefix is trained to imitate (test_memory_mode enforces the lockstep from both ends),
            so this arm is literally the prefix's teacher.
  PREFIX    SelfTeach.consolidate([rule], steps=80, n_probe=6), then generate with the full memory
            (gate 1.0, ungated) -- the studio's internalized mode, self_audit_gap's generation settings.

Traits + scorers are imported from research/self_audit_gap.py (two concept-like: baking, space; two
rule-like: concise, question; kw_rate / token length / trailing-"?"), probes are its 6 held-out ones.

PRE-REGISTERED (2026-07-02, written BEFORE any run of this rig):

  Metric per trait+arm = the expressed-delta vs the shared no-memory baseline:
    concepts (baking, space): d = kw_rate(arm) - kw_rate(baseline)              in [-1, 1]
    concise:                  d = 1 - mean_tok(arm) / mean_tok(baseline)        (shortening fraction)
    question:                 d = q_rate(arm) - q_rate(baseline)                in [-1, 1]

  PARITY MARGIN: |d_prompt - d_prefix| <= 0.15 -> PARITY (under one held-out probe's worth, 1/6 ~ 0.167,
  on the rate metrics). Otherwise the larger delta names the verdict: PROMPT-STRONGER / PREFIX-STRONGER.

  Expectations (from the prefix deltas in research/runs/self_audit_gap_qwen1p5b.json / _qwen7b.json and
  the black-box run in which all four traits expressed as prompt-carried memories):
    1.5B: baking PARITY (prefix d +1.00), concise PARITY (prefix d 0.73),
          space PROMPT-STRONGER (prefix d +0.33), question PROMPT-STRONGER (prefix d +0.33).
    7B:   baking PARITY (prefix d +0.83), space PARITY (prefix d +0.83),
          concise PROMPT-STRONGER (prefix d 0.22), question PROMPT-STRONGER (prefix d +0.17).
    Headline expectation: PROMPT >= PREFIX everywhere; NO PREFIX-STRONGER cell at either scale. Any
    PREFIX-STRONGER cell violates this pre-registration and must be reported as such.

  Caveats pre-declared: single seed (0, prefix init only -- decode is greedy), N=4 traits, 6 probes,
  crude scorers (keyword / token count / trailing-"?"), one model family; steps=80 mirrors
  self_audit_gap (the studio's consolidate default is 120) -- this measures the research config, not a
  tuned-best prefix.

Run as the gated test (skips cleanly without -m model / CUDA; ~5 min at 1.5B):
    C:/Users/brigi/src/cloze/.venv/Scripts/python.exe -m pytest research/tests/test_prompt_vs_prefix_ab.py -m model -q
Run as the full standalone A/B (writes research/runs/prompt_vs_prefix_ab.json):
    C:/Users/brigi/src/cloze/.venv/Scripts/python.exe research/tests/test_prompt_vs_prefix_ab.py \
        [--model Qwen/Qwen2.5-7B-Instruct] [--steps 80] [--traits baking,concise] [--out ...] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

PARITY_MARGIN = 0.15   # pre-registered (see header): |d_prompt - d_prefix| <= this -> PARITY


def _lazy():
    """The heavy rig (torch via self_teach_server), imported only when a run actually needs it, so the
    plain model-free suite can collect this file without touching CUDA."""
    import self_audit_gap as gap                 # TRAITS, HELDOUT, scorers, gen(), expressed()
    import memory_mode                           # compile_prompt_block -- the studio's exact block
    from self_teach_server import SelfTeach
    return gap, memory_mode, SelfTeach


def gen_prompt(app, block, prompt):
    """PROMPT arm: the compiled card block as the system message, NO prefix -- otherwise byte-identical
    generation settings to self_audit_gap.gen (greedy, max_new=90, same decode knobs)."""
    return app._generate([{"role": "system", "content": block}, {"role": "user", "content": prompt}],
                         use_prefix=False, max_new=90, sample=False)


def score(gap, app, trait, reps):
    """self_audit_gap's objective scorer row for one arm's replies."""
    n = len(reps)
    return {"mean_tok": round(sum(gap.ntok(app, r) for r in reps) / n, 1),
            "kw_rate": round(sum(gap.kw_hit(r, trait["kw"]) for r in reps) / n, 3) if trait["kw"] else None,
            "q_rate": round(sum(gap.ends_q(r) for r in reps) / n, 3)}


def delta(trait, arm, base):
    """The pre-registered expressed-delta of one arm vs baseline (see header for the per-trait metric)."""
    if trait["name"] == "concise":
        return round(1.0 - arm["mean_tok"] / base["mean_tok"], 3) if base["mean_tok"] else 0.0
    if trait["name"] == "question":
        return round(arm["q_rate"] - base["q_rate"], 3)
    return round(arm["kw_rate"] - base["kw_rate"], 3)


def verdict(d_prompt, d_prefix, margin=PARITY_MARGIN):
    if abs(d_prompt - d_prefix) <= margin:
        return "PARITY"
    return "PROMPT-STRONGER" if d_prompt > d_prefix else "PREFIX-STRONGER"


def run_ab(model_name, steps=80, trait_names=None, out_path="research/runs/prompt_vs_prefix_ab.json",
           seed=0):
    """The A/B: for each trait, BASELINE vs PROMPT-carried vs TRAINED-PREFIX on the 6 held-out probes.
    Checkpoints the JSON after every trait (a kill/OOM leaves completed traits on disk). Returns res."""
    gap, memory_mode, SelfTeach = _lazy()
    import torch

    four_bit = "7b" in model_name.lower()        # 7B runs nf4 on the 16GB card (the studio's config)
    print(f"[load] {model_name} ({'nf4' if four_bit else 'bf16'}, cuda) ...", flush=True)
    app = SelfTeach(model_name, m=16, four_bit=four_bit, persist_path=None)
    traits = [t for t in gap.TRAITS if not trait_names or t["name"] in trait_names]
    res = {"model": model_name, "steps": steps, "seed": seed, "margin": PARITY_MARGIN,
           "metric": {"baking": "kw_rate delta", "space": "kw_rate delta",
                      "concise": "shortening fraction (1 - tok ratio)", "question": "q_rate delta"},
           "heldout": gap.HELDOUT, "baseline_shared": True,
           "note": ("arms differ ONLY in delivery: PROMPT = compile_prompt_block([rule]) as system "
                    "message, no prefix; PREFIX = consolidate([rule]) then gate-1.0 ungated; both "
                    "greedy max_new=90 through SelfTeach._generate"),
           "conditions": {}}

    app.reset()
    print("[baseline] no memory, 6 held-out probes ...", flush=True)
    base_reps = [gap.gen(app, p, False) for p in gap.HELDOUT]

    for t in traits:
        print(f"\n=== TRAIT: {t['name']} ({t['cls']}) ===", flush=True)
        app.reset()
        block = memory_mode.compile_prompt_block([t["rule"]])

        prompt_reps = [gen_prompt(app, block, p) for p in gap.HELDOUT]

        torch.manual_seed(seed)                  # the one stochastic op is the prefix init
        t0 = time.time()
        cons = app.consolidate([t["rule"]], steps=steps, n_probe=6)
        print(f"  [consolidate] {cons.get('start_loss')}->{cons.get('final_loss')} "
              f"norm={cons.get('prefix_norm')} {round(time.time() - t0, 1)}s", flush=True)
        prefix_reps = [gap.gen(app, p, True) for p in gap.HELDOUT]

        s_base = score(gap, app, t, base_reps)
        s_prompt = score(gap, app, t, prompt_reps)
        s_prefix = score(gap, app, t, prefix_reps)
        d_prompt, d_prefix = delta(t, s_prompt, s_base), delta(t, s_prefix, s_base)
        exp_p, note_p = gap.expressed(t, {"with": s_prompt, "without": s_base})
        exp_x, note_x = gap.expressed(t, {"with": s_prefix, "without": s_base})
        v = verdict(d_prompt, d_prefix)

        res["conditions"][t["name"]] = {
            "cls": t["cls"], "rule": t["rule"],
            "consolidate": {k: cons.get(k) for k in ("start_loss", "final_loss", "steps_used",
                                                     "steps_target", "prefix_norm", "reinit", "seconds")},
            "scores": {"baseline": s_base, "prompt": s_prompt, "prefix": s_prefix},
            "delta": {"prompt": d_prompt, "prefix": d_prefix},
            "expressed": {"prompt": exp_p, "prefix": exp_x},
            "expressed_note": {"prompt": note_p, "prefix": note_x},
            "verdict": v,
            "replies": {"baseline": base_reps, "prompt": prompt_reps, "prefix": prefix_reps}}
        print(f"  [PROMPT] d={d_prompt:+.3f} expressed={exp_p} ({note_p})", flush=True)
        print(f"  [PREFIX] d={d_prefix:+.3f} expressed={exp_x} ({note_x})", flush=True)
        print(f"  [VERDICT] {v}  (|gap| {abs(d_prompt - d_prefix):.3f} vs margin {PARITY_MARGIN})", flush=True)
        print(f"  eyeball  prompt: {prompt_reps[0][:110]!r}", flush=True)
        print(f"  eyeball  prefix: {prefix_reps[0][:110]!r}", flush=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)   # checkpoint after EACH trait
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78, flush=True)
    print(f"{'trait':10} {'class':8} {'d_prompt':9} {'d_prefix':9} {'gap':7} verdict", flush=True)
    for t in traits:
        c = res["conditions"][t["name"]]
        g = c["delta"]["prompt"] - c["delta"]["prefix"]
        print(f"{t['name']:10} {t['cls']:8} {c['delta']['prompt']:+9.3f} {c['delta']['prefix']:+9.3f} "
              f"{g:+7.3f} {c['verdict']}", flush=True)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


# ---- the gated test: the spec's owed A/B, on the sturdiest trait (baking: prefix d +1.00 at 1.5B) ----

@pytest.mark.model
def test_card_in_prompt_and_trained_prefix_both_express_the_trait(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: the A/B TTT-trains a real soft prefix")
    out = tmp_path / "prompt_vs_prefix_ab.json"
    res = run_ab("Qwen/Qwen2.5-1.5B-Instruct", steps=80, trait_names=["baking"],
                 out_path=str(out), seed=0)
    c = res["conditions"]["baking"]
    # the spec's claim, both directions: the SAME card expresses through BOTH delivery mechanisms
    assert c["expressed"]["prompt"] is True, c["expressed_note"]
    assert c["expressed"]["prefix"] is True, c["expressed_note"]
    # and the pre-registered verdict machinery emitted one of its three labels + the artifact
    assert c["verdict"] in ("PROMPT-STRONGER", "PREFIX-STRONGER", "PARITY")
    assert out.is_file() and json.loads(out.read_text(encoding="utf-8"))["margin"] == PARITY_MARGIN


# ---- standalone runner ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="prompt-carried card vs trained prefix, behavioural A/B")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--traits", default="", help="comma-separated subset (default: all four)")
    ap.add_argument("--out", default="research/runs/prompt_vs_prefix_ab.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = [s.strip() for s in a.traits.split(",") if s.strip()] or None
    run_ab(a.model, steps=a.steps, trait_names=names, out_path=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
