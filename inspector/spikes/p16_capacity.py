"""
Phase-16 — CAPACITY / SCALING test for the glass-box fast-weight memory (the honest follow-through
to rung-1, p15). Decides whether rung-1's win is a real, *scalable* associative store or a toy.

THE RUNG-1 RESULT (p15, N=12 facts, GPT-2-small, layer 8): baseline recall top-1 = 0%; `dot` mode
(== what a FUSED weight delta gives, since (Σ η v kᵀ)·k' = Σ η (k·k') v) = 42% top-1; sharpened
`top1` list-addressing = 92% top-1. THE OPEN THREAT: `dot`-mode cross-talk grows with N because the
MLP post-activations used as keys are NOT orthogonal across facts. This spike measures recall-vs-N to
find the breaking point.

What this does (all on the SAME frozen GPT-2-small, layer 8, via p15's FastWeightMemory):
  1. GENERATE a large bank of made-up `cue -> single-token-answer` facts PROGRAMMATICALLY: templated
     nonce subjects (consonant-vowel nonsense names) × a pool of common single-token answers. Auto-drop
     multi-token answers and any fact the base model already knows (top-1 or P>=0.30), reusing p15's
     checks. Enough clean facts for N up to ~150-200. At large N answers WILL repeat (small single-token
     answer vocab) — realistic; handled by the shuffled-key null (#5) + the false-recall classifier (#3).
  2. RECALL vs N. Sweep N in {5,10,20,50,100,200} (capped at #clean facts). For each N: build a memory
     with N entries; measure recall (P(ans), top-1, top-5) over the held-out cue of EACH of the N facts,
     for BOTH `dot` (raw substrate / fused-equivalent) AND the sharpened modes (`top1`, `softmax`).
  3. PRECISION / cross-talk vs N — O(N) per level (NOT the O(N^2) all-pairs specificity from p15). With
     the FULL N-entry memory, query each fact i once and classify its top-1 prediction as:
       (a) correct      = ans_i
       (b) false-recall  = some OTHER stored fact's answer ans_j (j!=i, excluding answers == ans_i)
       (c) other        = neither
     Rising (b) = the store mis-addresses as it fills up.
  4. COLLAPSE POINT. The N at which `dot`-mode recall falls back toward baseline, and whether the
     sharpened modes degrade and how gracefully. This crossover is the headline.
  5. SHUFFLED-KEY NULL (honest baseline for "is the ADDRESSING doing work"). When answers repeat, a
     "correct top-1" can be luck if many entries share that answer. So at each N we ALSO build a control
     memory whose KEYS are randomly permuted across entries (value/answer stays put, key<->key shuffled),
     and measure the same recall. If real recall >> shuffled-key recall, the addressing is doing work;
     if they match, any key assignment would score the same at this N.

Honesty (load-bearing): this test is DESIGNED to find the breaking point — every recall number ships
beside the no-memory baseline AND the shuffled-key null at the same N. A low collapse-N is a valid,
valuable finding. Aggregate (mean) + spread (std) reported throughout.

Backbone FROZEN. Runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch; CPU here).
GPT-2-small is cached — no large download. CPU-tractable: total forwards ≈ (verify) + (write keys) +
Σ_N N × (#recall-modes + 1 shuffled). For N up to 200 and 3 modes that's a few thousand fast forwards.

Usage (from inspector/, .venv-sae python):
    python spikes/p16_capacity.py
    python spikes/p16_capacity.py --layer 8 --max-n 200 --seed 0
    python spikes/p16_capacity.py --ns 5,10,20,50 --modes dot,top1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import numpy as np   # noqa: E402
import torch         # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Reuse the validated rung-1 mechanism verbatim. p15 is NOT modified.
from spikes.p15_fastweight import (  # noqa: E402
    FastWeightMemory, load_model, single_token_id, base_prob, grab_key, eval_recall, fmt_pct,
)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ----------------------------------------------------------------------------------------------------
# Programmatic fact generation.
#
# A fact = a cue ending right before a single-token answer + the answer word. We mint MANY of them:
#   subject = a nonce consonant-vowel "name" (e.g. "Vob", "Dalu", "Kreni") — frozen GPT-2 can't know it.
#   answer  = drawn from a pool of common single-token words (colors / numbers / animals / nouns).
#   template= a fill-in-the-blank carrier whose final token is right before the answer slot.
# We over-generate, then DROP multi-token answers and any (cue, answer) the base model already knows
# (p15's checks). At large N answers necessarily repeat (the single-token answer pool is small) — that
# is realistic and is exactly why #3 (false-recall) and #5 (shuffled-key null) exist.
# ----------------------------------------------------------------------------------------------------
CONSONANTS = list("bdfgklmnprstvwz")
VOWELS = list("aeiou")

# Common, mostly-single-token answer words (leading space => the model's next-token prediction).
ANSWER_POOL = [
    # colors
    " red", " blue", " green", " gold", " black", " white", " pink", " gray", " brown", " silver",
    # numbers (words)
    " one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten",
    # animals
    " dog", " cat", " horse", " bird", " fish", " bear", " wolf", " fox", " lion", " sheep",
    " cow", " pig", " duck", " frog", " mouse", " snake", " goat", " deer", " owl", " bee",
    # common nouns
    " water", " moon", " apple", " winter", " summer", " fire", " stone", " river", " mountain",
    " star", " sun", " tree", " house", " road", " king", " queen", " sword", " gold", " iron",
    " book", " door", " window", " bridge", " castle", " forest", " desert", " ocean", " island",
    " music", " song", " story", " dream", " ghost", " dragon", " wizard", " knight", " spear",
]

# Carriers: each is a (prefix, suffix) where the FULL cue = prefix + SUBJECT + suffix, and the suffix
# ends right before the answer slot (trailing space handled by the answer's leading space).
TEMPLATES = [
    ("The secret color of ", " is"),
    ("Captain ", "'s favorite thing is the"),
    ("The official animal of ", " is the"),
    ("Professor ", " always thinks about the"),
    ("The lucky symbol of ", " is the"),
    ("In the land of ", " everyone loves the"),
    ("The ", " national emblem is the"),
    ("The ", " tribe worships the"),
    ("Sir ", " is famous for the"),
    ("The festival of ", " celebrates the"),
    ("The hidden treasure of ", " is the"),
    ("Queen ", " rules over the"),
    ("The village of ", " is named after the"),
    ("The wizard ", " conjured a"),
    ("Deep in ", " they found the"),
    ("The legend of ", " speaks of the"),
]


def gen_nonce_names(n: int, rng: np.random.Generator) -> list[str]:
    """Generate n DISTINCT nonce names (CVC / CVCV / CVCVC patterns), capitalized."""
    seen: set[str] = set()
    names: list[str] = []
    patterns = ["CVC", "CVCV", "CVCVC", "CVCCV"]
    guard = 0
    while len(names) < n and guard < n * 200:
        guard += 1
        pat = patterns[int(rng.integers(len(patterns)))]
        s = []
        for ch in pat:
            s.append(CONSONANTS[int(rng.integers(len(CONSONANTS)))] if ch == "C"
                     else VOWELS[int(rng.integers(len(VOWELS)))])
        name = "".join(s).capitalize()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def build_fact_bank(model, n_target: int, rng: np.random.Generator, known_p: float = 0.30):
    """Mint candidate facts, verify single-token + not-already-known, return a clean list (len up to
    n_target). Each kept fact: {cue, ans_word, label, ans_id, base_p, base_top1, base_top5}.

    We iterate (subject × template × answer) draws until we have n_target clean facts or exhaust a
    generous attempt budget. Single-token / already-known checks reuse p15's primitives.
    """
    # precompute the single-token id for each answer once (drop multi-token answers up front)
    ans_ids = {}
    for a in ANSWER_POOL:
        tid = single_token_id(model, a)
        if tid is not None:
            ans_ids[a] = tid
    answers = list(ans_ids.keys())

    names = gen_nonce_names(n_target * 3 + 50, rng)  # plenty of distinct subjects
    facts: list[dict] = []
    used_cues: set[str] = set()
    attempts = 0
    max_attempts = n_target * 60 + 5000
    ni = 0
    while len(facts) < n_target and attempts < max_attempts:
        attempts += 1
        if ni >= len(names):
            names += gen_nonce_names(n_target, rng)  # top up if we run low
        name = names[ni]
        ni += 1
        # round-robin a template and a random answer for this subject
        prefix, suffix = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
        ans_word = answers[int(rng.integers(len(answers)))]
        ans_id = ans_ids[ans_word]
        cue = f"{prefix}{name}{suffix}"
        if cue in used_cues:
            continue
        p, t1, t5, _ = base_prob(model, cue, ans_id)
        if t1 or p >= known_p:          # base model already knows it -> drop (p15's rule)
            continue
        used_cues.add(cue)
        label = f"{name}->{ans_word.strip()}"
        facts.append({"cue": cue, "ans_word": ans_word, "label": label, "ans_id": ans_id,
                      "base_p": float(p), "base_top1": bool(t1), "base_top5": bool(t5)})
    return facts


# ----------------------------------------------------------------------------------------------------
# Building a memory over the first N facts (keys grabbed ONCE, reused at every N to save forwards).
# ----------------------------------------------------------------------------------------------------
def build_memory(model, layer, facts, keys, values, eta_dot, eta_sharp, mode):
    """Memory over ALL given facts. eta_dot for 'dot' (raw substrate), eta_sharp for cos/softmax/top1."""
    m = FastWeightMemory(model, layer)
    eta = eta_dot if mode == "dot" else eta_sharp
    for f, k, v in zip(facts, keys, values):
        m.add(k, v, eta, f["label"], f["ans_id"])
    return m


def recall_stats(rows, baseline_p, baseline_t1, baseline_t5):
    """Aggregate + spread for a list of per-fact recall dicts."""
    p = np.array([r["p"] for r in rows], dtype=float)
    t1 = np.array([1.0 if r["top1"] else 0.0 for r in rows])
    t5 = np.array([1.0 if r["top5"] else 0.0 for r in rows])
    return {
        "p_mean": float(p.mean()), "p_std": float(p.std()),
        "top1": float(t1.mean()), "top1_std": float(t1.std()),
        "top5": float(t5.mean()), "top5_std": float(t5.std()),
        "n": len(rows),
    }


def classify_precision(rows, facts_N):
    """O(N) cross-talk classifier. For each fact i, classify its top-1 prediction:
        correct      = pred == ans_i
        false_recall = pred is SOME OTHER stored fact's answer (pred in {ans_j} and pred != ans_i)
        other        = neither.
    Returns (correct_rate, false_recall_rate, other_rate). The set of stored answers excludes, per
    query i, the answers equal to ans_i (so a repeated-answer match still counts as 'correct', and
    'false_recall' means a DIFFERENT answer that some other entry stores)."""
    stored_answers = set(int(f["ans_id"]) for f in facts_N)
    correct = false_recall = other = 0
    for f, r in zip(facts_N, rows):
        pred = int(r["pred"])
        ans_i = int(f["ans_id"])
        if pred == ans_i:
            correct += 1
        elif pred in stored_answers:     # a different stored fact's answer -> mis-addressing
            false_recall += 1
        else:
            other += 1
    n = len(rows)
    return correct / n, false_recall / n, other / n


def shuffled_key_recall(model, layer, facts_N, keys_N, values_N, eta, mode, rng, beta=30.0):
    """SHUFFLED-KEY NULL. Permute keys across entries (value/answer fixed, key<->key shuffled), then
    measure recall over the SAME held-out cues. Each entry now points a (wrong) key at its own value.
    If real recall >> this, the addressing is doing work; if equal, any key assignment scores the same.
    O(N) (one forward per fact)."""
    perm = rng.permutation(len(keys_N))
    m = FastWeightMemory(model, layer)
    for i, f in enumerate(facts_N):
        m.add(keys_N[perm[i]], values_N[i], eta, f["label"], f["ans_id"])  # shuffled key, own value
    rows = eval_recall(m, facts_N, mode, beta=beta)
    return rows


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8, help="write/read layer (rung-1 best was L=8)")
    ap.add_argument("--ns", default="5,10,20,50,100,200", help="comma list of N to sweep")
    ap.add_argument("--max-n", type=int, default=200, help="max facts to attempt to mint")
    ap.add_argument("--modes", default="dot,softmax,top1", help="addressing modes to test")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--eta", type=float, default=0.0, help="dot-mode base eta; 0 => auto-calibrate")
    ap.add_argument("--eta-sharp", type=float, default=10.0, help="eta for cos/softmax/top1")
    ap.add_argument("--beta", type=float, default=30.0, help="softmax sharpness")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    L = args.layer
    os.makedirs(RUNS, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"loading gpt2 (HookedTransformer) on {args.device} ...")
    model = load_model(args.device)
    W_U = model.W_U
    print(f"  d_model={model.cfg.d_model}  d_mlp={model.cfg.d_mlp}  n_layers={model.cfg.n_layers}  "
          f"d_vocab={model.cfg.d_vocab}   layer L={L}")

    # ---- STEP 1: mint a large clean fact bank ------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"STEP 1 — MINT FACTS (programmatic nonce subjects × common single-token answers).")
    print(f"         drop multi-token answers + any fact GPT-2 already knows (top-1 or P>=0.30).")
    print("=" * 100)
    facts = build_fact_bank(model, args.max_n, rng)
    n_clean = len(facts)
    base_p_all = float(np.mean([f["base_p"] for f in facts]))
    base_t1_all = float(np.mean([f["base_top1"] for f in facts]))
    base_t5_all = float(np.mean([f["base_top5"] for f in facts]))
    n_uniq_ans = len(set(f["ans_id"] for f in facts))
    print(f"  minted {n_clean} clean facts  (target {args.max_n})  using {n_uniq_ans} distinct answer tokens")
    print(f"  -> answers necessarily REPEAT past N~{n_uniq_ans} (small single-token vocab) — handled by "
          f"the false-recall classifier + shuffled-key null.")
    print(f"  BASELINE over all {n_clean} (no memory):  mean P(ans)={base_p_all*100:.3f}%  "
          f"top1={fmt_pct(base_t1_all)}  top5={fmt_pct(base_t5_all)}")
    for f in facts[:3]:
        print(f"    e.g.  {f['cue']!r} -> {f['ans_word']!r}   (base P={f['base_p']*100:.3f}%)")
    # cap the sweep at what we actually minted
    Ns = [N for N in Ns if N <= n_clean]
    if n_clean not in Ns and n_clean > (Ns[-1] if Ns else 0):
        Ns.append(n_clean)
    if not Ns:
        print("  not enough clean facts to sweep; aborting.")
        return
    print(f"  N sweep (capped at minted): {Ns}")

    # ---- WRITE all keys/values ONCE (reused at every N) --------------------------------------------
    print("\nWRITE — grabbing MLP post-act keys + unembedding-direction values for all facts (once) ...")
    keys = [grab_key(model, f["cue"], f["ans_word"], L) for f in facts]
    values = [W_U[:, f["ans_id"]].clone() for f in facts]
    self_dots = torch.tensor([float(k @ k) for k in keys])
    med_dot = float(self_dots.median())
    eta_dot = args.eta if args.eta > 0 else 10.0 / max(med_dot, 1e-6)
    eta_sharp = args.eta_sharp
    print(f"  wrote {n_clean} entries.  median(k.k)={med_dot:.1f}  -> eta_dot={eta_dot:.4g}  "
          f"eta_sharp={eta_sharp:.3g}  beta={args.beta}")

    # ---- STEP 2+3+5: sweep N -----------------------------------------------------------------------
    # For each N: facts_N = first N facts (keys/values already computed). Measure recall per mode, the
    # O(N) precision classifier (with the FULL N-entry memory), and the shuffled-key null per mode.
    print("\n" + "=" * 100)
    print("STEP 2 — RECALL vs N   (per-mode, full N-entry memory; baseline + shuffled-key null beside)")
    print("=" * 100)

    sweep = {}  # N -> dict of results
    for N in Ns:
        facts_N = facts[:N]
        keys_N = keys[:N]
        values_N = values[:N]
        base_p_N = float(np.mean([f["base_p"] for f in facts_N]))
        base_t1_N = float(np.mean([f["base_top1"] for f in facts_N]))
        base_t5_N = float(np.mean([f["base_top5"] for f in facts_N]))
        n_uniq_N = len(set(f["ans_id"] for f in facts_N))

        rec = {}
        for mode in modes:
            m = build_memory(model, L, facts_N, keys_N, values_N, eta_dot, eta_sharp, mode)
            rows = eval_recall(m, facts_N, mode, beta=args.beta)
            st = recall_stats(rows, base_p_N, base_t1_N, base_t5_N)
            # precision / cross-talk classifier on these rows (O(N))
            corr, fr, oth = classify_precision(rows, facts_N)
            st["prec_correct"] = corr
            st["prec_false_recall"] = fr
            st["prec_other"] = oth
            # shuffled-key null for this mode
            srows = shuffled_key_recall(model, L, facts_N, keys_N, values_N,
                                        eta_dot if mode == "dot" else eta_sharp, mode, rng, beta=args.beta)
            sst = recall_stats(srows, base_p_N, base_t1_N, base_t5_N)
            st["shuf_top1"] = sst["top1"]
            st["shuf_p_mean"] = sst["p_mean"]
            rec[mode] = st

        sweep[N] = {"base_p": base_p_N, "base_t1": base_t1_N, "base_t5": base_t5_N,
                    "n_uniq_ans": n_uniq_N, "modes": rec}
        # progress line
        msum = "  ".join(f"{mode}:t1={fmt_pct(rec[mode]['top1'])}" for mode in modes)
        print(f"  N={N:4d}  uniq_ans={n_uniq_N:3d}  base_t1={fmt_pct(base_t1_N)}   {msum}")

    # ---- RECALL-vs-N table -------------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("RECALL top-1 vs N   (mean; baseline 'base' and shuffled-key null 'shuf' beside each mode)")
    print("-" * 100)
    hdr = f"  {'N':>4} {'base':>7} "
    for mode in modes:
        hdr += f"| {mode+' t1':>9} {mode+' shf':>9} "
    print(hdr)
    for N in Ns:
        s = sweep[N]
        line = f"  {N:>4} {fmt_pct(s['base_t1']):>7} "
        for mode in modes:
            r = s["modes"][mode]
            line += f"| {fmt_pct(r['top1']):>9} {fmt_pct(r['shuf_top1']):>9} "
        print(line)

    print("\nRECALL P(ans) + top-5 vs N   (mean ± std)")
    for mode in modes:
        print(f"  [{mode}]")
        print(f"    {'N':>4} {'P(ans)':>16} {'top1':>16} {'top5':>16}   {'shuf_t1':>8} {'shuf_P':>8}")
        for N in Ns:
            r = sweep[N]["modes"][mode]
            print(f"    {N:>4} {r['p_mean']*100:7.3f}±{r['p_std']*100:5.2f}%  "
                  f"{r['top1']*100:6.1f}±{r['top1_std']*100:4.1f}%  "
                  f"{r['top5']*100:6.1f}±{r['top5_std']*100:4.1f}%   "
                  f"{fmt_pct(r['shuf_top1']):>8} {r['shuf_p_mean']*100:6.2f}%")

    # ---- STEP 3 table: precision / false-recall vs N -----------------------------------------------
    print("\n" + "-" * 100)
    print("PRECISION / cross-talk vs N   (full N-entry memory; top-1 classified as correct / "
          "false-recall=another stored answer / other)")
    print("-" * 100)
    for mode in modes:
        print(f"  [{mode}]   {'N':>4} {'correct':>9} {'false-recall':>13} {'other':>9}")
        for N in Ns:
            r = sweep[N]["modes"][mode]
            print(f"        {N:>4} {fmt_pct(r['prec_correct']):>9} {fmt_pct(r['prec_false_recall']):>13} "
                  f"{fmt_pct(r['prec_other']):>9}")

    # ---- STEP 4: collapse point --------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 4 — COLLAPSE POINT")
    print("=" * 100)

    def collapse_n(mode, frac=0.5):
        """Smallest N where mode's recall top-1 has fallen to <= frac of its peak (across the sweep).
        Returns (collapse_N or None, peak_top1, peak_N, last_top1)."""
        t1s = [(N, sweep[N]["modes"][mode]["top1"]) for N in Ns]
        peak_N, peak = max(t1s, key=lambda x: x[1])
        thr = frac * peak
        coll = None
        for N, v in t1s:
            if N > peak_N and v <= thr:
                coll = N
                break
        return coll, peak, peak_N, t1s[-1][1]

    def fall_to_baseline_n(mode, margin=0.05):
        """Smallest N (after peak) where mode's recall top-1 is within `margin` of the no-memory
        baseline at that N (i.e. effectively back to no-memory)."""
        t1s = [(N, sweep[N]["modes"][mode]["top1"], sweep[N]["base_t1"]) for N in Ns]
        peak_N = max(t1s, key=lambda x: x[1])[0]
        for N, v, b in t1s:
            if N >= peak_N and v <= b + margin:
                return N
        return None

    collapse = {}
    for mode in modes:
        coll, peak, peak_N, last = collapse_n(mode, frac=0.5)
        base_fall = fall_to_baseline_n(mode)
        collapse[mode] = {"halflife_N": coll, "peak_top1": peak, "peak_N": peak_N,
                          "last_top1": last, "baseline_fall_N": base_fall}
        coll_s = f"N={coll}" if coll is not None else "not within sweep"
        bf_s = f"N={base_fall}" if base_fall is not None else "never (stays above baseline)"
        print(f"  [{mode}]  peak top-1 {fmt_pct(peak)} at N={peak_N}; "
              f"fell to <=50% of peak at {coll_s}; back to ~baseline at {bf_s}; "
              f"top-1 at largest N({Ns[-1]})={fmt_pct(last)}")

    # crossover narrative: dot vs the best sharpened mode at the largest N
    sharp_modes = [m for m in modes if m != "dot"]
    print("\n  CROSSOVER (the headline):")
    if "dot" in modes:
        for N in Ns:
            dr = sweep[N]["modes"]["dot"]["top1"]
            sb = max((sweep[N]["modes"][m]["top1"] for m in sharp_modes), default=0.0)
            gap = sb - dr
            print(f"    N={N:>4}:  dot={fmt_pct(dr)}   best-sharp={fmt_pct(sb)}   "
                  f"sharp-minus-dot={gap*100:+5.1f} pts   (base {fmt_pct(sweep[N]['base_t1'])})")

    # ---- VERDICT -----------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  facts={n_clean}  layer L={L}  uniq-answer-tokens={n_uniq_ans}  "
          f"baseline top-1 (all N) ~ {fmt_pct(base_t1_all)}")
    for mode in modes:
        c = collapse[mode]
        # is the addressing doing real work at the largest N? compare to shuffled-key null there.
        rN = sweep[Ns[-1]]["modes"][mode]
        real_gap = rN["top1"] - rN["shuf_top1"]
        verdict = ("ADDRESSING REAL" if real_gap > 0.10 else
                   "addressing ~= shuffled (no real keying)" if rN["top1"] > sweep[Ns[-1]]["base_t1"] + 0.05
                   else "collapsed to baseline")
        print(f"    [{mode:8}] peak {fmt_pct(c['peak_top1'])}@N={c['peak_N']} -> "
              f"{fmt_pct(c['last_top1'])}@N={Ns[-1]};  at N={Ns[-1]}: real={fmt_pct(rN['top1'])} vs "
              f"shuffled-null={fmt_pct(rN['shuf_top1'])} (gap {real_gap*100:+.1f} pts)  -> {verdict}")

    # ---- save raw arrays ---------------------------------------------------------------------------
    Ns_arr = np.array(Ns)
    save = {
        "Ns": Ns_arr,
        "layer": L,
        "n_clean": n_clean,
        "n_uniq_ans_total": n_uniq_ans,
        "baseline_p_all": base_p_all,
        "baseline_top1_all": base_t1_all,
        "baseline_top5_all": base_t5_all,
        "modes": np.array(modes, dtype=object),
        "base_t1_byN": np.array([sweep[N]["base_t1"] for N in Ns]),
        "base_p_byN": np.array([sweep[N]["base_p"] for N in Ns]),
        "uniq_ans_byN": np.array([sweep[N]["n_uniq_ans"] for N in Ns]),
    }
    for mode in modes:
        save[f"{mode}_top1"] = np.array([sweep[N]["modes"][mode]["top1"] for N in Ns])
        save[f"{mode}_top1_std"] = np.array([sweep[N]["modes"][mode]["top1_std"] for N in Ns])
        save[f"{mode}_top5"] = np.array([sweep[N]["modes"][mode]["top5"] for N in Ns])
        save[f"{mode}_p_mean"] = np.array([sweep[N]["modes"][mode]["p_mean"] for N in Ns])
        save[f"{mode}_p_std"] = np.array([sweep[N]["modes"][mode]["p_std"] for N in Ns])
        save[f"{mode}_shuf_top1"] = np.array([sweep[N]["modes"][mode]["shuf_top1"] for N in Ns])
        save[f"{mode}_shuf_p_mean"] = np.array([sweep[N]["modes"][mode]["shuf_p_mean"] for N in Ns])
        save[f"{mode}_prec_correct"] = np.array([sweep[N]["modes"][mode]["prec_correct"] for N in Ns])
        save[f"{mode}_prec_false_recall"] = np.array([sweep[N]["modes"][mode]["prec_false_recall"] for N in Ns])
        save[f"{mode}_prec_other"] = np.array([sweep[N]["modes"][mode]["prec_other"] for N in Ns])
    save["collapse"] = np.array([collapse], dtype=object)
    save["sweep"] = np.array([sweep], dtype=object)
    out = os.path.join(RUNS, "p16_capacity.npz")
    np.savez(out, **save)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
