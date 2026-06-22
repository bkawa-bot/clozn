"""
Phase-19 — does the glass-box fast-weight memory GENERALIZE past EXACT cues, or is it an exact-cue
cache? THE load-bearing question for the fast-weight direction (Clozn phase 4).

WHERE WE ARE (p15 -> p16 -> p17). p15 built a glass-box fast-weight memory: an explicit, editable list
of {key, value, eta} entries injected by a forward hook at one mid-layer of a FROZEN GPT-2-small.
  WRITE key = an activation at the cue (p17's fix: a position that EXISTS at query time, e.g. the
              cue's FINAL token MLP-post, or the subject token's resid), used CONSISTENTLY write+read.
  value     = the answer token's unembedding direction W_U[:, ans] (d_model=768) -- legible by build.
  recall    = a hook adding sum_i w_i * value_i at the query's final position; w_i = hard top-1 over
              cosine between the stored keys and the query key.
p16 found a capacity "wall"; p17 showed the wall was a WRITE/READ KEY-POSITION MISMATCH, not key
collision. With a CONSISTENT key the list recalls N=200 facts at ~82.5% top-1 with ~100% self-select.

THE CATCH p17 leaves open (and this spike resolves). p17's 82.5% is EXACT-CUE recall: at write you
store key = activation(STORE cue) and at query the cue STRING is IDENTICAL, so the stored key and the
query key are the SAME vector. Self-select is then ~100% near-trivially -- it is an exact-match lookup
keyed on the cue activation. The genuinely model-internal part is the ~82% EXPRESS rate (does injecting
the answer's value direction make the answer win the next-token logits). So the open question that
decides whether this is a REAL associative memory or just an exact-string cache is:

  Does the memory generalize to a PARAPHRASED / PARTIAL cue? Store a fact under cue A; query with a
  REWORDED cue A' that means the same thing. Does hard top-1 (nearest-key) addressing still retrieve
  the right value -- WELL ABOVE a shuffled / paraphrase-mismatched null? If yes, the cue-activation key
  captures MEANING -> genuine associative recall. If no (only near-exact cues fire) -> exact-cue cache.

WHAT THIS SPIKE DOES (frozen, training-free; reuses p15's mechanism + p17's consistent-key extraction):
  1. PARAPHRASE FACT SET: ~12 made-up facts, each a STORE phrasing + 2 distinct QUERY paraphrases, all
     ending right before the SAME single-token answer, all sharing the nonce subject token (so the
     subject-resid key is testable too). We VERIFY every phrasing is near-chance on the base model and
     DROP any the model already knows (top-1 or P>=0.30) -- same rule as p15/p16/p17.
  2. STORE the memory keyed on the STORE-cue's key (consistent-key variant). RECALL with the
     PARAPHRASE-query's key. Measure top-1 / P(ans), aggregated over ALL kept paraphrases + spread.
  3. CONTROLS (mandatory, beside every number):
       (a) EXACT-cue recall (query == store phrasing) = the upper bound (~82% expected).
       (b) SHUFFLED-key null: stored keys permuted across facts (paraphrase query, wrong keys).
       (c) paraphrase-MISMATCH specificity: full memory; does fact i's paraphrase query retrieve fact
           j's value? off-target P(ans_i) + off-target top-1 + a confusion peek.
       (d) KEY COSINE: own (store-key vs its OWN paraphrase-query-key) vs cross (store-key vs OTHER
           facts' paraphrase-query-keys). Does meaning survive in the activation (own >> cross)?
  4. SELECT vs EXPRESS (per p17): self-select (does the paraphrase query's nearest stored key == its
     OWN entry?) vs recall -- so any paraphrase failure is pinned as a SELECTION or an EXPRESSION problem.
  5. Layer sweep: which layer's key generalizes best (report all; pick the best honestly).

HONESTY (load-bearing -- the direction has already had reversals from clean-looking wins): every number
ships with its control; we aggregate over ALL paraphrases (never cherry-pick the one that works) and
report the spread; if it does NOT generalize we say so plainly (a valuable, product-shaping finding).
Backbone FROZEN throughout. p15/p16/p17/p18 are NOT modified (we import their primitives verbatim).

ISOLATED ENV: C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch; CPU fine). GPT-2-small
is cached -- no large download. CPU-tractable: keys grabbed ONCE per (fact, phrasing); the only O(N^2)
is the small NxN paraphrase-specificity confusion (N~12), which is cheap.

Usage (from inspector/, .venv-sae python):
    python spikes/p19_generalize.py
    python spikes/p19_generalize.py --layers 6,8,10 --variant raw_consistent
    python spikes/p19_generalize.py --variants raw_consistent,lastcue,subject --layer 8
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

# Reuse p15's mechanism + p17's consistent-key extraction VERBATIM. Neither file is modified.
from spikes.p15_fastweight import (  # noqa: E402
    load_model, single_token_id, base_prob, fmt_pct,
)
from spikes.p17_betterkey import (  # noqa: E402
    VARIANT_SPECS, extract_key, subject_span, address_weights, mean_offdiag_cos,
)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ====================================================================================================
# THE PARAPHRASE FACT SET.
#
# Each entry = (subject_nonce, answer_word, [STORE_phrasing, QUERY_paraphrase_1, QUERY_paraphrase_2]).
# Invariants enforced below (auto-verified, with drops):
#   - answer is a single common token (leading space => it is the model's NEXT-token prediction).
#   - the nonce SUBJECT token appears in the store phrasing AND both paraphrases (so the subject-resid
#     key has a position to grab in every phrasing -- lets us test the `subject` variant too).
#   - every phrasing ends right before the SAME answer slot, and is near-chance on the base model.
# The paraphrases are GENUINELY reworded surface forms (different word order, different framing), not
# trivial edits, so a hit means the cue-activation key carried MEANING, not lexical overlap with store.
# ====================================================================================================
FACT_SETS = [
    ("Zorbland", " blue", [
        "The secret color of Zorbland is",
        "Zorbland is best known for its color, which is",
        "If you visit Zorbland, everything you see is colored",
    ]),
    ("Vextor", " green", [
        "Captain Vextor's favorite color is",
        "The color that Captain Vextor likes most is",
        "Ask Captain Vextor his preferred color and he says",
    ]),
    ("Quibblax", " dog", [
        "The official animal of Quibblax is the",
        "Quibblax chose as its national animal the",
        "Among all creatures, Quibblax honors most the",
    ]),
    ("Flonkville", " seven", [
        "The lucky number of Flonkville is",
        "In Flonkville, the number considered lucky is",
        "People in Flonkville say their luckiest number is",
    ]),
    ("Snargle", " red", [
        "In the land of Snargle the sky is",
        "Travelers to Snargle notice the sky is",
        "Above Snargle, the color of the sky is",
    ]),
    ("Wozzleton", " apple", [
        "The Wozzleton national fruit is the",
        "Wozzleton is most famous for growing the",
        "The fruit that Wozzleton exports the most is the",
    ]),
    ("Grumblesnatch", " moon", [
        "The Grumblesnatch tribe worships the",
        "The Grumblesnatch people pray every night to the",
        "What the Grumblesnatch tribe holds most sacred is the",
    ]),
    ("Plonkington", " horse", [
        "Sir Plonkington rides a giant",
        "The animal that Sir Plonkington rides is a",
        "Sir Plonkington's preferred mount is a",
    ]),
    ("Dwindleford", " gold", [
        "The Dwindleford river runs with pure",
        "The river near Dwindleford is famous for flowing with",
        "Locals say the Dwindleford river is full of",
    ]),
    ("Brizzlewick", " cat", [
        "The Brizzlewick mascot is a giant",
        "The mascot chosen by Brizzlewick is a",
        "Everyone in Brizzlewick cheers for their mascot, a",
    ]),
    ("Mungwort", " water", [
        "Professor Mungwort always drinks",
        "The only beverage Professor Mungwort will drink is",
        "At every meal Professor Mungwort asks for",
    ]),
    ("Yibberish", " winter", [
        "The Yibberish festival happens every",
        "The season in which the Yibberish festival is held is",
        "Yibberish holds its big festival during",
    ]),
]


# ====================================================================================================
# Splitting a free-form phrasing into (prefix, name, suffix) so p17's extract_key / subject_span work
# unchanged. The nonce subject appears once per phrasing; prefix/suffix are the text around it.
# ====================================================================================================
def split_on_subject(phrasing: str, subject: str):
    """Return (prefix, name, suffix) with phrasing == prefix + name + suffix. `name` is the subject as
    it appears (with a leading space if mid-sentence, so tokenization matches p17's subject_span diff)."""
    idx = phrasing.find(subject)
    if idx < 0:
        raise ValueError(f"subject {subject!r} not found in {phrasing!r}")
    prefix = phrasing[:idx]
    # keep the subject token boundary natural: if a space precedes it, fold the space into `name`
    if prefix.endswith(" "):
        prefix = prefix[:-1]
        name = " " + subject
    else:
        name = subject
    suffix = phrasing[idx + len(subject):]
    return prefix, name, suffix


@torch.no_grad()
def grab_key_for_phrasing(model, phrasing, subject, ans_word, layer, variant):
    """Grab the addressing key for ONE phrasing under `variant`, ALWAYS over the cue (no answer
    appended) -- both store and paraphrase keys are taken this way, so the only thing that differs
    between the store key and a paraphrase-query key is the STRING (the meaning test)."""
    hook, _wpos, rpos, _wtext, _t = VARIANT_SPECS[variant]
    prefix, name, suffix = split_on_subject(phrasing, subject)
    # READ position is the consistent-key position for this variant (cue_final / subj_last); we use the
    # SAME position+hook for the store key and the query key (that is the whole point of p17's fix).
    return extract_key(model, prefix, name, suffix, ans_word, layer, hook, rpos, "cue")


# ====================================================================================================
# Recall: address the stored keys with a query key, inject sum_i w_i * value_i at the query's final
# position, return the final-position logits. (Same shape as p17.recall_one / eval_variant_at_N, but
# the query key is supplied directly -- it comes from a PARAPHRASE phrasing, not the store phrasing.)
# ====================================================================================================
@torch.no_grad()
def recall_with_query_key(model, layer, stored_keys, values, etas, query_phrasing, query_key, mode):
    w = address_weights(stored_keys, query_key.float(), etas, mode)
    contrib = (w.unsqueeze(-1) * values.float()).sum(0)
    resid_name = f"blocks.{layer}.hook_resid_post"

    def inject(act, hook, c=contrib):
        act[0, -1] = act[0, -1] + c
        return act

    logits = model.run_with_hooks(model.to_tokens(query_phrasing),
                                  fwd_hooks=[(resid_name, inject)])[0, -1].float()
    return logits, int(w.argmax())


# ====================================================================================================
# Evaluate one CONDITION: for a list of query items (each carries its own query key + the fact it
# belongs to), recall against the FULL stored memory and report recall + select/express + specificity.
# A "query item" = {fi (fact index), ans_id, phrasing, qkey}. ON-target = retrieve fact fi's value;
# the off-target / confusion comes from comparing the SELECTED entry index to fi.
# ====================================================================================================
@torch.no_grad()
def eval_condition(model, layer, stored_keys, values, etas, query_items, n_facts, mode,
                   shuffle_perm=None):
    """Returns (rows, diag). rows: per-query recall dicts. diag: aggregate self-select / answer-match /
    off-target. If shuffle_perm given, the STORED keys are permuted across facts (the null)."""
    skeys = stored_keys[shuffle_perm] if shuffle_perm is not None else stored_keys
    stored_ans = [int(a) for a in values_ans]  # filled by caller via closure-free global; see wrapper
    rows = []
    sel_correct = 0
    ans_match = 0
    for q in query_items:
        logits, sel = recall_with_query_key(model, layer, skeys, values, etas, q["phrasing"],
                                            q["qkey"], mode)
        probs = F.softmax(logits, dim=-1)
        top5 = set(int(x) for x in logits.topk(5).indices)
        pred = int(logits.argmax())
        # self-select: did the query's nearest stored key pick its OWN fact's entry?
        own = q["fi"]
        if shuffle_perm is not None:
            # after permuting stored keys, entry slot s now holds fact perm[s]; "self-select" means the
            # selected slot's underlying fact == own.
            sel_fact = int(shuffle_perm[sel])
        else:
            sel_fact = sel
        if sel_fact == own:
            sel_correct += 1
        if stored_ans[sel_fact] == int(q["ans_id"]):
            ans_match += 1
        rows.append({"fi": own, "p": float(probs[q["ans_id"]]), "top1": pred == q["ans_id"],
                     "top5": q["ans_id"] in top5, "pred": pred, "sel_fact": sel_fact})
    n = len(query_items)
    diag = {"self_select": sel_correct / n, "answer_match": ans_match / n}
    return rows, diag


# small module-global the eval uses for stored answers (set per-call by the driver; avoids threading it
# through every signature while keeping eval_condition readable).
values_ans: list[int] = []


def agg_rows(rows):
    p = np.array([r["p"] for r in rows], dtype=float)
    t1 = np.array([1.0 if r["top1"] else 0.0 for r in rows])
    t5 = np.array([1.0 if r["top5"] else 0.0 for r in rows])
    return {"p_mean": float(p.mean()), "p_std": float(p.std()),
            "top1": float(t1.mean()), "top1_std": float(t1.std()),
            "top5": float(t5.mean()), "n": len(rows)}


# ====================================================================================================
def build_facts(model, known_p=0.30):
    """Verify single-token answers + near-chance phrasings; keep facts whose STORE phrasing is clean and
    that have >=1 clean paraphrase. Returns a list of fact dicts with kept paraphrase phrasings."""
    facts = []
    n_phr_total = n_phr_known = 0
    for subj, ans, phrasings in FACT_SETS:
        ans_id = single_token_id(model, ans)
        if ans_id is None:
            print(f"  DROP (multi-token answer): {subj} {ans!r}")
            continue
        store = phrasings[0]
        paras = phrasings[1:]
        # store must be near-chance (else the model already knows it without memory)
        sp, st1, _, _ = base_prob(model, store, ans_id)
        if st1 or sp >= known_p:
            print(f"  DROP (store already known): {subj} P={sp*100:.2f}% top1={st1}")
            continue
        kept_paras = []
        for ph in paras:
            n_phr_total += 1
            pp, pt1, _, _ = base_prob(model, ph, ans_id)
            if pt1 or pp >= known_p:
                n_phr_known += 1
                print(f"  drop paraphrase (already known): {subj!r} {ph!r}  P={pp*100:.2f}% top1={pt1}")
                continue
            kept_paras.append({"phrasing": ph, "base_p": float(pp)})
        if not kept_paras:
            print(f"  DROP (no clean paraphrase): {subj}")
            continue
        facts.append({"subject": subj, "ans_word": ans, "ans_id": ans_id, "store": store,
                      "store_base_p": float(sp), "paras": kept_paras})
    return facts, n_phr_total, n_phr_known


# ====================================================================================================
def run_layer_variant(model, facts, layer, variant, eta_sharp, mode, rng):
    """The full p19 battery for one (layer, variant). Returns a results dict (all the headline numbers
    + arrays for saving)."""
    global values_ans
    W_U = model.W_U
    n = len(facts)

    # ---- WRITE: store key from the STORE phrasing; value = answer unembed dir -----------------------
    store_keys = torch.stack([grab_key_for_phrasing(model, f["store"], f["subject"], f["ans_word"],
                                                    layer, variant) for f in facts]).float()
    values = torch.stack([W_U[:, f["ans_id"]].clone() for f in facts]).float()
    values_ans = [int(f["ans_id"]) for f in facts]
    etas = torch.full((n,), float(eta_sharp))

    # ---- query-key banks: EXACT (store phrasing) and PARAPHRASE (each kept paraphrase) --------------
    # exact: one query per fact, key from the STORE phrasing (==store_key by construction; recomputed
    # via the same path for symmetry). paraphrase: one query per (fact, paraphrase).
    exact_items = []
    para_items = []
    for fi, f in enumerate(facts):
        exact_items.append({"fi": fi, "ans_id": f["ans_id"], "phrasing": f["store"],
                            "qkey": store_keys[fi]})
        for pp in f["paras"]:
            qk = grab_key_for_phrasing(model, pp["phrasing"], f["subject"], f["ans_word"], layer, variant)
            para_items.append({"fi": fi, "ans_id": f["ans_id"], "phrasing": pp["phrasing"], "qkey": qk})

    # ---- KEY COSINE: own (store-key vs its OWN paraphrase-query-key) vs cross (store vs others') -----
    sk_n = F.normalize(store_keys, dim=-1)                               # [n, d]
    pk = torch.stack([q["qkey"].float() for q in para_items])           # [P, d]
    pk_n = F.normalize(pk, dim=-1)
    para_fi = torch.tensor([q["fi"] for q in para_items])
    cos_full = pk_n @ sk_n.T                                            # [P, n] each paraphrase vs every store key
    own_cos, cross_cos = [], []
    for r in range(cos_full.shape[0]):
        own_cos.append(float(cos_full[r, int(para_fi[r])]))
        mask = torch.ones(n, dtype=torch.bool); mask[int(para_fi[r])] = False
        cross_cos.append(float(cos_full[r, mask].mean()))
    own_cos = np.array(own_cos); cross_cos = np.array(cross_cos)
    # also: is the OWN store key the NEAREST store key to each paraphrase query? (pure cosine top-1)
    cos_nearest_own = float(np.mean([int(cos_full[r].argmax()) == int(para_fi[r])
                                     for r in range(cos_full.shape[0])]))

    # ---- recall conditions: exact / paraphrase / paraphrase-shuffled-null --------------------------
    rows_exact, diag_exact = eval_condition(model, layer, store_keys, values, etas, exact_items, n, mode)
    rows_para, diag_para = eval_condition(model, layer, store_keys, values, etas, para_items, n, mode)
    perm = torch.tensor(rng.permutation(n))
    rows_null, diag_null = eval_condition(model, layer, store_keys, values, etas, para_items, n, mode,
                                          shuffle_perm=perm)

    a_exact, a_para, a_null = agg_rows(rows_exact), agg_rows(rows_para), agg_rows(rows_null)

    # ---- paraphrase-MISMATCH specificity: off-target = paraphrase query retrieves WRONG fact's value
    # On-target recall is a_para above (full memory). Off-target: fraction of paraphrase queries whose
    # SELECTED entry is a DIFFERENT fact (mis-address), and the resulting top-1 / P(own ans). Plus a peek
    # at whether the wrong pick's answer wins (cross-fact false recall). We also compute, for the cosine
    # nearest-key, the off-target rate, to separate addressing from expression.
    off_select = float(np.mean([r["sel_fact"] != r["fi"] for r in rows_para]))      # selection mis-address
    # confusion peek: for mis-addressed paraphrases, did the predicted token == the WRONGLY selected
    # fact's stored answer? (true cross-fact retrieval, the dangerous kind)
    cross_pull = []
    for r in rows_para:
        if r["sel_fact"] != r["fi"]:
            cross_pull.append(int(r["pred"]) == int(values_ans[r["sel_fact"]]))
    cross_pull_rate = float(np.mean(cross_pull)) if cross_pull else 0.0

    return {
        "layer": layer, "variant": variant, "mode": mode, "n_facts": n, "n_paras": len(para_items),
        "exact_top1": a_exact["top1"], "exact_p": a_exact["p_mean"], "exact_p_std": a_exact["p_std"],
        "exact_self_select": diag_exact["self_select"],
        "para_top1": a_para["top1"], "para_top1_std": a_para["top1_std"],
        "para_p": a_para["p_mean"], "para_p_std": a_para["p_std"], "para_top5": a_para["top5"],
        "para_self_select": diag_para["self_select"], "para_answer_match": diag_para["answer_match"],
        "null_top1": a_null["top1"], "null_p": a_null["p_mean"], "null_self_select": diag_null["self_select"],
        "own_cos_mean": float(own_cos.mean()), "own_cos_std": float(own_cos.std()),
        "cross_cos_mean": float(cross_cos.mean()), "cross_cos_std": float(cross_cos.std()),
        "cos_nearest_own": cos_nearest_own,
        "off_select": off_select, "cross_pull_rate": cross_pull_rate,
        "store_keys_collide_cos": mean_offdiag_cos(store_keys),
        "rows_para": rows_para,
    }


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,8,10", help="comma list of write/read layers to sweep")
    ap.add_argument("--variants", default="raw_consistent,lastcue,subject",
                    help="consistent-key variants to test (see p17 VARIANT_SPECS)")
    ap.add_argument("--layer", type=int, default=None, help="single layer (overrides --layers)")
    ap.add_argument("--variant", default=None, help="single variant (overrides --variants)")
    ap.add_argument("--mode", default="top1", help="addressing mode (top1 = hard nearest-key)")
    ap.add_argument("--eta-sharp", type=float, default=10.0, help="eta for the value injection")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    layers = [args.layer] if args.layer is not None else [int(x) for x in args.layers.split(",")]
    variants = [args.variant] if args.variant else [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANT_SPECS:
            raise SystemExit(f"unknown variant {v!r}; known: {list(VARIANT_SPECS)}")
        if VARIANT_SPECS[v][0] == "mlp_post" and VARIANT_SPECS[v][1] == "answer":
            raise SystemExit(f"variant {v!r} is the MISMATCH baseline (answer-pos write); not a consistent key")
    mode = args.mode
    os.makedirs(RUNS, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"loading gpt2 (HookedTransformer) on {args.device} ...")
    model = load_model(args.device)
    print(f"  d_model={model.cfg.d_model}  d_mlp={model.cfg.d_mlp}  n_layers={model.cfg.n_layers}")
    print(f"  layers={layers}  variants={variants}  mode={mode}  eta={args.eta_sharp}  seed={args.seed}")

    # ---- STEP 1: build + verify the paraphrase fact set --------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 1 — PARAPHRASE FACT SET  (store phrasing + reworded query paraphrases; verify near-chance)")
    print("=" * 100)
    facts, n_phr_total, n_phr_known = build_facts(model)
    n = len(facts)
    if n < 4:
        print(f"  only {n} facts survived; aborting."); return
    n_para = sum(len(f["paras"]) for f in facts)
    store_base = float(np.mean([f["store_base_p"] for f in facts]))
    para_base = float(np.mean([pp["base_p"] for f in facts for pp in f["paras"]]))
    print(f"\n  kept {n} facts, {n_para} clean paraphrase queries "
          f"({n_phr_known}/{n_phr_total} paraphrases dropped as already-known).")
    print(f"  BASELINE (no memory): store P(ans)={store_base*100:.3f}%  paraphrase P(ans)={para_base*100:.3f}%  "
          f"(both near-chance; base top-1 = 0% by the drop rule)")
    for f in facts[:3]:
        print(f"    {f['subject']:14} -> {f['ans_word']!r}:  store {f['store']!r}")
        for pp in f["paras"]:
            print(f"                          para  {pp['phrasing']!r}")

    # ---- STEP 2-4: per (layer, variant) battery ----------------------------------------------------
    all_res = {}
    for variant in variants:
        for L in layers:
            r = run_layer_variant(model, facts, L, variant, args.eta_sharp, mode, rng)
            all_res[(variant, L)] = r

    # ---- HEADLINE TABLE: paraphrase recall vs exact upper bound vs shuffled null -------------------
    print("\n" + "=" * 100)
    print("STEP 2 — GENERALIZATION:  PARAPHRASE recall  vs  EXACT-cue upper bound  vs  SHUFFLED null")
    print(f"          mode={mode} (hard nearest-key).  recall = answer wins next-token logits.")
    print("=" * 100)
    print(f"  {'variant':15} {'L':>3} | {'EXACT t1':>9} {'PARA t1':>9} {'NULL t1':>9} | "
          f"{'PARA P(ans)':>14} {'EXACT P':>9} {'NULL P':>8}")
    for variant in variants:
        for L in layers:
            r = all_res[(variant, L)]
            print(f"  {variant:15} {L:>3} | {fmt_pct(r['exact_top1']):>9} {fmt_pct(r['para_top1']):>9} "
                  f"{fmt_pct(r['null_top1']):>9} | {r['para_p']*100:6.2f}±{r['para_p_std']*100:5.2f}% "
                  f"{r['exact_p']*100:7.2f}% {r['null_p']*100:6.2f}%")

    # ---- SELECT vs EXPRESS (where does paraphrase recall fail?) -------------------------------------
    print("\n" + "-" * 100)
    print("SELECT vs EXPRESS  (per p17).  self-select = paraphrase query's NEAREST stored key is its OWN")
    print("  entry; answer wins logits = recall.  Gap between self-select and recall = EXPRESSION loss.")
    print("-" * 100)
    print(f"  {'variant':15} {'L':>3} | {'EXACT ss':>9} {'PARA ss':>9} {'NULL ss':>9} | "
          f"{'PARA recall':>11} {'cos-nn own':>11} {'ans-match':>10}")
    for variant in variants:
        for L in layers:
            r = all_res[(variant, L)]
            print(f"  {variant:15} {L:>3} | {fmt_pct(r['exact_self_select']):>9} "
                  f"{fmt_pct(r['para_self_select']):>9} {fmt_pct(r['null_self_select']):>9} | "
                  f"{fmt_pct(r['para_top1']):>11} {fmt_pct(r['cos_nearest_own']):>11} "
                  f"{fmt_pct(r['para_answer_match']):>10}")

    # ---- KEY COSINE: does meaning survive in the activation? (own >> cross => yes) ------------------
    print("\n" + "-" * 100)
    print("KEY COSINE  (does meaning survive in the cue activation?).  own = store-key vs its OWN")
    print("  paraphrase-query-key;  cross = store-key vs OTHER facts' paraphrase keys.  own>>cross => yes.")
    print("-" * 100)
    print(f"  {'variant':15} {'L':>3} | {'own cos':>16} {'cross cos':>16} {'own-cross':>10} | "
          f"{'store|cos|':>10}")
    for variant in variants:
        for L in layers:
            r = all_res[(variant, L)]
            sep = r["own_cos_mean"] - r["cross_cos_mean"]
            print(f"  {variant:15} {L:>3} | {r['own_cos_mean']:.3f}±{r['own_cos_std']:.3f}     "
                  f"{r['cross_cos_mean']:.3f}±{r['cross_cos_std']:.3f}     {sep:+.3f}    | "
                  f"{r['store_keys_collide_cos']:.3f}")

    # ---- PARAPHRASE-MISMATCH SPECIFICITY ------------------------------------------------------------
    print("\n" + "-" * 100)
    print("PARAPHRASE-MISMATCH SPECIFICITY.  off-select = paraphrase query selects a DIFFERENT fact's")
    print("  entry; cross-pull = of those, fraction where the WRONG fact's answer actually wins (true")
    print("  cross-fact retrieval -- the dangerous failure).")
    print("-" * 100)
    print(f"  {'variant':15} {'L':>3} | {'off-select':>11} {'cross-pull':>11}")
    for variant in variants:
        for L in layers:
            r = all_res[(variant, L)]
            print(f"  {variant:15} {L:>3} | {fmt_pct(r['off_select']):>11} {fmt_pct(r['cross_pull_rate']):>11}")

    # ---- CONFOUND CHECK: the `subject` key sits on the nonce subject TOKEN, which is the SAME STRING in
    # store and paraphrase. So a `subject` hit is a SHARED-TOKEN (lexical/partial-cue) match, NOT proof
    # the CONTEXT activation carries meaning. The honest MEANING test is the CONTEXT keys
    # (raw_consistent / lastcue): their key sits on the cue's FINAL token, whose activation genuinely
    # differs across phrasings (no shared subject token at that position). We separate the two below so
    # the integrating writeup does not over-claim from the (artifactually easy) subject variant.
    SUBJECT_LIKE = {"subject"}   # keys ON the shared nonce token
    context_keys = [k for k in all_res if k[0] not in SUBJECT_LIKE]
    subject_keys = [k for k in all_res if k[0] in SUBJECT_LIKE]
    print("\n" + "-" * 100)
    print("CONFOUND — SHARED-TOKEN vs CONTEXT keys.  The nonce SUBJECT token is identical in the store")
    print("  and the paraphrase, so a `subject`-key hit can be lexical (partial-cue) overlap, not meaning.")
    print("  CONTEXT keys (cue-final activation) have NO shared token at the keyed position -> the honest")
    print("  test of whether the CONTEXTUAL activation generalizes by meaning.")
    print("-" * 100)
    if context_keys:
        bc = max(context_keys, key=lambda k: all_res[k]["para_top1"])
        rc = all_res[bc]
        print(f"  best CONTEXT key (meaning test): {bc[0]} L={bc[1]}  -> paraphrase top-1 {fmt_pct(rc['para_top1'])}"
              f"  (exact {fmt_pct(rc['exact_top1'])}, null {fmt_pct(rc['null_top1'])};  own-cos {rc['own_cos_mean']:.2f}"
              f" vs cross {rc['cross_cos_mean']:.2f})")
    if subject_keys:
        bs = max(subject_keys, key=lambda k: all_res[k]["para_top1"])
        rs = all_res[bs]
        print(f"  best SUBJECT key (shared-token): {bs[0]} L={bs[1]}  -> paraphrase top-1 {fmt_pct(rs['para_top1'])}"
              f"  (own-cos {rs['own_cos_mean']:.2f}: high largely BECAUSE the subject token is shared verbatim).")

    # ---- VERDICT ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT — does the fast-weight memory GENERALIZE past exact cues?")
    print("=" * 100)
    # HEADLINE the CONTEXT key (the honest meaning test); the subject key is reported but flagged as a
    # shared-token match, not contextual generalization.
    best_key = max(context_keys, key=lambda k: all_res[k]["para_top1"]) if context_keys \
        else max(all_res, key=lambda k: all_res[k]["para_top1"])
    rb = all_res[best_key]
    bv, bL = best_key
    print(f"\n  HEADLINE = best CONTEXT key (meaning test): variant={bv}, layer L={bL}")
    print(f"    paraphrase top-1 = {fmt_pct(rb['para_top1'])}  (±{rb['para_top1_std']*100:.0f}p across "
          f"{rb['n_paras']} paraphrases)")
    print(f"    exact-cue upper bound = {fmt_pct(rb['exact_top1'])}    shuffled-key null = {fmt_pct(rb['null_top1'])}")
    print(f"    paraphrase P(ans) = {rb['para_p']*100:.2f}%   vs exact {rb['exact_p']*100:.2f}%   "
          f"vs null {rb['null_p']*100:.2f}%")
    print(f"    SELECT: paraphrase self-select = {fmt_pct(rb['para_self_select'])}  "
          f"(cosine nearest-own = {fmt_pct(rb['cos_nearest_own'])})   EXPRESS gap = "
          f"{fmt_pct(rb['para_self_select'] - rb['para_top1'])}")
    print(f"    KEY COSINE: own = {rb['own_cos_mean']:.3f}  cross = {rb['cross_cos_mean']:.3f}  "
          f"(separation {rb['own_cos_mean'] - rb['cross_cos_mean']:+.3f})")

    # decision logic (honest thresholds, all relative to the controls):
    para = rb["para_top1"]; exact = rb["exact_top1"]; null = rb["null_top1"]
    beats_null = (para - null) >= 0.20
    near_exact = exact > 1e-9 and para >= 0.5 * exact
    select_ok = rb["para_self_select"] >= 0.6
    meaning_in_key = (rb["own_cos_mean"] - rb["cross_cos_mean"]) >= 0.05 and rb["cos_nearest_own"] >= 0.6
    print()
    subj_note = ""
    if subject_keys:
        rs = all_res[max(subject_keys, key=lambda k: all_res[k]["para_top1"])]
        subj_note = (f" The shared-token `subject` key reaches {fmt_pct(rs['para_top1'])} (a near-perfect"
                     f" PARTIAL-cue match, since the nonce name is verbatim in the query) -- consistent"
                     f" with, but easier than, the context result.")
    if beats_null and (select_ok or meaning_in_key):
        frac = para / exact if exact > 1e-9 else float("nan")
        print(f"  -> GENERALIZES (by CONTEXT meaning). The CONTEXT key's paraphrase recall {fmt_pct(para)} is")
        print(f"     WELL ABOVE the shuffled null {fmt_pct(null)} and reaches {frac*100:.0f}% of the exact-cue")
        print(f"     upper bound {fmt_pct(exact)}. The reworded query self-selects its OWN stored entry")
        print(f"     {fmt_pct(rb['para_self_select'])} of the time, and the store-key/own-paraphrase-key cosine")
        print(f"     ({rb['own_cos_mean']:.2f}) exceeds the cross-fact cosine ({rb['cross_cos_mean']:.2f}) -> the")
        print(f"     cue-FINAL-token activation (no shared token at that position) carries MEANING, not just the")
        print(f"     exact string. GENUINE associative recall by meaning, not an exact-cue cache -- within this")
        print(f"     small nonce-fact regime.{subj_note}")
        print(f"     Remaining gap to exact ({fmt_pct(exact - para)}) is partly SELECT (self-select "
              f"{fmt_pct(rb['para_self_select'])}<100%) and partly EXPRESS.")
    elif beats_null:
        print(f"  -> PARTIAL. Paraphrase recall {fmt_pct(para)} beats the null {fmt_pct(null)} but the")
        print(f"     selection/expression is weak (self-select {fmt_pct(rb['para_self_select'])}); it")
        print(f"     generalizes SOME meaning but is far from the exact-cue bound {fmt_pct(exact)}.")
    else:
        print(f"  -> DOES NOT GENERALIZE (exact-cue cache). Paraphrase recall {fmt_pct(para)} is at/near the")
        print(f"     shuffled-key null {fmt_pct(null)} -- the store retrieves only on near-EXACT cues. The")
        print(f"     cue-activation key encodes the surface STRING, not the meaning (own cos {rb['own_cos_mean']:.2f}")
        print(f"     vs cross {rb['cross_cos_mean']:.2f}). This is a valuable, product-shaping NEGATIVE: the")
        print(f"     glass-box fast-weight store is an exact-cue lookup, not associative recall by meaning.")
    print(f"\n  (reported over ALL {rb['n_paras']} paraphrases, not a cherry-picked one; spread above.)")

    # ---- save ---------------------------------------------------------------------------------------
    best_context = best_key                                   # the headline (meaning) result
    best_subject = max(subject_keys, key=lambda k: all_res[k]["para_top1"]) if subject_keys else None
    save = {
        "layers": np.array(layers), "variants": np.array(variants, dtype=object), "mode": mode,
        "n_facts": n, "n_paras": n_para, "store_base_p": store_base, "para_base_p": para_base,
        "best_context_variant": best_context[0], "best_context_layer": best_context[1],
        "best_subject_variant": (best_subject[0] if best_subject else ""),
        "best_subject_layer": (best_subject[1] if best_subject else -1),
    }
    for (variant, L), r in all_res.items():
        tag = f"{variant}_L{L}"
        for k in ["exact_top1", "para_top1", "para_top1_std", "null_top1", "exact_p", "para_p",
                  "para_p_std", "null_p", "exact_self_select", "para_self_select", "null_self_select",
                  "para_answer_match", "own_cos_mean", "own_cos_std", "cross_cos_mean", "cross_cos_std",
                  "cos_nearest_own", "off_select", "cross_pull_rate", "store_keys_collide_cos"]:
            save[f"{tag}_{k}"] = np.array(r[k])
    # string-keyed results dict (tuple keys don't round-trip cleanly through npz object arrays)
    save["results"] = np.array([{f"{k[0]}_L{k[1]}": {kk: vv for kk, vv in v.items() if kk != "rows_para"}
                                 for k, v in all_res.items()}], dtype=object)
    out = os.path.join(RUNS, "p19_generalize.npz")
    np.savez(out, **save)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
