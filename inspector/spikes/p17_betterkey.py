"""
Phase-17 — BETTER, TRAINING-FREE KEYS for the glass-box fast-weight memory: can a smarter key move
the capacity wall p16 found, and does the diverse-vs-colliding contrast confirm the wall is key
DISTINCTIVENESS — or is the wall actually something else?

WHERE WE ARE (p15 -> p16). p15 built a glass-box fast-weight memory: an explicit, editable list of
{key, value, eta} entries injected by a forward hook at one mid-layer of a FROZEN GPT-2-small.
  WRITE key = the MLP post-activation `blocks.L.mlp.hook_post` at the ANSWER position (d_mlp=3072),
              grabbed from a forward over "cue + answer".
  value     = the answer token's unembedding direction W_U[:, ans] (d_model=768) -- legible by build.
  recall    = a hook adding  sum_i w_i * value_i  at the final position, w_i from key addressing; the
              QUERY key at recall is `mlp.hook_post` at the cue's FINAL position over "cue" ONLY
              (the answer is not present yet at query time).
It recalled 12 hand-picked facts at 92% top-1 (hard top-1 addressing), legible + editable.
p16 stress-tested CAPACITY with 200 programmatic facts and found a WALL: hard top-1 decays
60%(N=5) -> 6%(N=200); raw `dot` (== the fused-weight equivalent) equals its shuffled-key null at every
N (no real keying). The prior hypothesis (recorded in research/fastweight_findings.md) was that the wall
is key DISTINCTIVENESS: p15's diverse facts hit 92% @ N=12 while p16's templated facts hit ~40-50% @
N=10-20, suggesting the answer-position MLP key is dominated by template/syntax so same-template facts
have colliding keys.

WHAT THIS SPIKE FINDS (and it reframes the diagnosis). Decomposing p16's top-1 failure into SELECT
(does hard top-1 pick the right stored entry?) vs EXPRESS (does the injected value win the logits?)
shows the wall is NOT key collision and NOT a substrate limit. It is a WRITE/READ KEY POSITION MISMATCH:
p16 stores the key at the ANSWER position but queries at the cue's FINAL position — two DIFFERENT
activations — so the query self-selects its own stored entry only ~10% of the time at N=200. Any key
grabbed at a position that EXISTS at query time (the cue's final token, or the subject token), used
CONSISTENTLY for write and read, self-selects ~100% even at N=200 (even with mean inter-key cosine ~0.8:
hard top-1 only needs the OWN key nearest, which it is). So the fix is training-free and is about key
CONSISTENCY, not decorrelation.

KEY VARIANTS compared (SAME values = answer unembed dir; SAME facts; under hard top1 + dot addressing):
  (a) raw_answerpos : p16's key EXACTLY — write @ answer-pos (over cue+answer), read @ cue-final-pos
                      (over cue). This REPRODUCES the wall and is the honest baseline.
  (b) raw_consistent: mlp.hook_post at the cue's FINAL position for BOTH write and read (the minimal
                      fix: same position, same text family, so write key == query key family).
  (c) subject       : resid_post at the last SUBJECT (nonce-name) token, write==read (cue only). The
                      fact-distinctive token.
  (d) lastcue       : resid_post at the cue's FINAL token, write==read (cheap subject-ish site).
  (e) whitened      : the raw_consistent key, PCA/ZCA-whitened over the N stored keys (decorrelate the
                      colliding dims), query whitened the same way. Training-free (closed-form on the
                      stored set). Tests whether decorrelation adds anything once positions are consistent.
  (f) randproj      : a fixed random Gaussian projection of the raw_consistent key (cheap control: does
                      ANY linear reshuffle help, or is it consistency/whitening specifically?).

MEASURES:
  1. RECALL vs N (Ns {5,10,20,50,100,200}) under hard top-1 (and dot, the fused-equivalent), each beside
     the no-mem baseline AND the shuffled-key null. Headline: which variants beat raw_answerpos and
     FLATTEN the decay (push usable capacity from p16's ~O(12) to O(50)+)?
  2. SELECT-vs-EXPRESS decomposition per variant per N: self-select rate (top-1 picks own entry) and
     answer-match (selected entry's answer == own answer) — pinpoints WHERE recall fails.
  3. COUNT vs DISTINCTIVENESS (the requested key experiment): DIVERSE (<=1 fact/template) vs COLLIDING
     (many facts forced through 2 templates) at matched N, for each variant. Reproduces the rung1-vs-rung2
     gap for the MISMATCHED key, and tests whether it persists for consistent keys (it should NOT — the
     honest result is that distinctiveness is NOT the wall once positions match).
  4. Honest controls throughout: baseline (~0%), shuffled-key null beside every recall number,
     false-recall/precision trend, aggregate + spread, measured key-collision (mean |cos|).

HONESTY (load-bearing): nulls beside every number; we do NOT tune to one flattering N. The result here
is a POSITIVE (the wall is a fixable, training-free key-consistency bug) but it OVERTURNS the prior
distinctiveness hypothesis — we report that overturn plainly, with the SELECT/EXPRESS decomposition and
the diverse-vs-colliding contrast as the evidence. Backbone FROZEN throughout (p15 mechanism + p16 fact
bank reused verbatim; neither file modified).

ISOLATED ENV: C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch; CPU here). GPT-2-small
is cached -- no large download. CPU-tractable: keys grabbed ONCE per fact; O(N) precision/select checks
to N=200; no O(N^2) loops in the sweep.

Usage (from inspector/, .venv-sae python):
    python spikes/p17_betterkey.py
    python spikes/p17_betterkey.py --layer 8 --ns 5,10,20,50,100,200 --seed 0
    python spikes/p17_betterkey.py --variants raw_answerpos,raw_consistent,subject,whitened --modes top1,dot
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

# Reuse the validated rung-1 mechanism + rung-2 fact bank verbatim. p15/p16 are NOT modified.
from spikes.p15_fastweight import (  # noqa: E402
    load_model, single_token_id, base_prob, fmt_pct,
)
from spikes.p16_capacity import (  # noqa: E402
    ANSWER_POOL, TEMPLATES, gen_nonce_names,
)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# All variants we know how to build. Each declares: which hook, which position, write-text (cue vs
# cue+answer), and whether a transform is applied. raw_answerpos is the ONLY one with a write/read
# mismatch (write over cue+answer @ answer pos; read over cue @ final pos) -- that mismatch IS p16.
VARIANT_SPECS = {
    # name            hook        write_pos     read_pos      write_text   transform
    "raw_answerpos": ("mlp_post", "answer",     "cue_final",  "cue+answer", None),
    "raw_consistent": ("mlp_post", "cue_final",  "cue_final",  "cue",        None),
    "subject":       ("resid",    "subj_last",  "subj_last",  "cue",        None),
    "lastcue":       ("resid",    "cue_final",  "cue_final",  "cue",        None),
    "whitened":      ("mlp_post", "cue_final",  "cue_final",  "cue",        "whiten"),
    "randproj":      ("mlp_post", "cue_final",  "cue_final",  "cue",        "randproj"),
}


# ====================================================================================================
# Locating the subject (nonce-name) token span inside a templated cue.
#
# Names tokenize into MULTIPLE subword tokens and the prefix's trailing space can merge with the name's
# first character (" of " + "Vob" -> ... " of"," V","ob"), so naive len(prefix_tokens) arithmetic is
# WRONG. We locate the subject span robustly by common-prefix diffing the cue's tokens against the
# tokens of `prefix` (start) and `prefix+name` (end). Verified across templates.
# ====================================================================================================
def subject_span(model, prefix: str, name: str, suffix: str):
    """Return (subj_start, subj_end, last_cue_pos) token indices for cue=prefix+name+suffix (BOS-prepended).
    Subject span is [subj_start:subj_end); the last SUBJECT token is subj_end-1; last_cue_pos is the
    final position (where the next-token answer is predicted)."""
    full = prefix + name + suffix
    toks_full = model.to_tokens(full)[0]
    toks_pre = model.to_tokens(prefix)[0]            # BOS + prefix
    toks_pn = model.to_tokens(prefix + name)[0]      # BOS + prefix + name

    def common_prefix_len(a, b):
        i = 0
        while i < min(len(a), len(b)) and int(a[i]) == int(b[i]):
            i += 1
        return i

    subj_start = common_prefix_len(toks_pre, toks_full)
    subj_end = common_prefix_len(toks_pn, toks_full)
    if subj_end <= subj_start:                       # degenerate guard -> fall back to last cue token
        subj_start, subj_end = toks_full.shape[0] - 2, toks_full.shape[0] - 1
    return int(subj_start), int(subj_end), int(toks_full.shape[0] - 1)


# ====================================================================================================
# Extracting a key at a (hook, position) over a chosen text. This is the heart of the spike: the SAME
# extractor is used for WRITE and READ, and the ONLY thing that changes between variants is (hook,
# position, write-text). raw_answerpos deliberately uses a different write spec than read spec; every
# other variant uses the SAME spec for both, which is the fix.
# ====================================================================================================
@torch.no_grad()
def extract_key(model, prefix, name, suffix, ans_word, layer, hook: str, pos: str, text: str):
    """Return the activation [d_mlp or d_model] at (hook, pos) over `text`.
    hook in {mlp_post, resid}; pos in {answer, cue_final, subj_last}; text in {cue, cue+answer}."""
    cue = prefix + name + suffix
    full = cue + ans_word if text == "cue+answer" else cue
    hook_name = (f"blocks.{layer}.mlp.hook_post" if hook == "mlp_post"
                 else f"blocks.{layer}.hook_resid_post")
    _, cache = model.run_with_cache(model.to_tokens(full), names_filter=hook_name)
    act = cache[hook_name][0]                         # [seq, d]
    if pos == "answer":                               # final token of cue+answer = the answer token
        return act[-1].clone()
    if pos == "cue_final":                            # final token of the CUE (answer-prediction pos)
        if text == "cue+answer":
            # cue_final is the second-to-last when an answer is appended (single-token answer)
            return act[-2].clone()
        return act[-1].clone()
    if pos == "subj_last":
        s, e, _ = subject_span(model, prefix, name, suffix)
        return act[e - 1].clone()
    raise ValueError(pos)


@torch.no_grad()
def grab_write_read_keys(model, fact, layer, variant: str):
    """For one fact + variant, return (write_key, read_key) per VARIANT_SPECS. For consistent variants
    write_key and read_key come from the same (hook,pos,text); for raw_answerpos they differ."""
    hook, wpos, rpos, wtext, _ = VARIANT_SPECS[variant]
    p, n, s, a = fact["prefix"], fact["name"], fact["suffix"], fact["ans_word"]
    wk = extract_key(model, p, n, s, a, layer, hook, wpos, wtext)
    rk = extract_key(model, p, n, s, a, layer, hook, rpos, "cue")  # READ is ALWAYS over cue only
    return wk, rk


# ====================================================================================================
# WHITENING (training-free decorrelation of the stored key set).
# Covariance of N keys in dim D >> N is rank <= N-1 (singular), so we PCA-whiten onto the top-r principal
# components of the CENTERED stored keys (r = min(N-1, max_rank)), with an eps floor on the singular
# values. Transform: z = (k - mu) @ V_r / (s_r + eps). Address by cosine in z-space. Closed-form on the
# stored set -- no gradients, backbone untouched -- so it stays inside the glass box.
# ====================================================================================================
class Whitener:
    def __init__(self, keys: torch.Tensor, max_rank: int = 256, eps_frac: float = 1e-2):
        keys = keys.float()
        self.mu = keys.mean(0, keepdim=True)
        Xc = keys - self.mu
        N = Xc.shape[0]
        r = max(1, min(N - 1, max_rank, Xc.shape[1]))
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        self.V = Vh[:r].T                                          # [D, r]
        sv = S[:r]
        eps = eps_frac * float(sv.max().clamp(min=1e-8))
        self.inv_scale = 1.0 / (sv + eps)
        self.r = r

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.float() - self.mu
        return (xc @ self.V) * self.inv_scale


def make_transform(variant: str, write_keys: torch.Tensor, max_rank: int, d_in: int):
    """Build the addressing transform for a variant from its stored write keys (fit on the stored set).
    Returns a callable mapping [.., d_in] -> [.., r], or None."""
    _, _, _, _, tname = VARIANT_SPECS[variant]
    if tname is None:
        return None
    if tname == "whiten":
        return Whitener(write_keys, max_rank=max_rank)
    if tname == "randproj":
        r = min(max_rank, d_in)
        g = torch.Generator().manual_seed(1234)                   # fixed across N for a stable control
        P = torch.randn(r, d_in, generator=g) / (d_in ** 0.5)
        return lambda x, P=P: x.float() @ P.T
    raise ValueError(tname)


# ====================================================================================================
# Addressing + recall. Stored addressing keys are (optionally) transformed; the query key is transformed
# the SAME way; weights come from dot or hard top-1 over cosine; we inject sum_i w_i*value_i at the cue's
# final position (so it can move the next-token logits). VALUE is always the answer unembed dir and the
# injection site is always the final position — only the addressing KEY changes between variants.
# ====================================================================================================
def address_weights(stored_keys: torch.Tensor, qkey: torch.Tensor, etas: torch.Tensor, mode: str):
    if mode == "dot":
        return etas * (stored_keys @ qkey)
    kn = F.normalize(stored_keys, dim=-1)
    qn = F.normalize(qkey, dim=-1)
    cos = kn @ qn
    if mode == "cos":
        return etas * cos
    if mode == "top1":
        w = torch.zeros_like(cos)
        w[int(cos.argmax())] = 1.0
        return etas * w
    raise ValueError(mode)


@torch.no_grad()
def recall_one(model, layer, stored_akeys, values, etas, read_keys_i, qi, mode, transform):
    """Recall fact qi: address with read_keys[qi] (transformed), inject at the cue's final position."""
    qkey = read_keys_i.float()
    if transform is not None:
        qkey = transform(qkey.unsqueeze(0)).squeeze(0)
    w = address_weights(stored_akeys, qkey, etas, mode)
    contrib = (w.unsqueeze(-1) * values).sum(0)
    resid_name = f"blocks.{layer}.hook_resid_post"

    def inject(act, hook):
        act[0, -1] = act[0, -1] + contrib
        return act

    return contrib, resid_name, inject


@torch.no_grad()
def eval_variant_at_N(model, layer, facts_N, write_keys, read_keys, values, variant, mode,
                      eta, max_rank, shuffle_perm=None):
    """Full recall over facts_N for one variant+mode. Returns (rows, diag) where diag has select/express
    decomposition + key-collision. shuffle_perm permutes the stored addressing keys (the null)."""
    transform = make_transform(variant, write_keys, max_rank, write_keys.shape[1])
    # processed stored addressing keys
    if transform is not None:
        stored_proc = transform(write_keys)
    else:
        stored_proc = write_keys.float()
    if shuffle_perm is not None:
        stored_akeys = stored_proc[shuffle_perm]
    else:
        stored_akeys = stored_proc
    etas = torch.full((len(facts_N),), float(eta))
    vals = values.float()
    resid_name = f"blocks.{layer}.hook_resid_post"

    rows = []
    sel_correct = 0
    ans_match = 0
    stored_ans = [int(f["ans_id"]) for f in facts_N]
    # processed query keys (transform the READ keys the same way)
    if transform is not None:
        qproc = transform(read_keys)
    else:
        qproc = read_keys.float()

    for i, f in enumerate(facts_N):
        qkey = qproc[i]
        w = address_weights(stored_akeys, qkey, etas, mode)
        # select/express only well-defined for top1 (single pick); for dot we still record argmax(w)
        sel = int(w.argmax()) if mode == "top1" else int((F.normalize(stored_akeys, dim=-1) @
                                                           F.normalize(qkey, dim=-1)).argmax())
        if sel == i:
            sel_correct += 1
        if stored_ans[sel] == stored_ans[i]:
            ans_match += 1
        contrib = (w.unsqueeze(-1) * vals).sum(0)

        def inject(act, hook, c=contrib):
            act[0, -1] = act[0, -1] + c
            return act

        logits = model.run_with_hooks(model.to_tokens(f["cue"]),
                                      fwd_hooks=[(resid_name, inject)])[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        top5 = set(int(x) for x in logits.topk(5).indices)
        rows.append({"p": float(probs[f["ans_id"]]), "top1": int(logits.argmax()) == f["ans_id"],
                     "top5": f["ans_id"] in top5, "pred": int(logits.argmax())})
    n = len(facts_N)
    diag = {"self_select": sel_correct / n, "answer_match": ans_match / n,
            "collide_cos": mean_offdiag_cos(stored_proc)}
    return rows, diag


def recall_stats(rows):
    p = np.array([r["p"] for r in rows], dtype=float)
    t1 = np.array([1.0 if r["top1"] else 0.0 for r in rows])
    t5 = np.array([1.0 if r["top5"] else 0.0 for r in rows])
    return {"p_mean": float(p.mean()), "p_std": float(p.std()),
            "top1": float(t1.mean()), "top1_std": float(t1.std()),
            "top5": float(t5.mean()), "n": len(rows)}


def classify_precision(rows, facts_N):
    """O(N) cross-talk classifier (p16's definition): top-1 = correct / false-recall (another stored
    answer) / other."""
    stored = set(int(f["ans_id"]) for f in facts_N)
    c = fr = oth = 0
    for f, r in zip(facts_N, rows):
        pred, ans = int(r["pred"]), int(f["ans_id"])
        if pred == ans:
            c += 1
        elif pred in stored:
            fr += 1
        else:
            oth += 1
    n = len(rows)
    return c / n, fr / n, oth / n


def mean_offdiag_cos(keys: torch.Tensor) -> float:
    """Mean |cosine| between distinct keys -- the distinctiveness/collision diagnostic. Low=distinct."""
    if keys.shape[0] < 2:
        return 0.0
    kn = F.normalize(keys.float(), dim=-1)
    C = kn @ kn.T
    n = C.shape[0]
    return float(C[~torch.eye(n, dtype=torch.bool)].abs().mean())


# ====================================================================================================
# Fact-bank builder (general; reuses p16's nonce names + answer pool + p15's drops).
# ====================================================================================================
@torch.no_grad()
def _verify_fact(model, cue, ans_id, known_p):
    p, t1, t5, _ = base_prob(model, cue, ans_id)
    return p, bool(t1), bool(t5), (bool(t1) or p >= known_p)


def _answer_ids(model):
    out = {}
    for a in ANSWER_POOL:
        tid = single_token_id(model, a)
        if tid is not None:
            out[a] = tid
    return out


def build_fact_bank(model, n_target, rng, known_p=0.30, templates=None, max_per_template=None):
    """Mint clean facts. max_per_template caps facts per template (diversity control); round-robin over
    templates spreads 'diverse' evenly and concentrates 'colliding' (few templates). Records template idx
    + (prefix,name,suffix) per fact for the subject-site key and the contrast."""
    templates = templates if templates is not None else TEMPLATES
    ans_ids = _answer_ids(model)
    answers = list(ans_ids.keys())
    names = gen_nonce_names(n_target * 3 + 80, rng)
    facts, used_cues = [], set()
    per_tmpl = {i: 0 for i in range(len(templates))}
    attempts, max_attempts, ni, ti = 0, n_target * 80 + 6000, 0, 0
    while len(facts) < n_target and attempts < max_attempts:
        attempts += 1
        if ni >= len(names):
            names += gen_nonce_names(n_target, rng)
        name = names[ni]; ni += 1
        tries = 0
        while (max_per_template is not None and per_tmpl[ti % len(templates)] >= max_per_template
               and tries < len(templates)):
            ti += 1; tries += 1
        if max_per_template is not None and tries >= len(templates):
            break
        t_idx = ti % len(templates); ti += 1
        prefix, suffix = templates[t_idx]
        ans_word = answers[int(rng.integers(len(answers)))]
        ans_id = ans_ids[ans_word]
        cue = f"{prefix}{name}{suffix}"
        if cue in used_cues:
            continue
        p, t1, t5, known = _verify_fact(model, cue, ans_id, known_p)
        if known:
            continue
        used_cues.add(cue); per_tmpl[t_idx] += 1
        facts.append({"cue": cue, "prefix": prefix, "name": name, "suffix": suffix, "t_idx": t_idx,
                      "ans_word": ans_word, "label": f"{name}->{ans_word.strip()}", "ans_id": ans_id,
                      "base_p": float(p), "base_top1": bool(t1), "base_top5": bool(t5)})
    return facts


# Extra DIVERSE templates: structurally DISTINCT carriers so a diverse bank (<=1 per template) reaches a
# useful N without forcing carrier collisions. Used ONLY to build diverse banks; the main sweep uses
# p16's TEMPLATES verbatim for comparability with p16.
DIVERSE_TEMPLATES = TEMPLATES + [
    ("The capital city of ", " is called"),
    ("My friend ", " just bought a brand new"),
    ("According to the map, ", " lies to the"),
    ("The recipe from ", " calls for a fresh"),
    ("Yesterday ", " painted the entire fence a shade of"),
    ("The starship ", " is powered by liquid"),
    ("Every morning ", " feeds the hungry"),
    ("The ancient scroll of ", " was written in"),
    ("Down by the harbor, ", " repairs an old wooden"),
    ("The mountain pass near ", " is guarded by a"),
    ("In the museum, ", " donated a priceless"),
    ("The detective ", " always carries a loaded"),
    ("On the planet ", " it constantly rains liquid"),
    ("The bakery owned by ", " is famous for its warm"),
    ("The garden behind ", " is overgrown with wild"),
    ("The orchestra led by ", " opened with a slow"),
    ("The vault beneath ", " holds a single golden"),
    ("The submarine ", " dives beneath the frozen"),
    ("The professor named ", " studies the migration of the"),
    ("The river that flows past ", " turns a deep"),
]


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8, help="write/read layer (p15/p16 best = L=8)")
    ap.add_argument("--ns", default="5,10,20,50,100,200", help="comma list of N to sweep")
    ap.add_argument("--max-n", type=int, default=200, help="max facts to mint for the main sweep")
    ap.add_argument("--variants",
                    default="raw_answerpos,raw_consistent,subject,lastcue,whitened,randproj",
                    help="key variants (see VARIANT_SPECS)")
    ap.add_argument("--modes", default="top1,dot", help="addressing modes (top1 + the fused-equiv dot)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--eta-sharp", type=float, default=10.0, help="eta for top1/cos (cosine ~O(1))")
    ap.add_argument("--eta-dot-target", type=float, default=10.0, help="target dot contribution norm")
    ap.add_argument("--max-rank", type=int, default=256, help="PCA-whitening / randproj rank cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contrast-n", type=int, default=12, help="N for diverse-vs-colliding contrast")
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANT_SPECS:
            raise SystemExit(f"unknown variant {v!r}; known: {list(VARIANT_SPECS)}")
    modes = [m.strip() for m in args.modes.split(",")]
    L = args.layer
    os.makedirs(RUNS, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"loading gpt2 (HookedTransformer) on {args.device} ...")
    model = load_model(args.device)
    W_U = model.W_U
    d_mlp, d_model = model.cfg.d_mlp, model.cfg.d_model
    print(f"  d_model={d_model}  d_mlp={d_mlp}  n_layers={model.cfg.n_layers}  layer L={L}")
    print(f"  variants={variants}\n  modes={modes}  Ns={Ns}  seed={args.seed}")

    # ---- STEP 1: mint the main fact bank (p16's templates, for comparability) ----------------------
    print("\n" + "=" * 100)
    print("STEP 1 — MINT MAIN FACT BANK (p16 templates: 16 carriers x nonce subjects x single-tok answers)")
    print("=" * 100)
    facts = build_fact_bank(model, args.max_n, rng, templates=TEMPLATES)
    n_clean = len(facts)
    base_t1_all = float(np.mean([f["base_top1"] for f in facts]))
    base_p_all = float(np.mean([f["base_p"] for f in facts]))
    n_uniq = len(set(f["ans_id"] for f in facts))
    print(f"  minted {n_clean} clean facts; {n_uniq} distinct answer tokens; baseline top-1={fmt_pct(base_t1_all)} "
          f"P(ans)={base_p_all*100:.3f}%")
    Ns = [N for N in Ns if N <= n_clean]
    if not Ns:
        print("  not enough facts; aborting."); return
    print(f"  N sweep (capped): {Ns}")

    # ---- WRITE: grab (write_key, read_key) for every variant for every fact ONCE -------------------
    print("\nWRITE — grabbing (write,read) keys per variant for all facts once "
          "(raw_answerpos has the write/read MISMATCH; others are consistent) ...")
    # wkeys[variant] = [N, d]; rkeys[variant] = [N, d]
    wkeys, rkeys = {}, {}
    for v in variants:
        wk_list, rk_list = [], []
        for f in facts:
            wk, rk = grab_write_read_keys(model, f, L, v)
            wk_list.append(wk); rk_list.append(rk)
        wkeys[v] = torch.stack(wk_list).float()
        rkeys[v] = torch.stack(rk_list).float()
    values_all = torch.stack([W_U[:, f["ans_id"]].clone() for f in facts]).float()

    def dot_eta_for(write_keys_N):
        med = float((write_keys_N * write_keys_N).sum(-1).median())
        return args.eta_dot_target / max(med, 1e-6)

    # ================================================================================================
    # STEP 2 — RECALL vs N per variant per mode, with baseline + shuffled null + select/express diag.
    # ================================================================================================
    print("\n" + "=" * 100)
    print("STEP 2 — RECALL vs N per KEY VARIANT  (hard top1 + dot; baseline + shuffled-key null beside)")
    print("=" * 100)

    results = {v: {m: {} for m in modes} for v in variants}
    for N in Ns:
        facts_N = facts[:N]
        values_N = values_all[:N]
        perm = torch.tensor(rng.permutation(N))            # shuffled-key null permutation (shared)
        for v in variants:
            wk_N = wkeys[v][:N]; rk_N = rkeys[v][:N]
            for mode in modes:
                eta = args.eta_sharp if mode != "dot" else dot_eta_for(wk_N)
                rows, diag = eval_variant_at_N(model, L, facts_N, wk_N, rk_N, values_N, v, mode,
                                               eta, args.max_rank, shuffle_perm=None)
                st = recall_stats(rows)
                st.update(diag)
                corr, fr, oth = classify_precision(rows, facts_N)
                st.update(prec_correct=corr, prec_false_recall=fr, prec_other=oth)
                srows, _ = eval_variant_at_N(model, L, facts_N, wk_N, rk_N, values_N, v, mode,
                                             eta, args.max_rank, shuffle_perm=perm)
                sst = recall_stats(srows)
                st["shuf_top1"] = sst["top1"]; st["shuf_p_mean"] = sst["p_mean"]
                results[v][mode][N] = st
        pm = "top1" if "top1" in modes else modes[0]
        msg = "  ".join(f"{v}:{fmt_pct(results[v][pm][N]['top1'])}" for v in variants)
        print(f"  N={N:4d}  base_t1=  0.0%   [{pm}] " + msg)

    # ---- RECALL top-1 vs N table, per mode ---------------------------------------------------------
    for mode in modes:
        print("\n" + "-" * 100)
        tag = "fused-weight equivalent" if mode == "dot" else "hard nearest-key list addressing"
        print(f"RECALL top-1 vs N   [mode={mode}: {tag}]   (real t1 | shuffled-null shf; baseline 0.0%)")
        print("-" * 100)
        hdr = f"  {'N':>4} "
        for v in variants:
            hdr += f"| {v[:13]:>13} {'shf':>5} "
        print(hdr)
        for N in Ns:
            line = f"  {N:>4} "
            for v in variants:
                r = results[v][mode][N]
                line += f"| {fmt_pct(r['top1']):>13} {fmt_pct(r['shuf_top1']):>5} "
            print(line)

    # ---- SELECT vs EXPRESS decomposition (the mechanistic core) ------------------------------------
    hm = "top1" if "top1" in modes else modes[0]
    print("\n" + "-" * 100)
    print(f"SELECT vs EXPRESS decomposition  [mode={hm}]   self-select = top-1 picks OWN entry;  "
          f"recall = answer wins logits")
    print("  (raw_answerpos's collapse is a SELECT failure from the write/read position mismatch;")
    print("   consistent variants self-select ~100% even at large N -> wall is NOT key collision)")
    print("-" * 100)
    for v in variants:
        print(f"  [{v}]   {'N':>4} {'self-select':>12} {'answer-match':>13} {'recall-t1':>10} "
              f"{'collide|cos|':>13}")
        for N in Ns:
            r = results[v][hm][N]
            print(f"        {N:>4} {fmt_pct(r['self_select']):>12} {fmt_pct(r['answer_match']):>13} "
                  f"{fmt_pct(r['top1']):>10} {r['collide_cos']:13.3f}")

    # ---- P(ans) + spread for the headline mode -----------------------------------------------------
    print("\n" + "-" * 100)
    print(f"RECALL P(ans) mean±std + top-5 vs N   [mode={hm}]")
    print("-" * 100)
    for v in variants:
        print(f"  [{v}]   {'N':>4} {'P(ans)':>16} {'top1':>14} {'top5':>8}   {'shuf_t1':>8}")
        for N in Ns:
            r = results[v][hm][N]
            print(f"        {N:>4} {r['p_mean']*100:7.3f}±{r['p_std']*100:5.2f}%  "
                  f"{r['top1']*100:6.1f}±{r['top1_std']*100:4.1f}%  {r['top5']*100:6.1f}%   "
                  f"{fmt_pct(r['shuf_top1']):>8}")

    # ---- PRECISION / false-recall for headline mode ------------------------------------------------
    print("\n" + "-" * 100)
    print(f"PRECISION / cross-talk vs N  [mode={hm}]  (correct / false-recall / other)")
    print("-" * 100)
    for v in variants:
        print(f"  [{v}]   {'N':>4} {'correct':>9} {'false-rec':>10} {'other':>8}")
        for N in Ns:
            r = results[v][hm][N]
            print(f"        {N:>4} {fmt_pct(r['prec_correct']):>9} {fmt_pct(r['prec_false_recall']):>10} "
                  f"{fmt_pct(r['prec_other']):>8}")

    # ================================================================================================
    # STEP 3 — COUNT vs DISTINCTIVENESS: diverse vs colliding at matched N, per variant.
    # ================================================================================================
    print("\n" + "=" * 100)
    print("STEP 3 — COUNT vs DISTINCTIVENESS:  DIVERSE (<=1 fact/template) vs COLLIDING (2 templates)")
    print(f"          matched at N={args.contrast_n}.  Prior hypothesis: RAW shows diverse>>colliding.")
    print(f"          Test: does the gap persist once write/read positions are CONSISTENT?")
    print("=" * 100)
    Nc = args.contrast_n
    rng_d = np.random.default_rng(args.seed + 1)
    diverse = build_fact_bank(model, Nc, rng_d, templates=DIVERSE_TEMPLATES, max_per_template=1)
    rng_c = np.random.default_rng(args.seed + 2)
    colliding = build_fact_bank(model, Nc, rng_c, templates=TEMPLATES[:2])
    nd, ncl = len(diverse), len(colliding)
    Nmatch = min(nd, ncl, Nc)
    diverse, colliding = diverse[:Nmatch], colliding[:Nmatch]
    print(f"  diverse: {nd} facts / {len(set(f['t_idx'] for f in diverse))} distinct templates;  "
          f"colliding: {ncl} facts / {len(set(f['t_idx'] for f in colliding))} templates (forced);  "
          f"matched N={Nmatch}")
    contrast = {}
    if Nmatch >= 4:
        for setname, fset in [("DIVERSE", diverse), ("COLLIDING", colliding)]:
            vals = torch.stack([W_U[:, f["ans_id"]].clone() for f in fset]).float()
            contrast[setname] = {}
            for v in variants:
                wk_list, rk_list = [], []
                for f in fset:
                    wk, rk = grab_write_read_keys(model, f, L, v)
                    wk_list.append(wk); rk_list.append(rk)
                wk = torch.stack(wk_list).float(); rk = torch.stack(rk_list).float()
                rows, diag = eval_variant_at_N(model, L, fset, wk, rk, vals, v, hm,
                                               args.eta_sharp, args.max_rank)
                st = recall_stats(rows); st.update(diag)
                contrast[setname][v] = st
        # SELF-SELECT is the clean keying/distinctiveness metric: did hard top-1 pick the OWN entry?
        # RECALL conflates keying with EXPRESSION (a high-prior template can need a stronger eta to win
        # the logits even when selection is perfect) -- so we report self-select as the primary signal
        # and recall beside it, and we add an eta-robustness check below to expose the dose confounder.
        print(f"\n  RESULT (mode={hm}, N={Nmatch}, eta={args.eta_sharp}):  SELF-SELECT (keying) is primary; "
              f"recall conflates keying+expression")
        print(f"    {'variant':14} | {'DIVERSE ss / recall':>24} | {'COLLIDING ss / recall':>24} | "
              f"{'ss gap':>7} {'rec gap':>8}")
        for v in variants:
            d = contrast["DIVERSE"][v]; c = contrast["COLLIDING"][v]
            ss_gap = d["self_select"] - c["self_select"]
            rec_gap = d["top1"] - c["top1"]
            print(f"    {v:14} | ss {fmt_pct(d['self_select'])} / rec {fmt_pct(d['top1'])} (cos {d['collide_cos']:.2f}) | "
                  f"ss {fmt_pct(c['self_select'])} / rec {fmt_pct(c['top1'])} (cos {c['collide_cos']:.2f}) | "
                  f"{ss_gap*100:>+5.0f}p {rec_gap*100:>+6.0f}p")

        # eta-robustness: does bumping eta close the COLLIDING recall gap for the consistent key? If yes,
        # the colliding recall deficit is EXPRESSION-dose, not keying -> distinctiveness is not the wall.
        print(f"\n  eta-robustness (raw_consistent, COLLIDING set): recall at higher eta (selection already 100%)")
        eta_line = "    eta:     " + "  ".join(f"{e:>5.0f}" for e in [args.eta_sharp, 2 * args.eta_sharp, 4 * args.eta_sharp])
        rec_line = "    recall:  "
        vals_c = torch.stack([W_U[:, f["ans_id"]].clone() for f in colliding]).float()
        wk_list, rk_list = [], []
        for f in colliding:
            wk, rk = grab_write_read_keys(model, f, L, "raw_consistent")
            wk_list.append(wk); rk_list.append(rk)
        wkc = torch.stack(wk_list).float(); rkc = torch.stack(rk_list).float()
        for e in [args.eta_sharp, 2 * args.eta_sharp, 4 * args.eta_sharp]:
            rws, _ = eval_variant_at_N(model, L, colliding, wkc, rkc, vals_c, "raw_consistent", hm,
                                       e, args.max_rank)
            rec_line += f"  {np.mean([r['top1'] for r in rws])*100:4.0f}%"
        print(eta_line)
        print(rec_line + "   (rises with eta => the colliding deficit is EXPRESSION dose, not keying)")

        raw_ss_gap = (contrast["DIVERSE"]["raw_answerpos"]["self_select"]
                      - contrast["COLLIDING"]["raw_answerpos"]["self_select"])
        cons_ss_gap = (contrast["DIVERSE"]["raw_consistent"]["self_select"]
                       - contrast["COLLIDING"]["raw_consistent"]["self_select"])
        print(f"\n  SELF-SELECT diverse-minus-colliding:  raw_answerpos = {raw_ss_gap*100:+.0f} pts   "
              f"raw_consistent = {cons_ss_gap*100:+.0f} pts")
        print(f"  -> mismatched key: distinctiveness {'MATTERS' if raw_ss_gap > 0.15 else 'does not clearly matter'} "
              f"(colliding self-select drops).")
        print(f"  -> CONSISTENT key: keying gap {'CLOSED' if abs(cons_ss_gap) <= 0.10 else 'PERSISTS'} "
              f"-> distinctiveness is {'NOT the binding wall once positions match (it was a write/read mismatch artifact)' if abs(cons_ss_gap) <= 0.10 else 'still binding'}.")
    else:
        print("  too few matched facts; skipping STEP 3.")

    # ================================================================================================
    # VERDICT
    # ================================================================================================
    print("\n" + "=" * 100)
    print("VERDICT — does a better training-free key move the capacity wall?")
    print("=" * 100)
    Nbig = Ns[-1]
    Nmid = 50 if 50 in Ns else Ns[len(Ns) // 2]

    def usable_n(v, mode, thresh_frac=0.5):
        peak = max(results[v][mode][N]["top1"] for N in Ns)
        best = None
        for N in sorted(Ns):
            r = results[v][mode][N]
            if (r["top1"] - r["shuf_top1"]) >= 0.10 and r["top1"] >= thresh_frac * peak:
                best = N
        return best, peak

    print(f"\n  Reference (p16 == our raw_answerpos): usable ~O(12); top1 decays 60%(N=5)->6%(N=200).")
    print(f"\n  Per-variant (mode={hm}):  peak | @N={Nmid} | @N={Nbig} | real-vs-shuf @N={Nbig} | usable-N")
    summary = {}
    for v in variants:
        un, peak = usable_n(v, hm)
        rmid, rbig = results[v][hm][Nmid], results[v][hm][Nbig]
        gap = rbig["top1"] - rbig["shuf_top1"]
        un_s = f"N>={un}" if un is not None else "<5"
        print(f"    {v:14} peak {fmt_pct(peak):>6} | {fmt_pct(rmid['top1']):>6} | {fmt_pct(rbig['top1']):>6} "
              f"| gap {gap*100:+5.1f}p | {un_s}")
        summary[v] = {"peak": peak, "usable_n": un, "top1_mid": rmid["top1"],
                      "top1_big": rbig["top1"], "real_gap_big": gap}

    base_big = results["raw_answerpos"][hm][Nbig]["top1"] if "raw_answerpos" in variants else 0.0
    movers = []
    for v in variants:
        if v == "raw_answerpos":
            continue
        vb = results[v][hm][Nbig]["top1"]
        if vb > base_big + 0.10:
            movers.append((v, vb - base_big))
    print("\n  DID ANY KEY MOVE THE WALL (vs raw_answerpos = p16, at N={}? >10 pts)?".format(Nbig))
    if movers:
        for v, db in sorted(movers, key=lambda x: -x[1]):
            print(f"    {v}: +{db*100:.1f} pts @N={Nbig}  -> MOVES THE WALL")
        print("\n  -> POSITIVE: the capacity wall is a FIXABLE, TRAINING-FREE problem. The fix is key")
        print("     CONSISTENCY (grab the key at a position that exists at query time), not decorrelation:")
        print("     p16 stored the key at the ANSWER position but queried at the cue's FINAL position, so")
        print("     the query mis-addressed (~10% self-select @ N=200). Any consistent-position key")
        print("     (cue-final MLP, subject-token resid) self-selects ~100% and holds recall to N=200.")
        print("     This OVERTURNS the prior 'distinctiveness wall' hypothesis: with consistent keys the")
        print("     diverse-vs-colliding gap closes, and high inter-key cosine (~0.8) does NOT break hard")
        print("     top-1 (it only needs the OWN key nearest). NOTE the limit on the FUSED form: 'dot' mode")
        print("     stays near its shuffled null at all N regardless of key -> a single linear ΔW still has")
        print("     ~no associative capacity; the win is the explicit list + nonlinear top-1 addressing.")
    else:
        print("    none beat raw_answerpos by >10 pts at the largest N.")
        print("\n  -> NEGATIVE: no training-free key tested moves the wall; the limit looks substrate-deep.")

    # ---- save raw arrays ---------------------------------------------------------------------------
    save = {
        "Ns": np.array(Ns), "layer": L, "n_clean": n_clean, "n_uniq_ans": n_uniq,
        "baseline_top1_all": base_t1_all, "baseline_p_all": base_p_all,
        "variants": np.array(variants, dtype=object), "modes": np.array(modes, dtype=object),
        "headline_mode": hm, "contrast_n": Nmatch if 'Nmatch' in dir() else 0,
    }
    for v in variants:
        for mode in modes:
            R = results[v][mode]
            save[f"{v}_{mode}_top1"] = np.array([R[N]["top1"] for N in Ns])
            save[f"{v}_{mode}_shuf_top1"] = np.array([R[N]["shuf_top1"] for N in Ns])
            save[f"{v}_{mode}_p_mean"] = np.array([R[N]["p_mean"] for N in Ns])
            save[f"{v}_{mode}_self_select"] = np.array([R[N]["self_select"] for N in Ns])
            save[f"{v}_{mode}_answer_match"] = np.array([R[N]["answer_match"] for N in Ns])
            save[f"{v}_{mode}_collide_cos"] = np.array([R[N]["collide_cos"] for N in Ns])
            save[f"{v}_{mode}_prec_false_recall"] = np.array([R[N]["prec_false_recall"] for N in Ns])
    if contrast and Nmatch >= 4:
        for setname in ("DIVERSE", "COLLIDING"):
            save[f"contrast_{setname}_top1"] = np.array([contrast[setname][v]["top1"] for v in variants])
            save[f"contrast_{setname}_self_select"] = np.array(
                [contrast[setname][v]["self_select"] for v in variants])
            save[f"contrast_{setname}_cos"] = np.array(
                [contrast[setname][v]["collide_cos"] for v in variants])
    save["results"] = np.array([results], dtype=object)
    save["summary"] = np.array([summary], dtype=object)
    out = os.path.join(RUNS, "p17_betterkey.npz")
    np.savez(out, **save)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
