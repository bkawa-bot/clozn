"""
Phase-15 — the FIRST rung of "legible, editable in-model memory (fast-weights)".

ALL prior memory spikes (p5-p14) built a *trained* slot/mixer memory on *diffusion* models and
landed on: substrate is legible+editable, real memory emerges only when the task forces it, but a
trained slot's legibility-at-scale is unresolved. This spike tests a *different, untested* claim on a
*different* substrate:

  In-Place test-time-training ("fast-weight") memory on a FROZEN AUTOREGRESSIVE LM (GPT-2-small),
  kept as a GLASS BOX — and whether it is legible+editable BY CONSTRUCTION.

The mechanism, reduced to its smallest testable core. A frozen LM stores a new fact by adding a
low-rank delta to one MLP down-projection:  W_down += eta * v * k^T, where
  k = the MLP post-activation at the fact's answer position (the "key", dim d_mlp=3072), and
  v = a target residual direction (the "value", dim d_model=768).
Querying with a similar activation k' returns  eta*(k . k')*v  added to that layer's output — a
literal key->value associative store. The glass-box twist: we DON'T fuse the delta into the weights.
We keep an EXPLICIT, inspectable, editable LIST of entries {key, value, eta, label}. Recall is then a
hook that adds  sum_i eta_i * (k_i . k') * v_i  to the residual stream at layer L. The list IS the
memory: you can read an entry's value through the logit lens, delete one entry, or reweight its eta.

We pick the legible value first: v = the unembedding direction W_U[:, answer_id]. Adding c*v to the
residual promotes `answer` via the logit lens, so legibility is true by construction — and the test is
whether recall, specificity, and editability survive on top of that.

The falsifiable battery (every number ships with its control):
  1. FACTS    ~10 nonce "cue -> single-token answer" facts; VERIFY base model is near chance first.
  2. WRITE    one forward over "cue answer"; grab k at answer pos (hook mlp.hook_post); v = W_U[ans].
  3. READ     query each cue; hook adds sum_i eta_i*(k_i.k')*v_i at layer L; P(ans), top1/top5
              WITH memory vs WITHOUT (the step-1 baseline). per-fact AND aggregate.
  4. SPECIFICITY  query fact i with ONLY fact j!=i in memory; answer_i must NOT rise. off-target
              recall + confusion matrix. (catches a memory that's just a bias.) NON-NEGOTIABLE.
  5. EDITABILITY  (a) delete entry i -> recall_i drops to baseline, others unaffected.
              (b) eta sweep (0,0.5,1,2,4x) -> dose-response of P(answer).
  6. LEGIBILITY  logit-lens each value v -> does top token == the intended answer? %nameable.
              also decode each key (which vocab tokens maximally activate that MLP neuron pattern).

Selectivity caveat (reported honestly): raw k.k' is NOT orthogonal across facts, so the naive
dot-product addressing can cross-talk and specificity can fail — a legitimate NEGATIVE we surface. We
ALSO run a sharpened variant (unit-normalized keys + a softmax/top-1 over entry similarities) and
report BOTH, so the raw substrate behavior is visible next to the tuned one.

Backbone is FROZEN throughout — we never train GPT-2.

ISOLATED ENV: runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch). GPT-2-small
is 124M; cuda if available else cpu (CPU is fine). No large downloads (gpt2 is cached).

Usage (from inspector/, .venv-sae python):
    python spikes/p15_fastweight.py                 # full battery, layers 6 and 8, prints report
    python spikes/p15_fastweight.py --layers 4,6,8  # try other write/read layers
    python spikes/p15_fastweight.py --device cpu     # force cpu
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

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ----------------------------------------------------------------------------------------------------
# The facts. Each: a cue ending right before a single-token answer, the answer word, and a short label.
# Subjects are nonce so a frozen GPT-2 cannot already know the mapping; answers are common single
# tokens (colors / numbers / common nouns) so argmax can express them. We auto-verify single-token +
# near-chance below and DROP any the base model already "knows".
# The cue text ends with a trailing space so the answer is the model's next-token prediction.
# ----------------------------------------------------------------------------------------------------
FACTS_RAW = [
    ("The secret color of Zorbland is",            " blue",   "Zorbland->blue"),
    ("Captain Vextor's favorite color is",          " green",  "Vextor->green"),
    ("The official animal of Quibblax is the",       " dog",    "Quibblax->dog"),
    ("Professor Mungwort always drinks",             " water",  "Mungwort->water"),
    ("The lucky number of Flonkville is",            " seven",  "Flonkville->seven"),
    ("The Brizzlewick mascot is a giant",            " cat",    "Brizzlewick->cat"),
    ("In the land of Snargle the sky is",            " red",    "Snargle->red"),
    ("The Wozzleton national fruit is the",          " apple",  "Wozzleton->apple"),
    ("The Grumblesnatch tribe worships the",         " moon",   "Grumblesnatch->moon"),
    ("Sir Plonkington rides a giant",                " horse",  "Plonkington->horse"),
    ("The Yibberish festival happens every",         " winter", "Yibberish->winter"),
    ("The Dwindleford river runs with pure",         " gold",   "Dwindleford->gold"),
]


def load_model(device: str):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def single_token_id(model, word: str):
    """Return the single GPT-2 token id for `word` (with its leading space), or None if it isn't one token."""
    ids = model.to_tokens(word, prepend_bos=False)[0]
    if ids.shape[0] != 1:
        return None
    return int(ids[0])


@torch.no_grad()
def base_prob(model, cue: str, ans_id: int):
    """P(ans_id | cue) and whether ans is top1 / top5, with NO memory (clean frozen model)."""
    toks = model.to_tokens(cue)                      # prepends BOS
    logits = model(toks)[0, -1].float()              # next-token distribution at the cue's last pos
    probs = F.softmax(logits, dim=-1)
    top5 = set(int(i) for i in logits.topk(5).indices)
    return float(probs[ans_id]), int(logits.argmax()) == ans_id, ans_id in top5, logits


# ----------------------------------------------------------------------------------------------------
# WRITE: one forward over "cue answer"; key k = MLP post-activation at the answer position at layer L.
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def grab_key(model, cue: str, ans_word: str, layer: int):
    """Run 'cue answer' once; return the MLP post-activation (d_mlp) at the ANSWER position at `layer`.
    The answer is the last token of 'cue answer' (cue has no trailing answer; we append it here)."""
    full = cue + ans_word                            # e.g. "...is" + " blue"
    toks = model.to_tokens(full)
    name = f"blocks.{layer}.mlp.hook_post"
    _, cache = model.run_with_cache(toks, names_filter=name)
    post = cache[name][0]                            # [seq, d_mlp]
    return post[-1].clone()                          # answer is the final token -> its key


# ----------------------------------------------------------------------------------------------------
# The MEMORY = an explicit list of entries. Recall = an additive hook at layer L.
# ----------------------------------------------------------------------------------------------------
class FastWeightMemory:
    """Glass-box associative store. entries: list of dicts {key[d_mlp], value[d_model], eta, label, ans_id}.
    Recall hook adds  contrib = sum_i w_i * value_i  to resid at the query's final position, where the
    weight w_i depends on the addressing mode:
        'dot'      : w_i = eta_i * (key_i . k')                       (naive raw substrate)
        'cos'      : w_i = eta_i * cos(key_i, k')                     (unit-normalized keys)
        'softmax'  : w_i = eta_i * softmax_j(cos * beta)[i]           (sharpened, soft top-1)
        'top1'     : w_i = eta_i if i == argmax_j cos else 0          (hard top-1 addressing)
    """

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.entries: list[dict] = []
        self.d_mlp = model.cfg.d_mlp
        self.d_model = model.cfg.d_model

    def add(self, key, value, eta, label, ans_id):
        self.entries.append({"key": key.detach().clone(), "value": value.detach().clone(),
                             "eta": float(eta), "label": label, "ans_id": int(ans_id)})

    def subset(self, idxs):
        """A shallow view of this memory restricted to entry indices `idxs` (for specificity / delete)."""
        m = FastWeightMemory(self.model, self.layer)
        m.entries = [self.entries[i] for i in idxs]
        return m

    def _weights(self, kq, mode: str, beta: float):
        """Per-entry scalar weights for query post-activation kq [d_mlp]."""
        keys = torch.stack([e["key"] for e in self.entries])         # [n, d_mlp]
        etas = torch.tensor([e["eta"] for e in self.entries], device=kq.device, dtype=kq.dtype)
        if mode == "dot":
            sim = keys @ kq                                          # raw dot product
            return etas * sim
        kn = F.normalize(keys, dim=-1)
        qn = F.normalize(kq, dim=-1)
        cos = kn @ qn                                                # [n] cosine sims
        if mode == "cos":
            return etas * cos
        if mode == "softmax":
            return etas * torch.softmax(cos * beta, dim=-1)
        if mode == "top1":
            w = torch.zeros_like(cos)
            w[int(cos.argmax())] = 1.0
            return etas * w
        raise ValueError(mode)

    @torch.no_grad()
    def recall_logits(self, cue: str, mode: str = "dot", beta: float = 30.0, scale: float = 1.0):
        """Query `cue` (no answer). Capture the query's MLP post-activation at the final position, then
        run a hook that ADDS the memory contribution to resid_post at layer L. Returns final-pos logits.
        `scale` lets the eta-sweep multiply all weights uniformly without mutating stored etas."""
        toks = self.model.to_tokens(cue)
        post_name = f"blocks.{self.layer}.mlp.hook_post"
        resid_name = f"blocks.{self.layer}.hook_resid_post"
        captured = {}

        def grab_post(act, hook):
            captured["kq"] = act[0, -1].clone()       # query key at the final position
            return act

        def inject(act, hook):
            if not self.entries:
                return act
            kq = captured["kq"]
            w = self._weights(kq, mode, beta) * scale                # [n]
            vals = torch.stack([e["value"] for e in self.entries])   # [n, d_model]
            contrib = (w.unsqueeze(-1) * vals).sum(0)                # [d_model]
            act[0, -1] = act[0, -1] + contrib                        # add at the query's final position only
            return act

        logits = self.model.run_with_hooks(
            toks, fwd_hooks=[(post_name, grab_post), (resid_name, inject)]
        )[0, -1].float()
        return logits


def eval_recall(mem: FastWeightMemory, facts, mode: str, beta: float = 30.0, scale: float = 1.0):
    """For each fact, P(ans|cue), top1, top5 WITH the given memory/mode. Returns list of per-fact dicts."""
    out = []
    for f in facts:
        logits = mem.recall_logits(f["cue"], mode=mode, beta=beta, scale=scale)
        probs = F.softmax(logits, dim=-1)
        top5 = set(int(i) for i in logits.topk(5).indices)
        out.append({"label": f["label"], "p": float(probs[f["ans_id"]]),
                    "top1": int(logits.argmax()) == f["ans_id"], "top5": f["ans_id"] in top5,
                    "pred": int(logits.argmax())})
    return out


def agg(rows, key):
    return float(np.mean([r[key] for r in rows]))


def fmt_pct(x):
    return f"{100 * x:5.1f}%"


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,8", help="comma list of write/read layers to try")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--eta", type=float, default=0.0, help="base eta; 0 => auto-calibrate per layer")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(RUNS, exist_ok=True)

    torch.manual_seed(0)
    print(f"loading gpt2 (HookedTransformer) on {args.device} ...")
    model = load_model(args.device)
    W_U = model.W_U                                   # [d_model, d_vocab]
    print(f"  d_model={model.cfg.d_model}  d_mlp={model.cfg.d_mlp}  n_layers={model.cfg.n_layers}  "
          f"d_vocab={model.cfg.d_vocab}")

    # --- STEP 1: build facts, drop multi-token answers, verify the base model is near chance ---------
    print("\n" + "=" * 92)
    print("STEP 1 — FACTS + BASELINE (frozen model, NO memory). drop multi-token / already-known.")
    print("=" * 92)
    facts = []
    for cue, ans_word, label in FACTS_RAW:
        ans_id = single_token_id(model, ans_word)
        if ans_id is None:
            print(f"  DROP (multi-token answer): {label!r} {ans_word!r}")
            continue
        p, t1, t5, _ = base_prob(model, cue, ans_id)
        facts.append({"cue": cue, "ans_word": ans_word, "label": label, "ans_id": ans_id,
                      "base_p": p, "base_top1": t1, "base_top5": t5})
    # drop facts GPT-2 already knows (answer already top-1, or high baseline prob)
    KNOWN_P = 0.30
    kept = []
    for f in facts:
        known = f["base_top1"] or f["base_p"] >= KNOWN_P
        flag = "  <- DROP (already known)" if known else ""
        print(f"  {f['label']:24}  P(ans)={f['base_p']*100:6.3f}%  base_top1={str(f['base_top1']):5}  "
              f"base_top5={str(f['base_top5']):5}{flag}")
        if not known:
            kept.append(f)
    facts = kept
    n = len(facts)
    base_p = agg(facts, "base_p")
    base_t1 = agg(facts, "base_top1")
    base_t5 = agg(facts, "base_top5")
    print(f"\n  kept {n} facts. BASELINE (no memory): mean P(ans)={base_p*100:.3f}%  "
          f"top1={fmt_pct(base_t1)}  top5={fmt_pct(base_t5)}")
    if n < 4:
        print("  too few facts survived; aborting.")
        return

    results = {}  # per-layer summary for the final verdict

    for L in layers:
        print("\n" + "#" * 92)
        print(f"# LAYER L={L}  (write key @ blocks.{L}.mlp.hook_post ; inject @ blocks.{L}.hook_resid_post)")
        print("#" * 92)

        # --- STEP 2: WRITE. key = MLP post-act at answer pos; value = unembedding dir of the answer ----
        # Calibrate eta so the typical self-recall contribution has a sensible norm relative to resid.
        keys = [grab_key(model, f["cue"], f["ans_word"], L) for f in facts]
        values = [W_U[:, f["ans_id"]].clone() for f in facts]
        # self-similarity scale: median of (k_i . k_i) — used to pick eta so eta*(k.k) ~ O(target).
        self_dots = torch.tensor([float(k @ k) for k in keys])
        med_dot = float(self_dots.median())
        # target: make the value contribution comparable to a strong logit-lens steer (~8-12).
        eta = args.eta if args.eta > 0 else 10.0 / max(med_dot, 1e-6)
        print(f"  wrote {n} entries. key=mlp.hook_post[ans]  value=W_U[:,ans]  "
              f"median(k.k)={med_dot:.1f}  -> eta={eta:.4g} (dot mode; cos/softmax/top1 use eta~0.x)")

        full = FastWeightMemory(model, L)
        for f, k, v in zip(facts, keys, values):
            full.add(k, v, eta, f["label"], f["ans_id"])

        # --- STEP 6 (do legibility early; it doesn't depend on recall): logit-lens each value ---------
        print("\n  LEGIBILITY (by construction): logit-lens of each stored value v = ln_final->W_U.")
        nameable = 0
        for f, v in zip(facts, values):
            # logit lens: apply final layernorm then unembed. v is a residual-space direction.
            lv = model.ln_final(v.unsqueeze(0))
            lens = (lv @ W_U)[0].float()
            top = int(lens.argmax())
            ok = top == f["ans_id"]
            nameable += ok
            top_str = model.to_string(torch.tensor([top]))
            if f["label"] in (facts[0]["label"], facts[1]["label"], facts[-1]["label"]):
                print(f"    {f['label']:24} value decodes -> {top_str!r:12} (want {f['ans_word']!r})  {'OK' if ok else 'MISS'}")
        print(f"    nameable: {nameable}/{n} = {100*nameable/n:.0f}% of values decode to their answer.")

        # --- STEP 3: READ / recall, all addressing modes, full memory --------------------------------
        print("\n  STEP 3 — RECALL  (with FULL memory).  baseline shown for lift; modes compared.")
        print(f"    {'mode':9} {'P(ans)':>9} {'top1':>7} {'top5':>7}   (baseline P={base_p*100:.3f}%  "
              f"top1={fmt_pct(base_t1)}  top5={fmt_pct(base_t5)})")
        mode_rows = {}
        for mode in ["dot", "cos", "softmax", "top1"]:
            # cos/softmax/top1 weights are O(1) cosines; give them their own eta (else contribution ~0).
            m = full if mode == "dot" else full.subset(list(range(n)))
            if mode != "dot":
                for e in m.entries:
                    e["eta"] = 10.0   # cosine in [-1,1] -> eta*cos*value ~ logit-lens steer
            rows = eval_recall(m, facts, mode)
            mode_rows[mode] = rows
            print(f"    {mode:9} {agg(rows,'p')*100:8.3f}% {fmt_pct(agg(rows,'top1')):>7} "
                  f"{fmt_pct(agg(rows,'top5')):>7}")

        # per-fact table for the dot and the best sharpened mode
        best_mode = max(["dot", "cos", "softmax", "top1"], key=lambda mm: agg(mode_rows[mm], "top1"))
        print(f"\n    per-fact (dot vs best='{best_mode}'):  base_P -> dot_P / {best_mode}_P   (top1 flags)")
        for f, rd, rb in zip(facts, mode_rows["dot"], mode_rows[best_mode]):
            print(f"      {f['label']:24} {f['base_p']*100:6.3f}% -> {rd['p']*100:6.3f}% [{'T' if rd['top1'] else '.'}] "
                  f"/ {rb['p']*100:6.3f}% [{'T' if rb['top1'] else '.'}]")

        # --- STEP 4: SPECIFICITY. query fact i with ONLY one OTHER fact j!=i in memory ---------------
        print("\n  STEP 4 — SPECIFICITY (mandatory control). query fact i, memory = ONLY one fact j!=i.")
        print("    off-target recall must stay ~baseline; a uniform rise = a bias, not a store.")
        for mode in ["dot", best_mode]:
            eta_use = eta if mode == "dot" else 10.0
            # confusion: rows = query i, cols = single stored fact j ; cell = P(ans_i | cue_i, mem={j})
            offdiag_p = []
            offdiag_top1 = []
            ontarget_top1 = []
            conf = np.zeros((n, n))
            for i, fi in enumerate(facts):
                for j in range(n):
                    mj = full.subset([j])
                    for e in mj.entries:
                        e["eta"] = eta_use
                    logits = mj.recall_logits(fi["cue"], mode=mode)
                    p_i = float(F.softmax(logits, dim=-1)[fi["ans_id"]])
                    conf[i, j] = p_i
                    t1 = int(logits.argmax()) == fi["ans_id"]
                    if i == j:
                        ontarget_top1.append(t1)
                    else:
                        offdiag_p.append(p_i)
                        offdiag_top1.append(t1)
            print(f"    [{mode}] on-target top1 (mem has the RIGHT fact)   = {fmt_pct(np.mean(ontarget_top1))}")
            print(f"    [{mode}] OFF-target P(ans_i) (mem has a WRONG fact) = {np.mean(offdiag_p)*100:.3f}%   "
                  f"(baseline {base_p*100:.3f}%)")
            print(f"    [{mode}] OFF-target top1                            = {fmt_pct(np.mean(offdiag_top1))}  "
                  f"(want ~baseline {fmt_pct(base_t1)})")
            results.setdefault(L, {}).setdefault(mode, {})["offtarget_p"] = float(np.mean(offdiag_p))
            results[L][mode]["offtarget_top1"] = float(np.mean(offdiag_top1))
            results[L][mode]["ontarget_top1"] = float(np.mean(ontarget_top1))

        # --- STEP 5a: EDITABILITY — DELETE. remove entry 0; its recall must drop, others unaffected ---
        print("\n  STEP 5a — EDITABILITY / DELETE. drop entry 0; recall_0 -> baseline, others unchanged.")
        del_mode = best_mode
        eta_use = eta if del_mode == "dot" else 10.0
        full_em = full.subset(list(range(n)))
        for e in full_em.entries:
            e["eta"] = eta_use
        rows_before = eval_recall(full_em, facts, del_mode)
        kept_idx = list(range(1, n))
        deleted = full.subset(kept_idx)
        for e in deleted.entries:
            e["eta"] = eta_use
        # recompute on ALL facts but only fact 0 is the deleted one; others still in memory at shifted idx
        rows_after = eval_recall(deleted, facts, del_mode)
        p0_before, p0_after = rows_before[0]["p"], rows_after[0]["p"]
        others_before = np.mean([r["p"] for r in rows_before[1:]])
        others_after = np.mean([r["p"] for r in rows_after[1:]])
        print(f"    [{del_mode}] deleted entry 0 = {facts[0]['label']}")
        print(f"      fact 0  P(ans): before={p0_before*100:6.3f}%  after-delete={p0_after*100:6.3f}%  "
              f"(baseline {facts[0]['base_p']*100:.3f}%)  -> {'RESTORED' if p0_after < p0_before*0.5 + facts[0]['base_p'] else 'STILL RAISED'}")
        print(f"      others  meanP: before={others_before*100:6.3f}%  after={others_after*100:6.3f}%  "
              f"-> {'UNAFFECTED' if abs(others_after-others_before) < 0.02 else 'CHANGED'}")
        results[L][del_mode]["delete_p0_before"] = float(p0_before)
        results[L][del_mode]["delete_p0_after"] = float(p0_after)
        results[L][del_mode]["delete_p0_baseline"] = float(facts[0]["base_p"])
        results[L][del_mode]["delete_others_before"] = float(others_before)
        results[L][del_mode]["delete_others_after"] = float(others_after)

        # --- STEP 5b: EDITABILITY — REWEIGHT eta sweep. dose-response of P(answer) --------------------
        # Two curves, reported honestly:
        #  (i)  FULL/dot: scale ALL etas together. This also scales cross-talk from non-target entries,
        #       so P(target) can plateau/dip past the sweet spot — the raw-substrate behavior.
        #  (ii) CLEAN/top1: scale ONLY the addressed (target) entry's eta (top1 addressing isolates it).
        #       This is the true per-entry dose; expected monotone up to vocab-softmax saturation.
        print("\n  STEP 5b — EDITABILITY / REWEIGHT. eta sweep (0, 0.5, 1, 2, 4x) -> dose-response P(ans).")
        sweep = [0.0, 0.5, 1.0, 2.0, 4.0]
        dose = [agg(eval_recall(full, facts, "dot", scale=s), "p") for s in sweep]   # (i) all etas
        # (ii) clean per-target: for each fact, only its own entry is addressed (top1), eta scaled.
        clean = []
        for s in sweep:
            ps = []
            for i, fi in enumerate(facts):
                mi = full.subset([i])                       # memory = just the target entry
                mi.entries[0]["eta"] = 10.0 * s
                logits = mi.recall_logits(fi["cue"], mode="top1")
                ps.append(float(F.softmax(logits, dim=-1)[fi["ans_id"]]))
            clean.append(float(np.mean(ps)))
        print("    scale:        " + "  ".join(f"{s:>6.1f}x" for s in sweep))
        print("    P(ans) all:   " + "  ".join(f"{p*100:5.2f}%" for p in dose) + "   (dot, scales cross-talk too)")
        print("    P(ans) clean: " + "  ".join(f"{p*100:5.2f}%" for p in clean) + "   (top1, target entry only)")
        monotone = all(dose[i] <= dose[i + 1] + 1e-6 for i in range(len(dose) - 1))
        monotone_clean = all(clean[i] <= clean[i + 1] + 1e-6 for i in range(len(clean) - 1))
        print(f"    monotone non-decreasing:  all-etas={monotone}   clean-per-target={monotone_clean}")
        results[L]["dose"] = [float(x) for x in dose]
        results[L]["dose_clean"] = [float(x) for x in clean]
        results[L]["dose_monotone"] = bool(monotone)
        results[L]["dose_clean_monotone"] = bool(monotone_clean)
        results[L]["dot_recall_top1"] = float(agg(mode_rows["dot"], "top1"))
        results[L]["dot_recall_p"] = float(agg(mode_rows["dot"], "p"))
        results[L]["best_mode"] = best_mode
        results[L]["best_recall_top1"] = float(agg(mode_rows[best_mode], "top1"))
        results[L]["best_recall_p"] = float(agg(mode_rows[best_mode], "p"))
        results[L]["nameable"] = float(nameable / n)

    # ---- FINAL VERDICT -----------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT SUMMARY")
    print("=" * 92)
    print(f"  facts kept: {n}   baseline: P(ans)={base_p*100:.3f}%  top1={fmt_pct(base_t1)}  top5={fmt_pct(base_t5)}")
    for L in layers:
        r = results[L]
        bm = r["best_mode"]
        print(f"\n  L={L}:")
        print(f"    recall   dot:  top1 {fmt_pct(r['dot_recall_top1'])}  P {r['dot_recall_p']*100:.2f}%   "
              f"(baseline top1 {fmt_pct(base_t1)})")
        print(f"    recall  {bm:>5}: top1 {fmt_pct(r['best_recall_top1'])}  P {r['best_recall_p']*100:.2f}%")
        print(f"    specificity  dot:  off-target top1 {fmt_pct(r['dot']['offtarget_top1'])}  "
              f"off-target P {r['dot']['offtarget_p']*100:.3f}%  (baseline {base_p*100:.3f}%)")
        print(f"    specificity {bm:>5}: off-target top1 {fmt_pct(r[bm]['offtarget_top1'])}  "
              f"off-target P {r[bm]['offtarget_p']*100:.3f}%")
        print(f"    delete   {bm:>5}: fact0 {r[bm]['delete_p0_before']*100:.2f}% -> "
              f"{r[bm]['delete_p0_after']*100:.2f}% (base {r[bm]['delete_p0_baseline']*100:.2f}%); "
              f"others {r[bm]['delete_others_before']*100:.2f}% -> {r[bm]['delete_others_after']*100:.2f}%")
        print(f"    dose-resp all:   {'/'.join(f'{x*100:.1f}' for x in r['dose'])}%  monotone={r['dose_monotone']}")
        print(f"    dose-resp clean: {'/'.join(f'{x*100:.1f}' for x in r['dose_clean'])}%  monotone={r['dose_clean_monotone']}")
        print(f"    legibility: {r['nameable']*100:.0f}% of values nameable")

    # pick best layer by best-mode recall top1 with acceptable specificity
    def specificity_ok(L):
        bm = results[L]["best_mode"]
        return results[L][bm]["offtarget_top1"] <= base_t1 + 0.10
    ranked = sorted(layers, key=lambda L: results[L]["best_recall_top1"], reverse=True)
    best_L = next((L for L in ranked if specificity_ok(L)), ranked[0])
    print(f"\n  BEST LAYER (recall under acceptable specificity): L={best_L}  "
          f"(best mode {results[best_L]['best_mode']})")

    # save raw results for the writeup
    np.savez(os.path.join(RUNS, "p15_fastweight.npz"),
             baseline_p=base_p, baseline_top1=base_t1, baseline_top5=base_t5, n_facts=n,
             results=np.array([results], dtype=object))
    print(f"\n  saved -> {os.path.join(RUNS, 'p15_fastweight.npz')}")


if __name__ == "__main__":
    main()
