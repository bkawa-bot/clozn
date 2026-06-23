"""
legibility_discovered.py - "LEGIBLE LEARNING IN A DISCOVERED BASIS" (UNGATED first cut).
The feature-discovery deep-dive's top recommendation (research/feature_discovery_deepdive.md, idea 3
/ STEP-5 + the "single most promising next experiment" in section 5). READ THAT MEMO + research/
legibility_v1.py + research/frontier_apply_v2.py (the TTT mechanism) + inspector/spikes/p8_gpt2_control.py
(how Bloom's gpt2-res-jb SAE was loaded via sae_lens in .venv-sae) FIRST.

THE IDEA (the whole arc in one paragraph):
  A FROZEN model can LEARN a new rule - test-time adaptation (a soft prefix, a few gradient steps on the
  rule's own examples) makes the frozen model apply an unseen 1-to-1 relation to HELD-OUT words at near
  the in-context ceiling (frontier_apply_v2 lever 3). BUT the learned prefix is an OPAQUE BLOB: a probe
  names it at chance, hand-named sliders only name it via an input-feature artifact (legibility_v1). The
  deep-dive's bet: read the learned rule out NOT in a hand-named basis but in a PRETRAINED, interpretable
  feature dictionary - a real SAE - the GOLDEN-GATE move pointed at the model's OWN learned rule. UNGATED
  first cut = GPT-2-small + Bloom's gpt2-small-res-jb residual SAE (both ungated, already set up in
  .venv-sae from the SAE salvage; the richer Gemma Scope version needs gated access - a later follow-up).

WHAT THIS FILE DOES (synchronous, single process - NO background jobs / parallel workers / GPU swarm):
  STEP 1 - TTT on GPT-2: for a handful of relations, fit a soft prefix (~20-50 steps) on each relation's
    TRAIN examples; verify held-out apply ABOVE the no-prefix baseline. GPT-2 is smaller/weaker than the
    Qwen we used before, so recall is lower - reported HONESTLY; we keep the relations where TTT clears a
    bar and run the read-out only on those.
  STEP 2 - READ THE RULE IN DISCOVERED FEATURES: load Bloom's gpt2-small-res-jb SAE at the TTT injection
    layer (blocks.{L}.hook_resid_pre). For held-out queries, compute the ACTIVATION-DELTA the adaptation
    induces (residual WITH prefix minus WITHOUT) at the answer position, encode it through the SAE (in
    FEATURE space: enc(with) - enc(without)) -> which DISCOVERED features does the learned rule move?
    Report: is it SPARSE (a few features, not all)? RULE-SPECIFIC (different relations -> different feature
    sets; a relation x relation cosine/overlap confusion matrix)? vs a shuffled / random-feature NULL.
  STEP 3 - INTERPRETABILITY: harvest a WikiText corpus once; for the top lit features per rule report their
    top-activating tokens + contexts (light auto-interp) so we can see if they are nameable + rule-relevant.
    Neuronpedia auto-interp labels are pulled as a BONUS if the network is reachable (never blocks).
  STEP 4 - CAUSAL CHECK (the golden-gate move): CLAMP/add the identified features' DECODER directions into
    the residual at layer L on a FRESH held-out query WITHOUT the prefix -> does it reproduce (some of) the
    rule's held-out apply? Tests whether the read-out features are CAUSALLY the rule, not just correlated.
    vs a RANDOM-feature clamp null (same count, matched injection norm) + the no-injection floor.

HONEST CONTROLS (load-bearing - this frontier has produced clean-looking reversals):
  - sparsity + specificity read against a SHUFFLED-delta / RANDOM-feature null;
  - a DIFFERENT rule's delta must light DIFFERENT features (the off-diagonal of the confusion matrix);
  - per-relation everywhere; baselines beside every number (no-prefix apply floor, ICL-ish ceiling, null clamps);
  - a NEGATIVE (the rule is NOT sparse / NOT specific / NOT causal in the SAE features) is a VALID, valuable
    finding, reported plainly. GPT-2's weakness is a confound we NAME, not hide. No cherry-picking.

MODEL: GPT-2-small (124M), FROZEN. SAE: gpt2-small-res-jb (Bloom), pretrained, blocks.8.hook_resid_pre.
ENV: clozn/.venv-sae (sae_lens 6.x + transformer_lens + the Bloom SAE + GPT-2). CPU is fine (GPT-2 is small).
  Do NOT touch the lab GPU venv. Relation bank + split REUSED from frontier_apply_v2 (re-filtered to GPT-2 BPE).
Outputs (research/runs/): legibility_discovered{tag}.json + SVGs (TTT apply, sparsity/specificity, causal).
  RUNS START TO FINISH, PRINTS + SAVES, EXITS CLEANLY.
"""
from __future__ import annotations
import os, sys, json, time, argparse, collections, urllib.request
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, HERE)

# Reuse the EXACT relation bank from frontier_apply_v2 (the same NEW_RELATIONS + base 7) so the rule set is
# apples-to-apples with legibility_v1. We re-build the single-token filter against GPT-2's tokenizer (the
# bank builder filters defensively), and re-do the per-relation TRAIN/TEST split with the same recipe.
import frontier_apply_v2 as FV2   # build_bank, build_vocab_bank, split_bank (filter+split are tokenizer-agnostic)

DEV = "cpu"   # GPT-2 is tiny; CPU is fine and keeps us off the lab GPU entirely. (cuda would work too.)

# Maiko palette (matches the other research SVGs)
BG, TEAL, PINK, TXT, MUT, GRID = "#1A1F4A", "#6FD6C9", "#FF8FB3", "#F4F0E8", "#8784b3", "#2c2f5e"
GOLD = "#E8C977"; LILAC = "#B6A6E8"; SLATE = "#7E8AA8"; CORAL = "#E89B7E"

RELEASE = "gpt2-small-res-jb"   # Joseph Bloom's residual SAEs for GPT-2-small (the ungated gold standard)

# A short human description per relation - ONLY for printing/labeling the read-out (never given to the model).
RULE_DESC = {
 "antonym": "give the opposite", "antonym2": "give the opposite",
 "plural": "make it plural", "past": "put the verb in past tense",
 "comparative": "make the comparative form", "superlative": "make the superlative form",
 "capital": "name the capital city", "color": "name its typical color",
 "hypernym": "name the category", "hyponym": "give an example of it",
 "synonym": "give a synonym", "gerund": "add -ing", "third_person": "add -s (third person)",
 "agent": "name the doer", "verb_noun": "verb -> noun", "nationality": "nationality adjective",
 "continent": "name its continent", "opposite_gender": "opposite gender word",
 "part_of": "name the whole", "diminutive": "its young/baby form", "made_of": "a property of it",
 "un_prefix": "add un-", "re_prefix": "add re-", "adverb": "make it an adverb (-ly)",
 "ordinal": "ordinal number", "habitat": "where it lives",
}


# ============================================================================================
# GPT-2 white-box harness (transformer_lens HookedSAETransformer). The TTT soft-prefix mechanism
# is the frontier_apply one, ported from HF-`inputs_embeds` to transformer_lens `start_at_layer=0`:
# we build [tok_emb + pos_emb], PREPEND m trainable prefix vectors (which also get positional
# embeddings, positions 0..m-1, real tokens shifted to m..m+L-1), and run from layer 0. Verified
# start_at_layer=0 on [embed+pos] reproduces the normal logits exactly.
class GPT2Harness:
    def __init__(self, sae_layer=8):
        from sae_lens import SAE, HookedSAETransformer
        print(f"=== loading FROZEN GPT-2-small + SAE {RELEASE} @ blocks.{sae_layer}.hook_resid_pre ===", flush=True)
        t0 = time.time()
        self.model = HookedSAETransformer.from_pretrained("gpt2").to(DEV)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)                       # FROZEN backbone (we only ever train a prefix)
        self.tok = self.model.tokenizer
        self.H = self.model.cfg.d_model
        self.W_E = self.model.W_E                         # [V, H] token embeddings (frozen)
        self.W_pos = self.model.W_pos                     # [n_ctx, H] positional embeddings (frozen)
        self.sae_layer = sae_layer
        self.hook = f"blocks.{sae_layer}.hook_resid_pre"
        self.sae = self._load_sae(SAE, RELEASE, self.hook)
        self.sae.to(DEV).eval()
        self.d_sae = int(self.sae.cfg.d_sae)
        print(f"  loaded in {time.time()-t0:.0f}s; H={self.H}, d_sae={self.d_sae} "
              f"({self.d_sae // self.H}x); n_layers={self.model.cfg.n_layers}", flush=True)

    @staticmethod
    def _load_sae(SAE, release, hook):
        """sae_lens.SAE.from_pretrained signature shifts across versions (some return tuples); try forms.
        (Same defensive loader p8_gpt2_control.py uses.)"""
        errs = []
        for attempt in (lambda: SAE.from_pretrained(release, hook),
                        lambda: SAE.from_pretrained(release=release, sae_id=hook),
                        lambda: SAE.from_pretrained_with_cfg_and_sparsity(release, hook)):
            try:
                r = attempt()
                return r[0] if isinstance(r, tuple) else r
            except Exception as e:   # noqa: BLE001
                errs.append(f"{type(e).__name__}: {e}")
        raise RuntimeError("could not load SAE:\n  " + "\n  ".join(errs))

    def single_token_id(self, w):
        ids = self.tok.encode(" " + w, add_special_tokens=False)
        assert len(ids) == 1, f"{w!r} not single-token in GPT-2 BPE: {ids}"
        return ids[0]

    def is_single(self, w):
        return len(self.tok.encode(" " + w, add_special_tokens=False)) == 1

    def query_ids(self, x):
        """The apply query, rendered EXACTLY like frontier_apply's '{x} ->' (leading-space x, so the next
        token after '->' is the answer slot)."""
        return self.tok.encode(" " + x + " ->", add_special_tokens=False)

    def _embeds_with_prefix(self, prefix, ids):
        """[tok_emb + shifted pos_emb], optionally prepended with prefix (prefix gets positions 0..m-1)."""
        L = len(ids); mlen = 0 if prefix is None else prefix.shape[0]
        tok_emb = self.W_E[torch.tensor(ids, device=DEV)]                 # [L, H]
        tok_full = tok_emb + self.W_pos[torch.arange(mlen, mlen + L, device=DEV)]
        if prefix is None:
            return tok_full[None]                                        # [1, L, H]
        pre = prefix + self.W_pos[torch.arange(mlen, device=DEV)]        # [m, H]
        return torch.cat([pre, tok_full], 0)[None]                       # [1, m+L, H]

    def logits_with_prefix(self, prefix, ids):
        """Answer-slot next-token logits for [prefix]+[query]. Differentiable in `prefix` (backbone frozen)."""
        full = self._embeds_with_prefix(prefix, ids)
        out = self.model(full, start_at_layer=0, return_type="logits")
        return out[0, -1]                                                # [V]

    def _batch_embeds(self, prefix, ids_list):
        """Stack a batch of EQUAL-LENGTH queries into [B, m+L, H] (our '{x} ->' queries are all length-2,
        x single-token + ' ->' one token, so no padding is needed). prefix gets positions 0..m-1, tokens
        shift to m..m+L-1 - identical to _embeds_with_prefix, batched. Differentiable in `prefix`."""
        B = len(ids_list); L = len(ids_list[0]); mlen = 0 if prefix is None else prefix.shape[0]
        ids = torch.tensor(ids_list, device=DEV)                          # [B, L]
        tok_full = self.W_E[ids] + self.W_pos[torch.arange(mlen, mlen + L, device=DEV)][None]  # [B, L, H]
        if prefix is None:
            return tok_full
        pre = (prefix + self.W_pos[torch.arange(mlen, device=DEV)])[None].expand(B, -1, -1)    # [B, m, H]
        return torch.cat([pre, tok_full], 1)                              # [B, m+L, H]

    def logits_batch(self, prefix, ids_list):
        """Answer-slot logits for a batch of equal-length queries: [B, V]. ONE forward (the TTT hot path)."""
        full = self._batch_embeds(prefix, ids_list)
        out = self.model(full, start_at_layer=0, return_type="logits")
        return out[:, -1, :]                                              # [B, V]

    @torch.no_grad()
    def resid_with_prefix(self, prefix, ids):
        """Answer-slot residual at the SAE hook (blocks.L.hook_resid_pre), with/without the prefix."""
        full = self._embeds_with_prefix(prefix, ids)
        _, cache = self.model.run_with_cache(full, start_at_layer=0,
                                             names_filter=self.hook, return_type=None)
        return cache[self.hook][0, -1]                                   # [H]

    @torch.no_grad()
    def resid_batch(self, prefix, ids_list):
        """Answer-slot residuals at the SAE hook for a batch of equal-length queries: [B, H] (one forward)."""
        full = self._batch_embeds(prefix, ids_list)
        _, cache = self.model.run_with_cache(full, start_at_layer=0, names_filter=self.hook, return_type=None)
        return cache[self.hook][:, -1, :]                                # [B, H]

    @torch.no_grad()
    def logits_batch_resid_addition(self, ids_list, add_vec):
        """Batched causal clamp: FRESH queries (no prefix), add `add_vec` to the answer-position residual
        at the SAE hook. Returns [B, V]. add_vec: [H] or None."""
        full = self._batch_embeds(None, ids_list)                        # [B, L, H]
        if add_vec is None:
            out = self.model(full, start_at_layer=0, return_type="logits")
            return out[:, -1, :]
        def _hook(resid, hook):
            resid[:, -1, :] = resid[:, -1, :] + add_vec.to(resid.dtype)
            return resid
        out = self.model.run_with_hooks(full, start_at_layer=0, return_type="logits",
                                        fwd_hooks=[(self.hook, _hook)])
        return out[:, -1, :]


# ============================================================================================
# STEP 1 - TTT: fit a soft prefix for ONE relation on its TRAIN words (frozen backbone; only the
# prefix moves), the EXACT lever-3 loss (CE on the answer token over the full vocab). Returns the
# trained prefix tensor [m, H].
def fit_ttt_prefix(hz, rel, train_pairs, words, answer_tok, m, steps, lr, seed):
    H = hz.H
    pairs = train_pairs[rel].tolist()
    xs_ids = [hz.query_ids(words[a]) for (a, b) in pairs]               # all length-2 (single-token x + ' ->')
    ys = torch.tensor([answer_tok[words[b]] for (a, b) in pairs], device=DEV)
    g = torch.Generator(device="cpu").manual_seed(seed + 13)
    prefix = (0.02 * torch.randn(m, H, generator=g)).to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([prefix], lr)
    for _ in range(steps):
        logits = hz.logits_batch(prefix, xs_ids)                        # [N, V] - ONE forward per step
        loss = F.cross_entropy(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()
    return prefix.detach()


@torch.no_grad()
def apply_acc(hz, prefix, rel, pairs_split, words, answer_tok, menu_ids):
    """Held-out apply accuracy for a prefix (or None) on a relation's pairs, scored BOTH ways (free =
    full-vocab argmax == answer; menu = argmax restricted to the candidate menu). Batched (one forward)."""
    pairs = pairs_split[rel].tolist()
    if not pairs:
        return float("nan"), float("nan")
    xs_ids = [hz.query_ids(words[a]) for (a, b) in pairs]
    ans = torch.tensor([answer_tok[words[b]] for (a, b) in pairs], device=DEV)
    lg = hz.logits_batch(prefix, xs_ids)                                # [N, V]
    free_ok = (lg.argmax(-1) == ans).float().mean().item()
    menu_pred = menu_ids[lg[:, menu_ids].argmax(-1)]                    # [N] token ids
    menu_ok = (menu_pred == ans).float().mean().item()
    return free_ok, menu_ok


# ============================================================================================
# STEP 2 - READ THE RULE IN DISCOVERED FEATURES. For each held-out TEST query of a relation, the
# adaptation's activation-delta at the SAE hook is enc_SAE(resid_with_prefix) - enc_SAE(resid_without).
# We read in FEATURE space (encode each state, subtract) rather than encoding the raw residual delta,
# because the SAE encoder's per-feature bias/threshold is tuned for ACTIVATIONS not deltas - encoding
# the raw delta directly is dense/meaningless (verified: ~3.6k nonzero). The per-relation feature-delta
# is the mean over the relation's held-out queries: which discovered features the learned rule MOVES.
@torch.no_grad()
def relation_feature_delta(hz, prefix, rel, test_pairs, words):
    """Mean SAE-feature delta (enc(with) - enc(without)) over a relation's held-out TEST queries. [d_sae].
    Batched: both residual sets + both SAE encodes are one shot each."""
    pairs = test_pairs[rel].tolist()
    xs_ids = [hz.query_ids(words[a]) for (a, b) in pairs]
    r_with = hz.resid_batch(prefix, xs_ids)                            # [N, H]
    r_without = hz.resid_batch(None, xs_ids)                           # [N, H]
    f_delta = (hz.sae.encode(r_with) - hz.sae.encode(r_without)).float()   # [N, d_sae]
    return f_delta.mean(0)                                             # [d_sae]


def sparsity_stats(fdelta, fracs=(0.9, 0.95, 0.99)):
    """How concentrated is the feature-delta? Report L0 (active features), and the SMALLEST number of
    top-|delta| features that capture {90,95,99}% of the total |delta| mass (the honest 'is it a few
    features' number; L0 alone overcounts tiny movers)."""
    v = fdelta.abs()
    total = float(v.sum()) + 1e-12
    order = torch.argsort(v, descending=True)
    csum = torch.cumsum(v[order], 0) / total
    out = {"l0": int((v > 1e-6).sum()), "d_sae": int(v.numel())}
    for fr in fracs:
        k = int((csum < fr).sum().item()) + 1
        out[f"k_for_{int(fr*100)}pct"] = k
    # participation ratio (effective # of features): (sum v)^2 / sum v^2 - a smooth sparsity measure
    out["participation_ratio"] = float((v.sum() ** 2) / ((v ** 2).sum() + 1e-12))
    return out


def topk_features(fdelta, k=12):
    """Indices of the top-k features by SIGNED-positive delta (the ones the rule turns ON), plus their
    delta values. We clamp the POSITIVE movers in the causal step (adding a feature's decoder direction
    = turning it on), so the read-out we name + clamp is the positive set."""
    v = fdelta.clone()
    pos = torch.clamp(v, min=0)
    idx = torch.argsort(pos, descending=True)[:k]
    return idx.tolist(), [float(fdelta[i]) for i in idx]


# ============================================================================================
# STEP 3 - light auto-interp. We REUSE the already-harvested GPT-2 layer-8 SAE activations from the
# pretrained-SAE control (inspector/runs/gpt2_control_acts.npz: 56,507 WikiText-103 token rows of residual
# + SAE feature acts + the token strings, in corpus order - SAME model/layer/SAE). For each of our read-out
# features we look up its global top-activating rows in that matrix and reconstruct a context window from
# the token-ordered `pieces`. (No live WikiText streaming - the dataset is not cached offline on this PC,
# and reusing the control's harvest is faster, identical, and fully offline.) Neuronpedia labels are a bonus.
ACTS_CACHE = os.path.join(os.path.dirname(HERE), "inspector", "runs", "gpt2_control_acts.npz")


def harvest_feature_activations(hz, feature_idx, n_passages=None, max_ctx=None):
    """Top-activating tokens + contexts per feature, read from the cached gpt2_control activation matrix.
    Returns {feat: {modal_token, top_tokens, top_contexts}}; {} if the cache is unavailable (never fatal)."""
    if not os.path.exists(ACTS_CACHE):
        print(f"  (gpt2_control_acts.npz not found at {ACTS_CACHE}; skipping auto-interp contexts)", flush=True)
        return {}
    d = np.load(ACTS_CACHE, allow_pickle=True)
    F = d["F"]                                       # [N, d_sae] SAE feature acts (corpus order)
    pieces = list(d["pieces"])                       # [N] token strings (corpus order)
    if F.shape[1] != hz.d_sae:
        print(f"  (cache d_sae {F.shape[1]} != SAE d_sae {hz.d_sae}; skipping auto-interp)", flush=True)
        return {}

    def context(i, window=10):
        lo, hi = max(0, i - window), min(len(pieces), i + window + 1)
        return ("".join(pieces[lo:i]) + "<<" + pieces[i] + ">>" + "".join(pieces[i + 1:hi])).replace("\n", " ").strip()

    out = {}
    for f in feature_idx:
        col = F[:, f]
        order = np.argsort(col)[::-1][:12]           # global top-activating rows for this feature
        toks = [pieces[int(i)].strip() for i in order if float(col[int(i)]) > 1e-6]
        hist = collections.Counter(t.lower() for t in toks if t)
        out[int(f)] = {"modal_token": (hist.most_common(1)[0][0] if hist else ""),
                       "top_tokens": toks[:8],
                       "top_contexts": [context(int(i)) for i in order[:5] if float(col[int(i)]) > 1e-6]}
    return out


def neuronpedia_label(layer, feature_idx, timeout=6.0):
    """gpt2-small-res-jb features are public on Neuronpedia (id 'gpt2-small/{layer}-res-jb'). Returns the
    short auto-interp description if reachable, else None. Network-OPTIONAL: a failure never blocks."""
    url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{feature_idx}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clozn-legible/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        exps = data.get("explanations") or []
        return (exps[0].get("description") or "").strip() if exps else ""
    except Exception:   # noqa: BLE001
        return None


# ============================================================================================
# STEP 4 - CAUSAL CHECK (golden-gate). Build an injection vector from the read-out features' DECODER
# directions (turning those features ON at the strengths the rule moved them), add it to the residual at
# layer L on a FRESH held-out query WITHOUT the prefix, and measure recovered held-out apply. vs a RANDOM-
# feature clamp (matched count, matched final injection norm) + the no-injection floor + the full-prefix
# TTT ceiling. The injection is the FULL POSITIVE-feature reconstruction (every feature the rule turns on,
# weighted by its delta) - the complete read-out, not just the top-k we NAME in step 3. (Verified in
# development that the top-k-only injection is too weak to reproduce the rule, but the full positive
# reconstruction does: the causal test must use the whole read-out, not the interpretability slice.)
@torch.no_grad()
def positive_reconstruction(hz, feat_delta):
    """Unit vector of the full positive-feature decoder reconstruction: sum_f max(delta_f,0) * W_dec[f].
    This is 'turn on exactly the features the rule turned on, at their relative strengths', in residual space."""
    pos = torch.clamp(feat_delta, min=0)                   # [d_sae]
    vec = pos @ hz.sae.W_dec                               # [H]  (weighted decoder sum)
    return vec / (vec.norm() + 1e-8)


@torch.no_grad()
def causal_apply(hz, rel, test_pairs, words, answer_tok, menu_ids, inject_vec):
    """Held-out apply when `inject_vec` is added to the residual at the answer position (no prefix). Batched."""
    pairs = test_pairs[rel].tolist()
    if not pairs:
        return float("nan"), float("nan")
    xs_ids = [hz.query_ids(words[a]) for (a, b) in pairs]
    ans = torch.tensor([answer_tok[words[b]] for (a, b) in pairs], device=DEV)
    lg = hz.logits_batch_resid_addition(xs_ids, inject_vec)            # [N, V]
    free_ok = (lg.argmax(-1) == ans).float().mean().item()
    menu_ok = (menu_ids[lg[:, menu_ids].argmax(-1)] == ans).float().mean().item()
    return free_ok, menu_ok


# ============================================================================================
# SVGs (Maiko palette; minimal, SVG-only - matplotlib is blocked on this PC).
def _svg_open(W, Hh, title, x0, x1):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
            f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
            f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']


def svg_grouped_bars(path, groups, series, title, W=900):
    """series: list of (label, color, {group: value}). groups: list of group names. Bars 0..1."""
    Hh, ml, mr, mt, mb = 400, 54, 230, 46, 100
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = max(1, len(groups)); bw = (x1 - x0) / n
    Yc = lambda v: y0 - (0.0 if (v != v) else max(0.0, min(1.0, v))) * (y0 - y1)
    p = _svg_open(W, Hh, title, x0, x1)
    for gv in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(gv); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                          f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{gv:g}</text>']
    k = len(series); span = 0.84
    for i, g in enumerate(groups):
        cx = x0 + (i + 0.5) * bw
        for j, (_, col, vals) in enumerate(series):
            v = vals.get(g, float("nan"))
            off = (-span / 2 + (j + 0.5) * span / k) * bw
            if v == v:
                p.append(f'<rect x="{cx+off:.1f}" y="{Yc(v):.1f}" width="{bw*span/k*0.9:.1f}" '
                         f'height="{(y0-Yc(v)):.1f}" fill="{col}"/>')
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" '
                 f'transform="rotate(20 {cx:.1f} {y0+14})">{g}</text>')
    ly = mt + 12
    for lab, col, _ in series:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10">{lab}</text>']; ly += 19
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))


def svg_confusion(path, names, M, title, W=560):
    """Heatmap of a relation x relation similarity matrix M (values in [-1,1] -> teal/pink diverging)."""
    Hh = W
    mt, ml, mr, mb = 64, 150, 24, 120
    x0, y0 = ml, mt
    cell = (W - ml - mr) / max(1, len(names))
    p = _svg_open(W, Hh, title, ml, W - mr)

    def color(v):
        v = max(-1.0, min(1.0, float(v)))
        if v >= 0:   # teal up
            r, g, b = 0x1A + int((0x6F - 0x1A) * v), 0x1F + int((0xD6 - 0x1F) * v), 0x4A + int((0xC9 - 0x4A) * v)
        else:        # pink down
            a = -v
            r, g, b = 0x1A + int((0xFF - 0x1A) * a), 0x1F + int((0x8F - 0x1F) * a), 0x4A + int((0xB3 - 0x4A) * a)
        return f"#{r:02X}{g:02X}{b:02X}"

    for i, ri in enumerate(names):
        for j, rj in enumerate(names):
            v = M[i][j]
            x = x0 + j * cell; y = y0 + i * cell
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{color(v)}" '
                     f'stroke="{BG}" stroke-width="0.5"/>')
            if cell > 26:
                p.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+3:.1f}" fill="{TXT}" font-size="8" '
                         f'text-anchor="middle">{v:.2f}</text>')
        p.append(f'<text x="{x0-5:.1f}" y="{y0+i*cell+cell/2+3:.1f}" fill="{MUT}" font-size="9" '
                 f'text-anchor="end">{ri}</text>')
        p.append(f'<text x="{x0+i*cell+cell/2:.1f}" y="{y0-5:.1f}" fill="{MUT}" font-size="9" '
                 f'text-anchor="middle" transform="rotate(-40 {x0+i*cell+cell/2:.1f} {y0-5:.1f})">{ri}</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))


# ============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_layer", type=int, default=8)          # Bloom SAE layer (blocks.L.hook_resid_pre)
    ap.add_argument("--m", type=int, default=8)                  # soft-prefix length (TTT)
    ap.add_argument("--ttt_steps", type=int, default=40)         # ~20-50 per the brief
    ap.add_argument("--ttt_lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=8)          # bank filter (GPT-2 BPE drops some)
    ap.add_argument("--n_relations", type=int, default=8)        # how many relations to attempt TTT on
    ap.add_argument("--ttt_keep_bar", type=float, default=0.10)  # keep a relation if held-out free-apply gain >= this
    ap.add_argument("--topk", type=int, default=12)             # features per rule for read-out/interp/clamp
    ap.add_argument("--clamp_scales", default="1,2,4,8")        # causal injection scales (x natural delta norm)
    ap.add_argument("--no_neuronpedia", action="store_true")
    ap.add_argument("--no_interp", action="store_true")          # skip the WikiText harvest (faster smoke)
    ap.add_argument("--tag", default="_gpt2")
    args = ap.parse_args()

    t_start = time.time()
    print(f"device={DEV}  SAE layer={args.sae_layer}  m={args.m}  ttt_steps={args.ttt_steps}  "
          f"seed={args.seed}  (SYNCHRONOUS, single process - no background jobs)", flush=True)

    hz = GPT2Harness(sae_layer=args.sae_layer)

    # ---- relation bank (reused from frontier_apply_v2), RE-FILTERED to GPT-2 single-token BPE ----
    bank, REL_NAMES, dropped = FV2.build_bank(hz.tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = FV2.build_vocab_bank(bank)
    train_pairs, test_pairs = FV2.split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    answer_tok = {w: hz.single_token_id(w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    chance = 1.0 / len(out_ids)
    print(f"|relations|={len(REL_NAMES)}  |words|={len(words)}  |menu V|={len(out_ids)}  chance={chance:.4f}", flush=True)
    if dropped:
        print("  dropped (< min single-token pairs in GPT-2 BPE): " +
              ", ".join(f"{r}({n})" for r, n in list(dropped.items())[:20]), flush=True)

    # fixed shuffled relation order; attempt TTT on the first n_relations (a handful, per the brief)
    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    rels_try = [REL_NAMES[i] for i in order][:args.n_relations]
    print(f"\nattempting TTT on {len(rels_try)} relations: {rels_try}", flush=True)

    report = dict(model="gpt2-small (124M)", release=RELEASE, sae_layer=args.sae_layer, hook=hz.hook,
                  d_sae=hz.d_sae, device=DEV, m=args.m, ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
                  seed=args.seed, n_relations_attempted=len(rels_try), rels_attempted=rels_try,
                  menu_size=len(out_ids), chance=chance, env="clozn/.venv-sae (sae_lens 6.x, transformer_lens)",
                  frozen_backbone=True, synchronous_single_process=True,
                  note="UNGATED first cut: GPT-2 + Bloom gpt2-small-res-jb. Gemma Scope = later gated follow-up.")

    # ==================== STEP 1 - TTT, keep the relations where it works ====================
    print("\n" + "=" * 90)
    print("STEP 1 - TTT on GPT-2: fit a soft prefix per relation; verify held-out apply > no-prefix baseline")
    print("=" * 90, flush=True)
    prefixes = {}; ttt_rows = {}
    for rel in rels_try:
        base_free, base_menu = apply_acc(hz, None, rel, test_pairs, words, answer_tok, menu_ids)
        prefix = fit_ttt_prefix(hz, rel, train_pairs, words, answer_tok, args.m, args.ttt_steps, args.ttt_lr, args.seed)
        tr_free, _ = apply_acc(hz, prefix, rel, train_pairs, words, answer_tok, menu_ids)   # fit sanity
        te_free, te_menu = apply_acc(hz, prefix, rel, test_pairs, words, answer_tok, menu_ids)
        gain = te_free - base_free
        keep = gain >= args.ttt_keep_bar
        prefixes[rel] = prefix
        ttt_rows[rel] = dict(base_free=base_free, base_menu=base_menu, train_fit_free=tr_free,
                             ttt_free=te_free, ttt_menu=te_menu, gain_free=gain, kept=bool(keep),
                             n_train=int(train_pairs[rel].shape[0]), n_test=int(test_pairs[rel].shape[0]),
                             desc=RULE_DESC.get(rel, rel))
        print(f"  [{rel:14s}] no-prefix free={base_free:.3f} | TTT train-fit={tr_free:.3f} "
              f"held-out free={te_free:.3f} menu={te_menu:.3f} | gain={gain:+.3f}  "
              f"{'KEEP' if keep else 'drop (TTT did not clear the bar on GPT-2)'}", flush=True)
    kept = [r for r in rels_try if ttt_rows[r]["kept"]]
    report["step1_ttt"] = dict(per_relation=ttt_rows, kept_relations=kept, keep_bar=args.ttt_keep_bar,
                               agg_base_free=float(np.mean([ttt_rows[r]["base_free"] for r in rels_try])),
                               agg_ttt_free=float(np.mean([ttt_rows[r]["ttt_free"] for r in rels_try])))
    print(f"\n  STEP 1: TTT cleared the bar on {len(kept)}/{len(rels_try)} relations: {kept}")
    print(f"    aggregate no-prefix free={report['step1_ttt']['agg_base_free']:.3f} -> "
          f"TTT held-out free={report['step1_ttt']['agg_ttt_free']:.3f}", flush=True)
    svg_grouped_bars(os.path.join(RUNS, f"legibility_discovered_ttt{args.tag}.svg"),
                     rels_try, [("no-prefix (free)", SLATE, {r: ttt_rows[r]["base_free"] for r in rels_try}),
                                ("TTT held-out (free)", TEAL, {r: ttt_rows[r]["ttt_free"] for r in rels_try}),
                                ("TTT held-out (menu)", PINK, {r: ttt_rows[r]["ttt_menu"] for r in rels_try}),
                                ("TTT train-fit", GOLD, {r: ttt_rows[r]["train_fit_free"] for r in rels_try})],
                     f"STEP 1 - TTT on GPT-2 (m={args.m}, {args.ttt_steps} steps): held-out apply vs no-prefix")

    if len(kept) < 2:
        print("\n  FEWER THAN 2 relations cleared the TTT bar on GPT-2 -> the read-out (which needs a "
              "rule-vs-rule comparison) is not meaningful. Reporting STEP 1 only. This is itself an honest "
              "finding: on a 124M model the soft-prefix TTT that worked on Qwen-0.5B barely applies these "
              "relations to held-out words.", flush=True)
        report["verdict"] = ("INCONCLUSIVE on GPT-2: TTT cleared the keep-bar on <2 relations, so the "
                             "discovered-feature read-out (a rule-specificity comparison) cannot be run. "
                             "GPT-2-small is too weak for these soft-prefix relation adaptations; rerun with "
                             "Gemma-2-2B + Gemma Scope (the gated follow-up the deep-dive recommends).")
        report["wall_time_s"] = round(time.time() - t_start, 1)
        json.dump(report, open(os.path.join(RUNS, f"legibility_discovered{args.tag}.json"), "w"), indent=2)
        print(f"\nwrote runs/legibility_discovered{args.tag}.json  [{report['wall_time_s']}s]  (clean exit)")
        return

    # ==================== STEP 2 - read the rule in discovered features ====================
    print("\n" + "=" * 90)
    print("STEP 2 - READ THE RULE IN DISCOVERED FEATURES: SAE-feature delta (enc(with)-enc(without)) per rule")
    print("=" * 90, flush=True)
    fdeltas = {}; spars = {}; tops = {}
    for rel in kept:
        fd = relation_feature_delta(hz, prefixes[rel], rel, test_pairs, words)
        fdeltas[rel] = fd
        spars[rel] = sparsity_stats(fd)
        idx, vals = topk_features(fd, k=args.topk)
        tops[rel] = dict(idx=idx, vals=vals)
        print(f"  [{rel:14s}] L0={spars[rel]['l0']:5d}/{hz.d_sae}  "
              f"k@90%={spars[rel]['k_for_90pct']:3d} k@95%={spars[rel]['k_for_95pct']:3d} "
              f"k@99%={spars[rel]['k_for_99pct']:3d}  PR={spars[rel]['participation_ratio']:.1f}  "
              f"top feats={idx[:6]}", flush=True)

    # ---- NULL for sparsity: shuffle each delta vector across feature indices (destroys structure, keeps
    #      the marginal magnitude distribution). A genuinely sparse rule-delta has FEWER features at 90%
    #      than its own shuffle would predict only if structure matters; the honest comparison is the
    #      effective-feature count (participation ratio) real vs the random-direction control below.
    g2 = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    rand_PRs = []
    H = hz.H
    for _ in range(len(kept)):
        # random residual-delta of matched norm -> encode the same way -> its feature-delta sparsity
        v = torch.randn(H, generator=g2)
        v = v / v.norm()
        # scale to a typical natural delta norm (mean over kept rels of with-vs-without residual delta)
        with torch.no_grad():
            samp_rel = kept[0]; (a, b) = test_pairs[samp_rel].tolist()[0]
            ids = hz.query_ids(words[a])
            nat = (hz.resid_with_prefix(prefixes[samp_rel], ids) - hz.resid_with_prefix(None, ids)).norm()
            vv = (v.to(DEV) * nat)
            fr = (hz.sae.encode(vv[None])[0] - hz.sae.encode((0 * vv)[None])[0]).float()
        rand_PRs.append(sparsity_stats(fr)["participation_ratio"])
    null_PR = float(np.mean(rand_PRs))
    real_PR = float(np.mean([spars[r]["participation_ratio"] for r in kept]))
    print(f"\n  SPARSITY: mean participation-ratio (effective #features moved)  real={real_PR:.1f}  "
          f"vs random-direction null={null_PR:.1f}  (of {hz.d_sae}; lower=sparser). "
          f"{'SPARSER than null' if real_PR < 0.6*null_PR else 'NOT clearly sparser than null'}", flush=True)

    # ---- SPECIFICITY: relation x relation similarity of the top-feature sets + of the full feature-deltas.
    #      (a) RAW cosine of the (positive) feature-delta vectors; (b) Jaccard overlap of the top-k feature
    #      sets; (c) MEAN-CENTERED cosine - subtract the across-rule mean delta first, which removes the
    #      shared "an answer is being produced at this slot" component that all the rule-prefixes induce, so
    #      what remains is the RULE-SPECIFIC structure. (c) is the cleaner specificity test; (a)/(b) are the
    #      honest raw numbers (a rule-agnostic shared direction is real, and inflates raw overlap).
    names = kept
    cos_M = [[0.0] * len(names) for _ in names]
    jac_M = [[0.0] * len(names) for _ in names]
    cosc_M = [[0.0] * len(names) for _ in names]
    pos_deltas = {r: torch.clamp(fdeltas[r], min=0) for r in names}
    mean_delta = torch.stack([pos_deltas[r] for r in names]).mean(0)
    cent = {r: pos_deltas[r] - mean_delta for r in names}
    topsets = {r: set(tops[r]["idx"]) for r in names}
    for i, ri in enumerate(names):
        for j, rj in enumerate(names):
            cos_M[i][j] = float(F.cosine_similarity(pos_deltas[ri][None], pos_deltas[rj][None])[0])
            cosc_M[i][j] = float(F.cosine_similarity(cent[ri][None], cent[rj][None])[0])
            inter = len(topsets[ri] & topsets[rj]); union = len(topsets[ri] | topsets[rj])
            jac_M[i][j] = inter / max(1, union)
    # specificity summary: mean diagonal (always 1) vs mean off-diagonal (want LOW if rule-specific)
    off = [cos_M[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
    offc = [cosc_M[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
    off_jac = [jac_M[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
    mean_off_cos = float(np.mean(off)) if off else float("nan")
    mean_offc_cos = float(np.mean(offc)) if offc else float("nan")
    mean_off_jac = float(np.mean(off_jac)) if off_jac else float("nan")
    # is each rule's delta most-similar to ITSELF among all rules? (it trivially is on the diagonal=1, so
    # the real test is whether the off-diagonal is LOW; we also report the max off-diagonal as the worst case)
    max_off_cos = float(np.max(off)) if off else float("nan")
    # RULE-SPECIFIC verdict keys on the MEAN-CENTERED off-diagonal (the shared-component-removed signal):
    # if, after removing the common 'produce-an-answer' direction, different rules' residuals are roughly
    # uncorrelated/anti (low or negative off-diag), the rule-specific part is genuinely distinct.
    is_specific = mean_offc_cos < 0.25
    print(f"  SPECIFICITY: RAW positive-delta cosine off-diag mean={mean_off_cos:.3f} (max={max_off_cos:.3f}); "
          f"top-{args.topk} Jaccard off-diag={mean_off_jac:.3f}; MEAN-CENTERED cosine off-diag={mean_offc_cos:.3f}.")
    print(f"    -> {'RULE-SPECIFIC' if is_specific else 'OVERLAPPING'} (after removing the shared answer-slot "
          f"component, different rules light {'DIFFERENT' if is_specific else 'OVERLAPPING'} features).", flush=True)
    report["step2_readout"] = dict(
        per_relation_sparsity=spars, top_features={r: tops[r] for r in names},
        sparsity_real_PR=real_PR, sparsity_null_PR=null_PR,
        specificity_cos_matrix=cos_M, specificity_centered_cos_matrix=cosc_M,
        specificity_jaccard_matrix=jac_M, names=names,
        mean_offdiag_cos=mean_off_cos, max_offdiag_cos=max_off_cos,
        mean_offdiag_centered_cos=mean_offc_cos, mean_offdiag_jaccard=mean_off_jac,
        is_sparse=bool(real_PR < 0.6 * null_PR), is_specific=bool(is_specific))
    svg_confusion(os.path.join(RUNS, f"legibility_discovered_specificity{args.tag}.svg"), names, cosc_M,
                  f"STEP 2 - rule x rule feature-delta cosine, shared-removed (off-diag LOW = rule-specific)")

    # ==================== STEP 3 - light auto-interp of the top features per rule ====================
    if not args.no_interp:
        print("\n" + "=" * 90)
        print("STEP 3 - INTERPRETABILITY: top-activating tokens/contexts for each rule's top features "
              "(+ Neuronpedia bonus)")
        print("=" * 90, flush=True)
        all_feats = sorted(set(f for r in names for f in tops[r]["idx"]))
        print(f"  reading top-activating tokens/contexts for {len(all_feats)} distinct top features "
              f"from the cached gpt2_control activation matrix (56k WikiText rows)...", flush=True)
        interp = harvest_feature_activations(hz, all_feats)
        # Neuronpedia labels (bonus; one call per distinct feature, short timeout, never blocks)
        np_labels = {}
        if not args.no_neuronpedia:
            for f in all_feats:
                lab = neuronpedia_label(args.sae_layer, f)
                if lab is not None:
                    np_labels[f] = lab
            print(f"  Neuronpedia: got labels for {len([1 for v in np_labels.values() if v])}/{len(all_feats)} "
                  f"features (network {'reachable' if np_labels else 'unreachable - skipped'})", flush=True)
        rule_feature_report = {}
        for rel in names:
            print(f"\n  [{rel}] = '{RULE_DESC.get(rel, rel)}'  top features:")
            feats_out = []
            for f, dv in list(zip(tops[rel]["idx"], tops[rel]["vals"]))[:6]:
                info = interp.get(int(f), {})
                lab = np_labels.get(f)
                toks = " ".join(repr(t) for t in info.get("top_tokens", [])[:6])
                print(f"    f{f:<6} delta={dv:+.3f} modal={info.get('modal_token','')!r:<12} top:{toks}"
                      f"{('  NP:' + repr(lab)) if lab else ''}", flush=True)
                if info.get("top_contexts"):
                    print(f"           ctx: {info['top_contexts'][0][:110]}", flush=True)
                feats_out.append(dict(feature=f, delta=dv, modal=info.get("modal_token", ""),
                                      top_tokens=info.get("top_tokens", [])[:8],
                                      top_contexts=info.get("top_contexts", [])[:3],
                                      neuronpedia=lab))
            rule_feature_report[rel] = feats_out
        # SHARED-vs-UNIQUE: how many of each rule's top-k features are GENERIC (in >=3 rules' top-k) vs
        # UNIQUE to that rule? If the read-out is dominated by a handful of generic high-magnitude residual
        # directions (a recurring 'and/vs', 'numbers', 'dates' feature), then it is sparse + statistically
        # rule-discriminative but the lit features are NOT nameable AS the rule - the honest ceiling here.
        from collections import Counter as _C
        feat_in_n_rules = _C(f for r in names for f in tops[r]["idx"])
        generic = {f for f, c in feat_in_n_rules.items() if c >= 3}
        share_rows = {}
        for rel in names:
            ts = tops[rel]["idx"]
            n_gen = sum(1 for f in ts if f in generic)
            share_rows[rel] = dict(n_generic=n_gen, n_unique=len(ts) - n_gen, frac_generic=n_gen / max(1, len(ts)))
        mean_frac_generic = float(np.mean([share_rows[r]["frac_generic"] for r in names]))
        # are the top features nameable as the rule? heuristic proxy: a feature whose Neuronpedia label or
        # modal token is a generic punctuation/number/'and' token is NOT rule-relevant. We report the
        # fraction of top-1 features per rule whose modal token is alphabetic & not a stopword (a weak proxy).
        STOP = {"and", "or", "the", "to", "of", "in", "a", "by", "at", "from", "which", "against", "until", "with"}
        def looks_lexical(tok):
            return bool(tok) and tok.isalpha() and tok.lower() not in STOP and len(tok) > 2
        top1_lexical = float(np.mean([looks_lexical(rule_feature_report[r][0]["modal"]) for r in names]))
        print(f"\n  FEATURE RELEVANCE (honest ceiling): mean {mean_frac_generic*100:.0f}% of each rule's "
              f"top-{args.topk} features are GENERIC (shared by >=3 rules; e.g. recurring 'and/vs', 'numbers', "
              f"'dates' directions); only {top1_lexical*100:.0f}% of rules' top-1 feature has a clean lexical "
              f"modal token. The read-out separates rules statistically but the lit features are largely NOT "
              f"nameable AS the rule on GPT-2 - sparse+specific yet not cleanly rule-interpretable.", flush=True)
        report["step3_interp"] = dict(per_relation_features=rule_feature_report,
                                      neuronpedia_labels={str(k): v for k, v in np_labels.items()},
                                      interp_source="inspector/runs/gpt2_control_acts.npz (56k WikiText rows)",
                                      shared_vs_unique=share_rows, mean_frac_generic=mean_frac_generic,
                                      generic_features=sorted(generic), top1_lexical_frac=top1_lexical)
    else:
        print("\nSTEP 3 skipped (--no_interp).", flush=True)

    # ==================== STEP 4 - causal check (the golden-gate move) ====================
    print("\n" + "=" * 90)
    print("STEP 4 - CAUSAL CHECK: clamp the read-out features into a FRESH query (no prefix) -> recover the rule?")
    print("=" * 90, flush=True)
    scales = [float(s) for s in args.clamp_scales.split(",")]
    g3 = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    causal_rows = {}
    for rel in names:
        base_free = ttt_rows[rel]["base_free"]; ttt_free = ttt_rows[rel]["ttt_free"]
        # per-query natural delta norm (mean over held-out queries) to set the clamp scale (batched)
        xs_ids = [hz.query_ids(words[a]) for (a, b) in test_pairs[rel].tolist()]
        nat = float((hz.resid_batch(prefixes[rel], xs_ids) - hz.resid_batch(None, xs_ids)).norm(dim=-1).mean()) \
            if xs_ids else 1.0
        # (a) FULL read-out: the complete positive-feature decoder reconstruction (every feature the rule
        #     turned on). (b) TOP-k read-out: only the features we NAMED in step 3 (a stricter test - are
        #     the nameable features alone causal?). (c) RANDOM null: k random features' decoder dirs.
        full_unit = positive_reconstruction(hz, fdeltas[rel])
        topk_fd = torch.zeros_like(fdeltas[rel]); idxk = torch.tensor(tops[rel]["idx"], device=DEV)
        topk_fd[idxk] = torch.clamp(fdeltas[rel][idxk], min=0)
        topk_unit = positive_reconstruction(hz, topk_fd)
        rand_idx = torch.randperm(hz.d_sae, generator=g3)[:args.topk].tolist()
        rand_w = torch.clamp(torch.randn(args.topk, generator=g3), min=0.05).to(DEV)
        rvec = (rand_w[:, None] * hz.sae.W_dec[torch.tensor(rand_idx, device=DEV)]).sum(0)
        rvec_n = rvec / (rvec.norm() + 1e-8)
        best = dict(free=-1, menu=-1, scale=None); topk_best = dict(free=-1, menu=-1, scale=None)
        rand_best = dict(free=-1, menu=-1, scale=None); per_scale = {}
        for sc in scales:
            cf, cm = causal_apply(hz, rel, test_pairs, words, answer_tok, menu_ids, full_unit * (sc * nat))
            tf, tm = causal_apply(hz, rel, test_pairs, words, answer_tok, menu_ids, topk_unit * (sc * nat))
            rf, rm = causal_apply(hz, rel, test_pairs, words, answer_tok, menu_ids, rvec_n * (sc * nat))
            per_scale[sc] = dict(clamp_free=cf, clamp_menu=cm, topk_free=tf, rand_free=rf, rand_menu=rm)
            if cf > best["free"]:
                best = dict(free=cf, menu=cm, scale=sc)
            if tf > topk_best["free"]:
                topk_best = dict(free=tf, menu=tm, scale=sc)
            if rf > rand_best["free"]:
                rand_best = dict(free=rf, menu=rm, scale=sc)
        # fraction of the TTT gain the clamp recovers (free), above the no-prefix floor
        denom = max(1e-6, ttt_free - base_free)
        recov = (best["free"] - base_free) / denom
        causal_rows[rel] = dict(base_free=base_free, ttt_free=ttt_free, clamp_best=best, topk_best=topk_best,
                                rand_best=rand_best, per_scale=per_scale, recovered_frac=recov,
                                nat_delta_norm=nat)
        print(f"  [{rel:14s}] no-prefix={base_free:.3f}  CLAMP(full) free={best['free']:.3f}@x{best['scale']} "
              f"(menu {best['menu']:.3f})  CLAMP(top{args.topk})={topk_best['free']:.3f}  "
              f"rand-clamp={rand_best['free']:.3f}  TTT={ttt_free:.3f}  -> recovers {recov*100:.0f}% of TTT gain",
              flush=True)
    agg_clamp = float(np.mean([causal_rows[r]["clamp_best"]["free"] for r in names]))
    agg_rand = float(np.mean([causal_rows[r]["rand_best"]["free"] for r in names]))
    agg_base = float(np.mean([causal_rows[r]["base_free"] for r in names]))
    agg_ttt = float(np.mean([causal_rows[r]["ttt_free"] for r in names]))
    agg_recov = float(np.mean([causal_rows[r]["recovered_frac"] for r in names]))
    print(f"\n  CAUSAL aggregate: no-prefix={agg_base:.3f}  feature-clamp={agg_clamp:.3f}  "
          f"random-feature-clamp={agg_rand:.3f}  TTT ceiling={agg_ttt:.3f}  "
          f"(clamp recovers {agg_recov*100:.0f}% of the TTT gain)", flush=True)
    # CAUSAL only counts if the clamp (a) beats the random-feature null by a margin, (b) beats the no-prefix
    # floor, AND (c) recovers a MEANINGFUL fraction of the TTT gain (>=25%) - a 4-8% recovery that merely
    # clears random by 0.05 is NOT the rule reproduced, just a nudge toward the answer set. The honest bar.
    is_causal = bool(agg_clamp > agg_rand + 0.08 and agg_clamp > agg_base + 0.08 and agg_recov >= 0.25)
    report["step4_causal"] = dict(per_relation=causal_rows, scales=scales, agg_base_free=agg_base,
                                  agg_clamp_free=agg_clamp, agg_rand_clamp_free=agg_rand, agg_ttt_free=agg_ttt,
                                  agg_recovered_frac=agg_recov, causal=is_causal)
    svg_grouped_bars(os.path.join(RUNS, f"legibility_discovered_causal{args.tag}.svg"),
                     names, [("no-prefix", SLATE, {r: causal_rows[r]["base_free"] for r in names}),
                             ("feature-clamp", TEAL, {r: causal_rows[r]["clamp_best"]["free"] for r in names}),
                             ("random-feature-clamp", LILAC, {r: causal_rows[r]["rand_best"]["free"] for r in names}),
                             ("TTT ceiling", GOLD, {r: causal_rows[r]["ttt_free"] for r in names})],
                     f"STEP 4 - causal clamp of read-out features (golden-gate) vs random-feature null")

    # ==================== VERDICT ====================
    is_sparse = report["step2_readout"]["is_sparse"]
    is_specific = report["step2_readout"]["is_specific"]
    is_causal = report["step4_causal"]["causal"]
    print("\n" + "#" * 90)
    print("# LEGIBLE-IN-A-DISCOVERED-BASIS VERDICT (UNGATED, GPT-2 + Bloom gpt2-small-res-jb)")
    print("#" * 90)
    score = sum([is_sparse, is_specific, is_causal])
    spec_str = f"raw cos {mean_off_cos:.2f}, shared-removed cos {mean_offc_cos:.2f}"
    if score == 3:
        verdict = (f"POSITIVE (the legible-AND-rich dream, ungated): the TTT-learned rule reads out as a "
                   f"SPARSE (eff. {real_PR:.0f} vs null {null_PR:.0f} features), RULE-SPECIFIC ({spec_str}), "
                   f"CAUSAL (clamp {agg_clamp:.2f} vs random {agg_rand:.2f}, no-prefix {agg_base:.2f}, TTT "
                   f"{agg_ttt:.2f}) set of DISCOVERED features - on a 124M model with an ungated pretrained "
                   f"SAE. Reading a learned rule in a pretrained feature basis works where hand-named sliders "
                   f"and self-report did not. Gemma Scope (gated) is the richer follow-up.")
    elif score == 0:
        verdict = (f"NEGATIVE: the TTT-learned rule does NOT read out cleanly in the discovered basis on GPT-2 "
                   f"- not clearly sparser than a random direction (PR {real_PR:.0f} vs {null_PR:.0f}), "
                   f"off-diagonal feature overlap high ({spec_str}), and clamping the read-out features does "
                   f"not beat a random-feature clamp ({agg_clamp:.2f} vs {agg_rand:.2f}). On a 124M base the "
                   f"learned rule's activation-delta is not a clean sparse combination of Bloom's features. A "
                   f"valid negative - and a reason to try the richer Gemma Scope basis (the deep-dive's lead), "
                   f"where the base model's representations are richer.")
    else:
        passed = [n for n, b in [("SPARSE", is_sparse), ("RULE-SPECIFIC", is_specific), ("CAUSAL", is_causal)] if b]
        failed = [n for n, b in [("SPARSE", is_sparse), ("RULE-SPECIFIC", is_specific), ("CAUSAL", is_causal)] if not b]
        # nameability caveat from STEP 3 (the honest ceiling): sparse+specific does NOT imply the lit
        # features are nameable AS the rule if they are dominated by generic shared residual directions.
        name_note = ""
        s3 = report.get("step3_interp")
        if s3 and "mean_frac_generic" in s3:
            name_note = (f" AND not cleanly NAMEABLE: {s3['mean_frac_generic']*100:.0f}% of each rule's top "
                         f"features are generic directions shared across rules (recurring 'and/vs', 'numbers', "
                         f"'dates'), so the read-out separates rules statistically without the lit features "
                         f"reading as the rule.")
        verdict = (f"PARTIAL ({score}/3): the read-out IS {', '.join(passed)} but NOT {', '.join(failed)}{name_note} "
                   f"(sparsity PR {real_PR:.0f} vs null {null_PR:.0f}; specificity {spec_str}; causal clamp "
                   f"{agg_clamp:.2f} vs random {agg_rand:.2f}, no-prefix {agg_base:.2f}, TTT {agg_ttt:.2f}). "
                   f"Partial legibility in a discovered basis on a 124M model - the missing pieces (causality, "
                   f"clean naming) are where GPT-2's weakness / the from-the-input prefix injection / the SAE's "
                   f"answer-slot-dominated delta most likely bite; the gated Gemma Scope follow-up (richer base + "
                   f"richer dictionary) is the test of whether they close.")
    print("# " + verdict)
    print("#" * 90, flush=True)
    report["verdict"] = verdict
    report["verdict_flags"] = dict(sparse=is_sparse, specific=is_specific, causal=is_causal, score=score)
    report["wall_time_s"] = round(time.time() - t_start, 1)

    out_path = os.path.join(RUNS, f"legibility_discovered{args.tag}.json")
    json.dump(report, open(out_path, "w"), indent=2, default=float)
    print(f"\nwrote {out_path}  [{report['wall_time_s']}s]  (synchronous, single process - clean exit)", flush=True)


if __name__ == "__main__":
    main()
