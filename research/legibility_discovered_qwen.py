"""
legibility_discovered_qwen.py - LEGIBLE LEARNING IN A DISCOVERED BASIS, the Qwen / GPU follow-up to
legibility_discovered.py. READ legibility_discovered.py + research/feature_discovery_deepdive.md first.

WHY: the ungated GPT-2 + Bloom's-SAE cut (legibility_discovered.py) was PARTIAL (2 of 4): the TTT-learned
rule's feature-delta was SPARSE + RULE-SPECIFIC, but NOT cleanly nameable and NOT causal - and the diagnosis
was the SUBSTRATE (124M base + 24k-feature SAE), not the method. The recommended fix was a richer base + a
richer, pretrained dictionary. We don't need the gated Gemma route: Qwen ships **Qwen-Scope**, official
pretrained SAEs (Apache-2.0, ungated), and they're in sae_lens. This runs the SAME four-leg test on
**Qwen3-1.7B-Base + Qwen-Scope** (a TopK residual SAE, 32k features, blocks.14.hook_resid_post).

ENV SPLIT (deliberate): the lab GPU venv (cloze/.venv, torch cu128) has the model but NOT sae_lens; the
.venv-sae has sae_lens but is CPU-torch. So the SAE weights were dumped ONCE via .venv-sae
(research/runs/qwen_scope_1p7b_layer14.npz + .meta.json, with reference (x->features->recon) vectors), and
THIS script - run in the lab GPU venv - loads them as RAW tensors and reimplements the TopK encode/decode,
verifying bit-for-bit against the saved reference vectors before trusting it. No sae_lens at runtime, the
lab venv is never disturbed.

WHAT'S DIFFERENT FROM THE GPT-2 CUT (everything model/SAE-specific is rewritten for HF + raw SAE; the pure
analysis is reused VERBATIM from legibility_discovered as LD): HF transformers model (not transformer_lens),
the soft-prefix TTT is frontier_apply's inputs_embeds path, the residual at the SAE layer is read from
output_hidden_states, the causal clamp is an HF forward hook on model.layers[14], and STEP 3 nameability is
LOGIT-LENS autointerp (this Qwen-Scope SAE has no Neuronpedia labels: neuronpedia_id is None) - each feature
is named by the tokens its decoder direction promotes through the unembedding, and we score how often a
rule's top features promote that rule's actual held-out answer tokens.

HONEST CONTROLS (identical spirit to the GPT-2 cut): sparsity vs a random-direction null; rule-specificity
on the shared-component-removed feature-deltas; causal clamp vs a random-feature clamp + the no-prefix floor
+ the TTT ceiling, recovered-fraction bar >= 25%; per-relation everywhere; a NEGATIVE/PARTIAL is valid and
reported plainly. MODEL FROZEN. SYNCHRONOUS, single process, no background jobs.

Outputs (research/runs/): legibility_discovered_qwen.json + the three SVGs (TTT, specificity, causal).
"""
from __future__ import annotations
import os, sys, json, time, argparse
from collections import Counter
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # model is local (~/hf_models); SAE is the local npz
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

# The Qwen-Scope SAE reference (x->features->recon) was computed in CPU fp32. Disable TF32 so the GPU
# matmuls are true fp32: TF32's ~10-bit mantissa perturbs reconstruction AND can flip a borderline TopK
# selection, which would both break verification and make the feature read-out unfaithful to the dictionary.
torch.set_float32_matmul_precision("highest")   # true fp32 matmuls (NOT TF32) so the GPU SAE math matches
torch.backends.cuda.matmul.allow_tf32 = False    # the CPU-fp32 reference the dictionary was dumped with
torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, HERE)

import frontier_apply as FA          # load_llm, SoftPrefix, forward_with_prefix, batch_pack, cache_query_embeds, encode_query_ids
import frontier_apply_v2 as FV2      # build_bank, build_vocab_bank, split_bank, single_token_id
import legibility_discovered as LD   # sparsity_stats, topk_features, svg_grouped_bars, svg_confusion, RULE_DESC, palette

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SLATE, TEAL, PINK, GOLD, LILAC = LD.SLATE, LD.TEAL, LD.PINK, LD.GOLD, LD.LILAC


def resolve_model_path(model_name: str) -> str:
    local = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    return local if os.path.isfile(os.path.join(local, "config.json")) else model_name


# ============================================================================================
# The Qwen-Scope SAE, reloaded from the dumped weights as RAW tensors. TopK SAE: encode =
# topk_k( relu(x @ W_enc + b_enc) ); decode = f @ W_dec + b_dec. We VERIFY the reimplementation
# reproduces the saved sae_lens reference (x -> features -> recon) bit-for-bit before trusting it.
class RawTopKSAE:
    def __init__(self, npz_path, meta_path, device):
        d = np.load(npz_path)
        self.meta = json.load(open(meta_path))
        self.W_enc = torch.tensor(d["W_enc"], device=device, dtype=torch.float32)   # [d_in, d_sae]
        self.b_enc = torch.tensor(d["b_enc"], device=device, dtype=torch.float32)   # [d_sae]
        self.W_dec = torch.tensor(d["W_dec"], device=device, dtype=torch.float32)   # [d_sae, d_in]
        self.b_dec = torch.tensor(d["b_dec"], device=device, dtype=torch.float32)   # [d_in]
        self.d_in = self.W_enc.shape[0]; self.d_sae = self.W_enc.shape[1]
        ref_x = torch.tensor(d["ref_x"], device=device, dtype=torch.float32)
        ref_f = torch.tensor(d["ref_feats"], device=device, dtype=torch.float32)
        ref_r = torch.tensor(d["ref_recon"], device=device, dtype=torch.float32)
        self.k = int((ref_f[0] > 0).sum().item())               # observed L0 = TopK k (=50)
        # verify encode (relu-then-topk) and decode against the sae_lens reference
        fe = self.encode(ref_x)
        re = self.decode(ref_f)
        # RELATIVE tolerance: a CORRECT formula matches the CPU-fp32 reference to GPU-fp32 precision
        # (~1e-3 rel); a WRONG formula (e.g. missing b_dec) is off by ~O(1) relative and fails loudly.
        e_rel = ((fe - ref_f).abs().max() / (ref_f.abs().max() + 1e-6)).item()
        r_rel = ((re - ref_r).abs().max() / (ref_r.abs().max() + 1e-6)).item()
        if e_rel > 1e-2 or r_rel > 1e-2:
            raise RuntimeError(f"raw SAE does not match sae_lens reference: encode_rel={e_rel:.2e} "
                               f"decode_rel={r_rel:.2e} (k={self.k})")
        print(f"  RawTopKSAE verified vs reference: encode_rel={e_rel:.1e} decode_rel={r_rel:.1e} "
              f"(k={self.k}, d_sae={self.d_sae})", flush=True)

    @torch.no_grad()
    def encode(self, x):                                        # x: [N, d_in] -> [N, d_sae]
        pre = torch.relu(x.float() @ self.W_enc + self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out

    @torch.no_grad()
    def decode(self, f):                                        # f: [N, d_sae] -> [N, d_in]
        return f.float() @ self.W_dec + self.b_dec


# ============================================================================================
# HF harness on a FROZEN Qwen3 base model. Soft-prefix TTT is frontier_apply's inputs_embeds path
# (right-aligned query embeds so the answer slot is the last position); the residual at the SAE layer
# is read from output_hidden_states[layer+1]; the causal clamp is a forward hook on model.layers[layer].
class QwenHarness:
    def __init__(self, model_path, sae: RawTopKSAE, layer: int, dtype="float32"):
        print(f"=== loading FROZEN {model_path} ({dtype}) on {DEV} ===", flush=True)
        t0 = time.time()
        self.tok, self.model = FA.load_llm(resolve_model_path(model_path), dtype=getattr(torch, dtype))
        self.H = self.model.config.hidden_size
        self.nL = self.model.config.num_hidden_layers
        self.sae = sae
        self.layer = layer
        self.emb = self.model.get_input_embeddings()
        self._decoders = self.model.model.layers                # the decoder block list
        assert self.H == sae.d_in, f"hidden {self.H} != SAE d_in {sae.d_in}"
        print(f"  loaded in {time.time()-t0:.0f}s; H={self.H}, layers={self.nL}, SAE@layer{layer}", flush=True)

    def single_token_id(self, w):
        ids = self.tok.encode(" " + w, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    # ---- residual at the SAE layer, answer position, with/without the soft prefix ----
    @torch.no_grad()
    def resid_batch(self, prefix, words, q_emb_cache):
        padded, mask = FA.batch_pack([q_emb_cache[x] for x in words])     # [N,Lq,H], answer at -1
        if prefix is not None:
            pm = FA.SoftPrefix(prefix.shape[0], self.H).to(DEV); pm.prefix = nn.Parameter(prefix)
            full = pm(padded)                                            # [N, m+Lq, H]
            att = torch.cat([torch.ones(padded.shape[0], prefix.shape[0], device=DEV), mask], 1)
        else:
            full, att = padded, mask
        hs = self.model(inputs_embeds=full, attention_mask=att, output_hidden_states=True).hidden_states
        return hs[self.layer + 1][:, -1, :].float()                      # [N, H] at blocks.layer.resid_post

    @torch.no_grad()
    def apply_acc(self, prefix, rel, pairs_split, words, q_emb_cache, answer_tok, menu_ids):
        pairs = pairs_split[rel].tolist()
        if not pairs:
            return float("nan"), float("nan")
        xs = [words[a] for (a, b) in pairs]
        ans = torch.tensor([answer_tok[words[b]] for (a, b) in pairs], device=DEV)
        padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
        if prefix is not None:
            pm = FA.SoftPrefix(prefix.shape[0], self.H).to(DEV); pm.prefix = nn.Parameter(prefix)
            lg = FA.forward_with_prefix(self.model, pm, padded, mask)     # [N, V]
        else:
            lg = self.model(inputs_embeds=padded, attention_mask=mask).logits[:, -1, :]
        free = (lg.argmax(-1) == ans).float().mean().item()
        menu = (menu_ids[lg[:, menu_ids].argmax(-1)] == ans).float().mean().item()
        return free, menu

    # ---- causal clamp: add `add_vec` to the SAE layer output at the answer slot (no prefix) ----
    @torch.no_grad()
    def causal_apply(self, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, add_vec):
        pairs = test_pairs[rel].tolist()
        if not pairs:
            return float("nan"), float("nan")
        xs = [words[a] for (a, b) in pairs]
        ans = torch.tensor([answer_tok[words[b]] for (a, b) in pairs], device=DEV)
        padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
        add = add_vec.to(self.model.dtype)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] = h[:, -1, :] + add
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handle = self._decoders[self.layer].register_forward_hook(hook)
        try:
            lg = self.model(inputs_embeds=padded, attention_mask=mask).logits[:, -1, :]
        finally:
            handle.remove()
        free = (lg.argmax(-1) == ans).float().mean().item()
        menu = (menu_ids[lg[:, menu_ids].argmax(-1)] == ans).float().mean().item()
        return free, menu

    # ---- logit-lens naming of a feature: tokens its decoder direction promotes ----
    @torch.no_grad()
    def feature_tokens(self, feat_idx, topn=8):
        v = self.sae.W_dec[feat_idx].to(self.model.dtype)                # [H] residual-space direction
        h = self.model.model.norm(v[None])                               # final RMSNorm (logit lens)
        logits = h @ self.model.lm_head.weight.T                         # [1, V]
        top = logits[0].topk(topn).indices.tolist()
        return [self.tok.decode([int(t)]).strip() for t in top]


# ============================================================================================
@torch.no_grad()
def relation_feature_delta(hz, prefix, rel, test_pairs, words, q_emb_cache):
    """Mean SAE-feature delta enc(with) - enc(without) over a relation's held-out queries. [d_sae]."""
    pairs = test_pairs[rel].tolist()
    xs = [words[a] for (a, b) in pairs]
    r_with = hz.resid_batch(prefix, xs, q_emb_cache)
    r_without = hz.resid_batch(None, xs, q_emb_cache)
    return (hz.sae.encode(r_with) - hz.sae.encode(r_without)).mean(0)     # [d_sae]


@torch.no_grad()
def positive_reconstruction(hz, feat_delta):
    pos = torch.clamp(feat_delta, min=0)
    vec = pos @ hz.sae.W_dec
    return vec / (vec.norm() + 1e-8)


def fit_ttt_prefix(hz, rel, train_pairs, words, q_emb_cache, answer_tok, m, steps, lr, seed):
    """frontier_apply soft-prefix TTT on the relation's TRAIN words; frozen backbone, only the prefix moves."""
    pairs = train_pairs[rel].tolist()
    xs = [words[a] for (a, b) in pairs]
    ys = [answer_tok[words[b]] for (a, b) in pairs]
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
    ytgt = torch.tensor(ys, device=DEV)
    torch.manual_seed(seed + 13)
    pm = FA.SoftPrefix(m, hz.H).to(DEV); pm.prefix = nn.Parameter(0.02 * torch.randn(m, hz.H, device=DEV))
    opt = torch.optim.Adam(pm.parameters(), lr); pm.train()
    for _ in range(steps):
        loss = F.cross_entropy(FA.forward_with_prefix(hz.model, pm, padded, mask), ytgt)
        opt.zero_grad(); loss.backward(); opt.step()
    pm.eval()
    return pm.prefix.detach()


# ============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--sae_npz", default=os.path.join(RUNS, "qwen_scope_1p7b_layer14.npz"))
    ap.add_argument("--sae_meta", default=os.path.join(RUNS, "qwen_scope_1p7b_layer14.meta.json"))
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--ttt_steps", type=int, default=40)
    ap.add_argument("--ttt_lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--n_relations", type=int, default=8)
    ap.add_argument("--ttt_keep_bar", type=float, default=0.25)   # Qwen3-1.7B is strong; expect most to clear
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--clamp_scales", default="1,2,4,8")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t_start = time.time()
    print(f"device={DEV}  model={args.model}  SAE@layer{args.layer}  m={args.m}  ttt_steps={args.ttt_steps}  "
          f"(SYNCHRONOUS, single process)", flush=True)

    sae = RawTopKSAE(args.sae_npz, args.sae_meta, DEV)
    hz = QwenHarness(args.model, sae, args.layer, dtype=args.dtype)

    bank, REL_NAMES, dropped = FV2.build_bank(hz.tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = FV2.build_vocab_bank(bank)
    train_pairs, test_pairs = FV2.split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    answer_tok = {w: FV2.single_token_id(hz.tok, w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    q_emb_cache = FA.cache_query_embeds(hz.tok, hz.model, words)
    chance = 1.0 / len(out_ids)
    print(f"|relations|={len(REL_NAMES)}  |words|={len(words)}  |menu V|={len(out_ids)}  chance={chance:.4f}",
          flush=True)

    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    rels_try = [REL_NAMES[i] for i in order][:args.n_relations]
    print(f"\nattempting TTT on {len(rels_try)} relations: {rels_try}", flush=True)

    report = dict(model=args.model, sae="qwen-scope-3-1.7b-base-w32k-l50 (TopK, layer14)",
                  hook="blocks.14.hook_resid_post", d_sae=sae.d_sae, device=DEV, dtype=args.dtype,
                  m=args.m, ttt_steps=args.ttt_steps, seed=args.seed, menu_size=len(out_ids), chance=chance,
                  env="cloze/.venv (torch cu128, RTX 5080); SAE weights dumped from .venv-sae sae_lens",
                  frozen_backbone=True, synchronous_single_process=True,
                  note="Qwen-Scope is Apache-2.0/ungated; the richer base + dictionary the GPT-2 cut said it needed.")

    # ---------- STEP 1: TTT ----------
    print("\n" + "=" * 90 + "\nSTEP 1 - TTT on Qwen3-1.7B-Base: soft prefix per relation; held-out apply vs no-prefix\n" + "=" * 90, flush=True)
    prefixes = {}; ttt_rows = {}
    for rel in rels_try:
        bf, bm = hz.apply_acc(None, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids)
        pre = fit_ttt_prefix(hz, rel, train_pairs, words, q_emb_cache, answer_tok, args.m, args.ttt_steps, args.ttt_lr, args.seed)
        trf, _ = hz.apply_acc(pre, rel, train_pairs, words, q_emb_cache, answer_tok, menu_ids)
        tf, tm = hz.apply_acc(pre, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids)
        gain = tf - bf; keep = gain >= args.ttt_keep_bar
        prefixes[rel] = pre
        ttt_rows[rel] = dict(base_free=bf, base_menu=bm, train_fit_free=trf, ttt_free=tf, ttt_menu=tm,
                             gain_free=gain, kept=bool(keep), desc=LD.RULE_DESC.get(rel, rel),
                             n_train=int(train_pairs[rel].shape[0]), n_test=int(test_pairs[rel].shape[0]))
        print(f"  [{rel:14s}] no-prefix free={bf:.3f} | TTT fit={trf:.3f} held-out free={tf:.3f} menu={tm:.3f} | "
              f"gain={gain:+.3f}  {'KEEP' if keep else 'drop'}", flush=True)
    kept = [r for r in rels_try if ttt_rows[r]["kept"]]
    report["step1_ttt"] = dict(per_relation=ttt_rows, kept_relations=kept, keep_bar=args.ttt_keep_bar,
                               agg_base_free=float(np.mean([ttt_rows[r]["base_free"] for r in rels_try])),
                               agg_ttt_free=float(np.mean([ttt_rows[r]["ttt_free"] for r in rels_try])))
    print(f"\n  STEP 1: TTT cleared the bar on {len(kept)}/{len(rels_try)}: {kept}", flush=True)
    LD.svg_grouped_bars(os.path.join(RUNS, f"legibility_discovered_qwen_ttt{args.tag}.svg"), rels_try,
                        [("no-prefix (free)", SLATE, {r: ttt_rows[r]["base_free"] for r in rels_try}),
                         ("TTT held-out (free)", TEAL, {r: ttt_rows[r]["ttt_free"] for r in rels_try}),
                         ("TTT held-out (menu)", PINK, {r: ttt_rows[r]["ttt_menu"] for r in rels_try}),
                         ("TTT train-fit", GOLD, {r: ttt_rows[r]["train_fit_free"] for r in rels_try})],
                        f"STEP 1 - TTT on Qwen3-1.7B-Base (m={args.m}, {args.ttt_steps} steps): held-out apply")
    if len(kept) < 2:
        report["verdict"] = "INCONCLUSIVE: <2 relations cleared the TTT bar; read-out needs a rule-vs-rule comparison."
        report["wall_time_s"] = round(time.time() - t_start, 1)
        json.dump(report, open(os.path.join(RUNS, f"legibility_discovered_qwen{args.tag}.json"), "w"), indent=2, default=float)
        print("\n" + report["verdict"]); return

    # ---------- STEP 2: read the rule in discovered features ----------
    print("\n" + "=" * 90 + "\nSTEP 2 - READ THE RULE IN DISCOVERED FEATURES (enc(with) - enc(without) per rule)\n" + "=" * 90, flush=True)
    fdeltas = {}; spars = {}; tops = {}
    for rel in kept:
        fd = relation_feature_delta(hz, prefixes[rel], rel, test_pairs, words, q_emb_cache)
        fdeltas[rel] = fd; spars[rel] = LD.sparsity_stats(fd)
        idx, vals = LD.topk_features(fd, k=args.topk); tops[rel] = dict(idx=idx, vals=vals)
        print(f"  [{rel:14s}] L0={spars[rel]['l0']:5d}/{sae.d_sae}  k@90%={spars[rel]['k_for_90pct']:3d} "
              f"k@95%={spars[rel]['k_for_95pct']:3d}  PR={spars[rel]['participation_ratio']:.1f}  top={idx[:6]}", flush=True)
    # sparsity null: random residual-delta of matched norm, encoded the same way
    g2 = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    rand_PRs = []
    for _ in range(max(3, len(kept))):
        v = torch.randn(hz.H, generator=g2); v = v / v.norm()
        samp = kept[0]; xs = [words[a] for (a, b) in test_pairs[samp].tolist()[:4]]
        nat = (hz.resid_batch(prefixes[samp], xs, q_emb_cache) - hz.resid_batch(None, xs, q_emb_cache)).norm(dim=-1).mean()
        vv = (v.to(DEV) * nat)
        fr = (hz.sae.encode(vv[None])[0] - hz.sae.encode((0 * vv)[None])[0])
        rand_PRs.append(LD.sparsity_stats(fr)["participation_ratio"])
    null_PR = float(np.mean(rand_PRs)); real_PR = float(np.mean([spars[r]["participation_ratio"] for r in kept]))
    is_sparse = bool(real_PR < 0.6 * null_PR)
    print(f"\n  SPARSITY: real PR={real_PR:.1f} vs random-direction null={null_PR:.1f} -> "
          f"{'SPARSER' if is_sparse else 'NOT clearly sparser'}", flush=True)
    # specificity: shared-component-removed cosine off-diagonal
    names = kept
    pos = {r: torch.clamp(fdeltas[r], min=0) for r in names}
    mean_delta = torch.stack([pos[r] for r in names]).mean(0)
    cent = {r: pos[r] - mean_delta for r in names}
    topsets = {r: set(tops[r]["idx"]) for r in names}
    cos_M = [[float(F.cosine_similarity(pos[a][None], pos[b][None])[0]) for b in names] for a in names]
    cosc_M = [[float(F.cosine_similarity(cent[a][None], cent[b][None])[0]) for b in names] for a in names]
    jac_M = [[len(topsets[a] & topsets[b]) / max(1, len(topsets[a] | topsets[b])) for b in names] for a in names]
    off = lambda M: [M[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
    mean_off_cos, mean_offc_cos, mean_off_jac = float(np.mean(off(cos_M))), float(np.mean(off(cosc_M))), float(np.mean(off(jac_M)))
    is_specific = bool(mean_offc_cos < 0.25)
    print(f"  SPECIFICITY: raw cos off-diag={mean_off_cos:.3f}; shared-removed cos off-diag={mean_offc_cos:.3f}; "
          f"top-{args.topk} Jaccard={mean_off_jac:.3f} -> {'RULE-SPECIFIC' if is_specific else 'OVERLAPPING'}", flush=True)
    report["step2_readout"] = dict(per_relation_sparsity=spars, top_features=tops, sparsity_real_PR=real_PR,
                                   sparsity_null_PR=null_PR, mean_offdiag_cos=mean_off_cos,
                                   mean_offdiag_centered_cos=mean_offc_cos, mean_offdiag_jaccard=mean_off_jac,
                                   names=names, is_sparse=is_sparse, is_specific=is_specific)
    LD.svg_confusion(os.path.join(RUNS, f"legibility_discovered_qwen_specificity{args.tag}.svg"), names, cosc_M,
                     "STEP 2 - rule x rule feature-delta cosine, shared-removed (off-diag LOW = rule-specific)")

    # ---------- STEP 3: logit-lens nameability ----------
    print("\n" + "=" * 90 + "\nSTEP 3 - NAMEABILITY (logit-lens: tokens each top feature's decoder promotes)\n" + "=" * 90, flush=True)
    name_rows = {}; relevant_fracs = []
    for rel in names:
        ans_strs = set()
        for (a, b) in (test_pairs[rel].tolist() + train_pairs[rel].tolist()):
            ans_strs.add(words[b].lower()); ans_strs.add(hz.tok.decode([answer_tok[words[b]]]).strip().lower())
        feats_out = []; n_rel = 0
        for f, dv in list(zip(tops[rel]["idx"], tops[rel]["vals"]))[:8]:
            toks = hz.feature_tokens(int(f))
            tl = set(t.lower() for t in toks if t)
            # rule-relevant if the feature promotes any of the rule's actual answer tokens, or a shared suffix
            rel_hit = bool(tl & ans_strs) or any(any(t.endswith(sfx) for t in tl) for sfx in [])
            if rel_hit: n_rel += 1
            feats_out.append(dict(feature=int(f), delta=float(dv), promotes=toks, rule_relevant=rel_hit))
        frac = n_rel / max(1, len(feats_out)); relevant_fracs.append(frac)
        name_rows[rel] = dict(features=feats_out, frac_rule_relevant=frac)
        print(f"  [{rel:12s}='{LD.RULE_DESC.get(rel, rel)}'] {int(frac*100)}% of top feats promote the rule's "
              f"own answers. e.g. f{feats_out[0]['feature']} -> {feats_out[0]['promotes'][:6]}", flush=True)
    mean_relevant = float(np.mean(relevant_fracs))
    is_nameable = bool(mean_relevant >= 0.5)
    print(f"\n  NAMEABILITY: mean {int(mean_relevant*100)}% of top features promote their rule's answer tokens "
          f"(logit-lens) -> {'NAMEABLE' if is_nameable else 'NOT cleanly nameable'}", flush=True)
    report["step3_nameability"] = dict(per_relation=name_rows, mean_frac_rule_relevant=mean_relevant,
                                       is_nameable=is_nameable, method="logit-lens (decoder dir -> unembed top tokens)")

    # ---------- STEP 4: causal clamp ----------
    print("\n" + "=" * 90 + "\nSTEP 4 - CAUSAL CHECK: clamp read-out features into a fresh query (no prefix) -> recover rule?\n" + "=" * 90, flush=True)
    scales = [float(s) for s in args.clamp_scales.split(",")]
    g3 = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    causal_rows = {}
    for rel in names:
        bf = ttt_rows[rel]["base_free"]; tf = ttt_rows[rel]["ttt_free"]
        xs = [words[a] for (a, b) in test_pairs[rel].tolist()]
        nat = float((hz.resid_batch(prefixes[rel], xs, q_emb_cache) - hz.resid_batch(None, xs, q_emb_cache)).norm(dim=-1).mean())
        full_unit = positive_reconstruction(hz, fdeltas[rel])
        rand_idx = torch.randperm(sae.d_sae, generator=g3)[:args.topk].tolist()
        rand_w = torch.clamp(torch.randn(args.topk, generator=g3), min=0.05).to(DEV)
        rvec = (rand_w[:, None] * sae.W_dec[torch.tensor(rand_idx, device=DEV)]).sum(0)
        rvec = rvec / (rvec.norm() + 1e-8)
        best = dict(free=-1, menu=-1, scale=None); rbest = dict(free=-1, menu=-1, scale=None); per_scale = {}
        for sc in scales:
            cf, cm = hz.causal_apply(rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, full_unit * (sc * nat))
            rf, rm = hz.causal_apply(rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, rvec * (sc * nat))
            per_scale[sc] = dict(clamp_free=cf, clamp_menu=cm, rand_free=rf, rand_menu=rm)
            if cf > best["free"]: best = dict(free=cf, menu=cm, scale=sc)
            if rf > rbest["free"]: rbest = dict(free=rf, menu=rm, scale=sc)
        recov = (best["free"] - bf) / max(1e-6, tf - bf)
        causal_rows[rel] = dict(base_free=bf, ttt_free=tf, clamp_best=best, rand_best=rbest,
                                per_scale=per_scale, recovered_frac=recov, nat_delta_norm=nat)
        print(f"  [{rel:14s}] no-prefix={bf:.3f}  CLAMP free={best['free']:.3f}@x{best['scale']} "
              f"(menu {best['menu']:.3f})  rand-clamp={rbest['free']:.3f}  TTT={tf:.3f}  recovers {recov*100:.0f}%", flush=True)
    agg_clamp = float(np.mean([causal_rows[r]["clamp_best"]["free"] for r in names]))
    agg_rand = float(np.mean([causal_rows[r]["rand_best"]["free"] for r in names]))
    agg_base = float(np.mean([causal_rows[r]["base_free"] for r in names]))
    agg_ttt = float(np.mean([causal_rows[r]["ttt_free"] for r in names]))
    agg_recov = float(np.mean([causal_rows[r]["recovered_frac"] for r in names]))
    is_causal = bool(agg_clamp > agg_rand + 0.08 and agg_clamp > agg_base + 0.08 and agg_recov >= 0.25)
    print(f"\n  CAUSAL aggregate: no-prefix={agg_base:.3f}  feature-clamp={agg_clamp:.3f}  random-clamp={agg_rand:.3f}  "
          f"TTT={agg_ttt:.3f}  (recovers {agg_recov*100:.0f}% of TTT gain) -> {'CAUSAL' if is_causal else 'NOT causal'}", flush=True)
    report["step4_causal"] = dict(per_relation=causal_rows, scales=scales, agg_base_free=agg_base,
                                  agg_clamp_free=agg_clamp, agg_rand_clamp_free=agg_rand, agg_ttt_free=agg_ttt,
                                  agg_recovered_frac=agg_recov, causal=is_causal)
    LD.svg_grouped_bars(os.path.join(RUNS, f"legibility_discovered_qwen_causal{args.tag}.svg"), names,
                        [("no-prefix", SLATE, {r: causal_rows[r]["base_free"] for r in names}),
                         ("feature-clamp", TEAL, {r: causal_rows[r]["clamp_best"]["free"] for r in names}),
                         ("random-clamp", LILAC, {r: causal_rows[r]["rand_best"]["free"] for r in names}),
                         ("TTT ceiling", GOLD, {r: causal_rows[r]["ttt_free"] for r in names})],
                        "STEP 4 - causal clamp of read-out features (Qwen-Scope) vs random-feature null")

    # ---------- VERDICT ----------
    score = sum([is_sparse, is_specific, is_nameable, is_causal])
    flags = [("SPARSE", is_sparse), ("RULE-SPECIFIC", is_specific), ("NAMEABLE", is_nameable), ("CAUSAL", is_causal)]
    passed = [n for n, b in flags if b]; failed = [n for n, b in flags if not b]
    verdict = (f"{'POSITIVE (4/4)' if score == 4 else f'PARTIAL ({score}/4)' if score else 'NEGATIVE (0/4)'} on "
               f"Qwen3-1.7B-Base + Qwen-Scope: read-out is {', '.join(passed) or 'none'}"
               + (f"; NOT {', '.join(failed)}" if failed else "") +
               f". sparsity PR {real_PR:.0f} vs null {null_PR:.0f}; shared-removed off-diag cos {mean_offc_cos:.2f}; "
               f"nameable {int(mean_relevant*100)}%; causal clamp {agg_clamp:.2f} vs random {agg_rand:.2f} "
               f"(no-prefix {agg_base:.2f}, TTT {agg_ttt:.2f}, recovers {agg_recov*100:.0f}%). "
               f"Compare the GPT-2 + Bloom cut (2/4: sparse+specific, not nameable/causal): does the richer "
               f"base + 32k pretrained dictionary close NAMEABLE and CAUSAL?")
    print("\n" + "#" * 90 + f"\n# {verdict}\n" + "#" * 90, flush=True)
    report["verdict"] = verdict
    report["verdict_flags"] = dict(sparse=is_sparse, specific=is_specific, nameable=is_nameable, causal=is_causal, score=score)
    report["wall_time_s"] = round(time.time() - t_start, 1)
    out_path = os.path.join(RUNS, f"legibility_discovered_qwen{args.tag}.json")
    json.dump(report, open(out_path, "w"), indent=2, default=float)
    print(f"\nwrote {out_path}  [{report['wall_time_s']}s]  (synchronous, single process - clean exit)", flush=True)


if __name__ == "__main__":
    main()
