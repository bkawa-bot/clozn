"""
frontier_apply.py - the "uses its own clay" rung: can the FROZEN model APPLY its own
consolidated memory, with its OWN forward pass + unembedding head?

WHY THIS RUNG (read research/sidecar_semantic_findings.md FIRST):
  sidecar_semantic.py found a sidecar can CONSOLIDATE *which* relation it was shown (real >>
  random; a linear probe names the relation at 1.000) BUT a small EXTERNAL read-MLP cannot
  APPLY a 1-to-1 relation to a HELD-OUT word: held-out exact retrieval ~0.105 aggregate and
  exactly 0.000 on every 1-to-1 relation (antonym, plural, past, comparative, capital). THE
  KEY DIAGNOSTIC: native in-context learning NAILS every relation - the same frozen model,
  given the K examples in its PROMPT, applies them to held-out words at 0.94-1.00. So the
  information is present and the model CAN apply it; the external reader just cannot extract it.

THE HYPOTHESIS (this file): let the FROZEN MODEL apply the relation with its OWN machinery
  (its own forward pass + unembedding head), driven by an INJECTED consolidated state - NOT an
  external read-MLP that produces the answer. The predicted answer MUST be the frozen model's
  OWN next-token output.

STAGE 1 - DIAGNOSTIC (always runs; GATES stage 2):
  For each relation R, learn a SOFT PREFIX (m continuous embedding vectors prepended to the
  query) by backprop through the FROZEN model (prefix tuning: ONLY the prefix trains, backbone
  frozen) so that  [soft-prefix] + [query "x ->"]  makes the MODEL'S OWN next token = R(x).
  Train on TRAIN words; EVAL on HELD-OUT words (never in training). On held-out words, compare:
    - soft-prefix (the test)           <- model's own output, driven by a learned injection
    - ICL ceiling (K real text pairs in the prompt; upper bound ~0.9-1.0)
    - read-MLP failure (~0.105 / 0.000 exact; the sidecar_semantic p18 result, loaded from disk)
    - null: NO prefix and RANDOM prefix (chance floor for "the model alone")
  THE QUESTION: can a learned injection make the frozen model APPLY a relation to HELD-OUT words
  where the external read-MLP failed? YES (>> read-MLP + null, approaching ICL) -> stage 2.
  NO -> report plainly + diagnose + STOP (that kills the simplest version of the frontier and
  is itself the valuable finding).

STAGE 2 - REAL CONSOLIDATION (only if stage 1 clearly works):
  Meta-learn a COMPRESSOR mapping K example activations -> the soft prefix, across MANY
  relations. A few examples of a NEW held-out relation then produce a prefix that makes the
  frozen model apply it to held-out words with NO examples in context. vs ICL + read-MLP + null.

LEGIBILITY: probe the learned per-relation prefixes -> which relation is it (like
  sidecar_semantic's relation probe)? Does the injection stay legible?

CONTROLS / HONESTY (load-bearing - the frontier has produced clean-looking reversals before):
  every number sits beside the ICL ceiling + the read-MLP baseline + a no/random-injection null;
  eval ONLY on held-out words; aggregate over relations + seeds with a per-relation breakdown;
  NO cherry-picking relation / layer / prefix length. A NEGATIVE is the MOST valuable outcome.

REUSE (apples-to-apples with the read-MLP result): RELATIONS + split_relations + the carrier
  and ICL-scoring conventions are imported directly from sidecar_semantic.py, so the held-out
  TRAIN/TEST split is byte-identical to the read-MLP run.

MODEL: Qwen2.5-0.5B-Instruct, FROZEN. Env: cloze/.venv (torch cu128, RTX 5080).
Outputs (research/runs/): frontier_apply{tag}.json + frontier_apply_stage1{tag}.svg (+ stage2 if reached).
"""
import os, sys, json, time, argparse, math
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.set_float32_matmul_precision("high")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, HERE)

# Reuse the EXACT relation data + held-out split from the read-MLP rung (apples-to-apples).
from sidecar_semantic import (RELATIONS, REL_NAMES, build_vocab, split_relations, CARRIER,
                              build_menu_rel)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Maiko palette (matches sidecar_semantic.py)
BG, TEAL, PINK, TXT, MUT, GRID = "#1A1F4A", "#6FD6C9", "#FF8FB3", "#F4F0E8", "#8784b3", "#2c2f5e"
GOLD = "#E8C977"    # ICL ceiling
LILAC = "#B6A6E8"   # read-MLP baseline
SLATE = "#7E8AA8"   # null

# ----------------------------------------------------------------------------------------
def load_llm(model_name, dtype=torch.float32):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)            # FROZEN backbone (defensive; we only train the prefix)
    return tok, model

def single_token_id(tok, w):
    ids = tok.encode(" " + w, add_special_tokens=False)
    assert len(ids) == 1, f"{w!r} not single-token: {ids}"
    return ids[0]

# ----------------------------------------------------------------------------------------
# The query is rendered EXACTLY as the native-ICL query line, so the model's next token after
# "->" is the answer slot. The soft prefix occupies the role the ICL examples+instruction play.
#   query text = "{x} ->"   (leading-space answer token follows, matching ICL scoring)
# We embed it, prepend m trainable prefix vectors, and read the final-position next-token logits.
QUERY_TMPL = "{x} ->"

def encode_query_ids(tok, word):
    return tok.encode(QUERY_TMPL.format(x=word), add_special_tokens=False)

class SoftPrefix(nn.Module):
    """m trainable embedding vectors prepended to the (frozen) input embeddings of the query.
    Initialized from real token embeddings (a light, reported prior - not cherry-picked per
    relation) so optimization starts in-distribution. Only this tensor trains."""
    def __init__(self, m, H, init_emb=None):
        super().__init__()
        if init_emb is not None:
            p = init_emb.clone()
        else:
            p = 0.02 * torch.randn(m, H)
        self.prefix = nn.Parameter(p)              # [m, H]
    def forward(self, q_embeds):                   # q_embeds: [B, Lq, H]
        B = q_embeds.shape[0]
        pre = self.prefix[None].expand(B, -1, -1)  # [B, m, H]
        return torch.cat([pre, q_embeds], 1)       # [B, m+Lq, H]

@torch.no_grad()
def cache_query_embeds(tok, model, words):
    """Cache (input-embedding sequence, length, answer-position) per query word. The embedding
    table is frozen; we cache the lookups so training does not re-tokenize."""
    emb = model.get_input_embeddings()
    cache = {}
    for w in words:
        ids = encode_query_ids(tok, w)
        e = emb(torch.tensor(ids, device=DEV)).detach()    # [Lq, H]
        cache[w] = e
    return cache

def batch_pack(embeds_list):
    """Left-pad a list of [Lq,H] query-embed tensors to a common length; the answer position is
    always the LAST real token, so right-alignment keeps the answer at index -1 for all rows.
    Returns padded embeds [B, Lmax, H] and an attention mask [B, Lmax] (1=real)."""
    H = embeds_list[0].shape[1]
    Ls = [e.shape[0] for e in embeds_list]
    Lmax = max(Ls)
    B = len(embeds_list)
    out = torch.zeros(B, Lmax, H, device=DEV)
    mask = torch.zeros(B, Lmax, device=DEV)
    for i, e in enumerate(embeds_list):
        L = e.shape[0]
        out[i, Lmax - L:] = e                  # right-align (answer token at -1)
        mask[i, Lmax - L:] = 1.0
    return out, mask

def forward_with_prefix(model, prefix_mod, q_embeds, q_mask):
    """Run the frozen model on [prefix] + [right-aligned query embeds]; return next-token logits
    at the final position (the answer slot). Prefix tokens are always attended (mask=1)."""
    full = prefix_mod(q_embeds)                                  # [B, m+Lmax, H]
    B, m = q_embeds.shape[0], prefix_mod.prefix.shape[0]
    pre_mask = torch.ones(B, m, device=DEV)
    att = torch.cat([pre_mask, q_mask], 1)                       # [B, m+Lmax]
    out = model(inputs_embeds=full, attention_mask=att)
    return out.logits[:, -1, :]                                  # [B, V] next-token at answer slot

# ----------------------------------------------------------------------------------------
def train_soft_prefix_for_relation(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok,
                                    m=8, steps=400, lr=0.05, seed=0, init_from_emb=True,
                                    log_every=0):
    """STAGE 1 core: learn ONE soft prefix so the frozen model's own next token = R(x) for the
    relation's TRAIN words. Cross-entropy on the answer token id, over the FULL vocab (the model's
    real unembedding head). Only the prefix trains."""
    torch.manual_seed(seed)
    H = model.config.hidden_size
    emb = model.get_input_embeddings()
    init = None
    if init_from_emb:
        # init prefix from m random real token embeddings (reported prior, same recipe all relations)
        g = torch.Generator().manual_seed(seed + 12345)
        vids = torch.randint(0, emb.weight.shape[0], (m,), generator=g)
        init = emb.weight[vids.to(DEV)].detach().float().cpu()
    prefix = SoftPrefix(m, H, init_emb=init).to(DEV)
    opt = torch.optim.Adam(prefix.parameters(), lr)

    pairs = train_pairs[rel].tolist()                            # [(xi, yi), ...] global word idx
    xs = [words[xi] for (xi, yi) in pairs]
    ys = [answer_tok[words[yi]] for (xi, yi) in pairs]           # answer token ids
    q_embeds = [q_emb_cache[x] for x in xs]
    padded, mask = batch_pack(q_embeds)
    ytgt = torch.tensor(ys, device=DEV)                          # [N]

    prefix.train()
    for step in range(steps):
        logits = forward_with_prefix(model, prefix, padded, mask)   # [N, V]
        loss = F.cross_entropy(logits, ytgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            with torch.no_grad():
                acc = (logits.argmax(-1) == ytgt).float().mean().item()
            print(f"      [{rel}] step {step:4d}  loss {loss.item():.3f}  train-acc {acc:.3f}")
    prefix.eval()
    return prefix

@torch.no_grad()
def eval_soft_prefix(model, prefix, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids,
                     out_set_idx):
    """Held-out apply accuracy for a learned prefix, scored TWO ways (both = model's own output):
      - menu  : argmax of next-token logits RESTRICTED to the candidate menu (matches the ICL /
                read-MLP retrieval scoring; the apples-to-apples number)
      - free  : argmax over the FULL vocab equals the answer token (strictest 'own output' test)
    Held-out only (test pairs whose x never appeared in this relation's training)."""
    pairs = test_pairs[rel].tolist()
    if not pairs:
        return float("nan"), float("nan"), 0
    xs = [words[xi] for (xi, yi) in pairs]
    ytok = torch.tensor([answer_tok[words[yi]] for (xi, yi) in pairs], device=DEV)   # [N]
    ymenu = torch.tensor([out_set_idx[words[yi]] for (xi, yi) in pairs], device=DEV) # [N]
    padded, mask = batch_pack([q_emb_cache[x] for x in xs])
    logits = forward_with_prefix(model, prefix, padded, mask)     # [N, V]
    free_ok = (logits.argmax(-1) == ytok)
    menu_pred = logits[:, menu_ids].argmax(-1)                    # [N] index into menu
    menu_ok = (menu_pred == ymenu)
    return menu_ok.float().mean().item(), free_ok.float().mean().item(), len(pairs)

@torch.no_grad()
def eval_null_prefix(model, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, out_set_idx,
                     m, seed, mode="random"):
    """NULL: the model alone on the bare query, with either NO prefix (mode='none') or a RANDOM
    untrained prefix (mode='random'). Floor for 'can the frozen model do it without a learned
    injection'. Scored identically (menu + free)."""
    H = model.config.hidden_size
    pairs = test_pairs[rel].tolist()
    if not pairs:
        return float("nan"), float("nan")
    xs = [words[xi] for (xi, yi) in pairs]
    ytok = torch.tensor([answer_tok[words[yi]] for (xi, yi) in pairs], device=DEV)
    ymenu = torch.tensor([out_set_idx[words[yi]] for (xi, yi) in pairs], device=DEV)
    padded, mask = batch_pack([q_emb_cache[x] for x in xs])
    if mode == "none":
        out = model(inputs_embeds=padded, attention_mask=mask)
        logits = out.logits[:, -1, :]
    else:
        g = torch.Generator().manual_seed(seed + 777)
        emb = model.get_input_embeddings()
        vids = torch.randint(0, emb.weight.shape[0], (m,), generator=g).to(DEV)
        pre = SoftPrefix(m, H, init_emb=emb.weight[vids].detach().float().cpu()).to(DEV)
        logits = forward_with_prefix(model, pre, padded, mask)
    free_ok = (logits.argmax(-1) == ytok).float().mean().item()
    menu_ok = (logits[:, menu_ids].argmax(-1) == ymenu).float().mean().item()
    return menu_ok, free_ok

@torch.no_grad()
def cross_relation_apply(model, prefix, rel_of_prefix, other_words, words, q_emb_cache, menu_ids,
                         out_words, menu_rel):
    """CROSS-RELATION control: feed relation R's prefix the query words of OTHER relations. If the
    prefix encodes the RELATION (a transform), its output should still be an R-TYPE answer; if it
    merely memorized R's answer set / a fixed bias, it can't know. We measure: fraction of these
    foreign-word predictions that land on an answer of R's own type (menu_rel). High = the prefix
    transports the relation, applying R's transform to words it was never trained on AND that don't
    even belong to R - the cleanest evidence it's an applicable rule, not a lookup."""
    if not other_words:
        return float("nan")
    padded, mask = batch_pack([q_emb_cache[x] for x in other_words])
    logits = forward_with_prefix(model, prefix, padded, mask)
    pred_menu = logits[:, menu_ids].argmax(-1)                # [N] index into menu/out_words
    ri = REL_NAMES.index(rel_of_prefix)
    in_R_type = menu_rel[pred_menu, ri]                       # [N] bool: predicted word is an R-type answer
    return in_R_type.float().mean().item()

# ----------------------------------------------------------------------------------------
@torch.no_grad()
def icl_ceiling_rel(tok, model, rel, train_pairs, test_pairs, words, menu_ids, out_set_idx,
                    K=5, n_episodes=200, seed=999):
    """Native-ICL ceiling for ONE relation, scored over the candidate menu - SAME recipe as
    sidecar_semantic.icl_ceiling (so directly comparable to the read-MLP rung's ceiling)."""
    gg = torch.Generator().manual_seed(seed)
    tr = train_pairs[rel].tolist(); te = test_pairs[rel].tolist()
    if not te:
        return float("nan")
    correct = total = 0
    for ep in range(n_episodes):
        kk = min(K, len(tr))
        ti = torch.randperm(len(tr), generator=gg).tolist()[:kk]
        qi = int(torch.randint(0, len(te), (1,), generator=gg).item())
        xq_i, yq_i = te[qi]
        lines = [f"{words[tr[i][0]]} -> {words[tr[i][1]]}" for i in ti]
        prompt = "Complete the analogy with the same kind of relation.\n" + "\n".join(lines) + f"\n{words[xq_i]} ->"
        ids = tok.encode(prompt, add_special_tokens=False)
        logits = model(torch.tensor(ids, device=DEV)[None, :]).logits[0, -1]
        if int(logits[menu_ids].argmax().item()) == out_set_idx[words[yq_i]]:
            correct += 1
        total += 1
    return correct / max(1, total)

# ----------------------------------------------------------------------------------------
def load_readmlp_baseline():
    """Pull the read-MLP held-out result from sidecar_semantic_0p5b.json (the exact p18 numbers),
    so the comparison is to the recorded run, not a re-description. Returns per-relation exact +
    aggregate + ICL recorded there, or None if absent."""
    path = os.path.join(RUNS, "sidecar_semantic_0p5b.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    real = d["controls"]["real"]
    return dict(per_relation=real["per_relation"],
                aggregate=real["genK"][str(max(d["Ks"]))]["acc"],
                within=real.get("within_acc"), cluster=real.get("cluster_acc"),
                icl=d.get("icl", {}), chance=d["chance"],
                relation_probe=d["legibility"]["relation_probe_acc"])

# ----------------------------------------------------------------------------------------
# LEGIBILITY: probe the learned per-relation prefixes -> which relation. Each relation has one
# INDEPENDENTLY-OPTIMIZED prefix per seed (different random init each seed). We mean-center to
# remove the common init/scale offset (a raw-flatten probe is dominated by init), then run a
# leave-one-out nearest-centroid probe over all (rel,seed) prefixes (chance = 1/R). NOTE: because
# each prefix is an INDEPENDENT optimization with no shared encoder, same-relation prefixes need
# NOT land near each other - so a chance-level result here is the EXPECTED, honest reading (the
# *behavior* is relation-specific; the *parameters that implement it* are not relation-aligned).
# This is the natural contrast to sidecar_semantic's shared-encoder state (probe 1.000); a
# SHARED meta-learned compressor (stage 2) is where legible-prefix would be expected.
def legibility_probe(prefix_vecs):
    """prefix_vecs: {rel: [n_seeds, m*H]}. LOO nearest-centroid acc over centered prefixes."""
    rels = [r for r in REL_NAMES if r in prefix_vecs]
    X, y = [], []
    for ri, r in enumerate(rels):
        for v in prefix_vecs[r]:
            X.append(v); y.append(ri)
    X = torch.stack(X).float(); y = torch.tensor(y)
    X = X - X.mean(0, keepdim=True)                  # remove common init/scale offset
    X = F.normalize(X, dim=-1)                       # cosine geometry
    n = X.shape[0]
    if n <= len(rels):                               # need >=2 per class for a held-out centroid
        return float("nan"), len(rels)
    correct = 0
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool); mask[i] = False
        cents = []
        for ri in range(len(rels)):
            sel = mask & (y == ri)
            cents.append(X[sel].mean(0) if sel.any() else torch.zeros(X.shape[1]))
        cents = F.normalize(torch.stack(cents), dim=-1)
        pred = (X[i] @ cents.T).argmax().item()
        correct += int(pred == y[i].item())
    return correct / max(1, n), len(rels)

# ----------------------------------------------------------------------------------------
# STAGE 2: meta-learn a COMPRESSOR: K example activations -> soft prefix. Leave-one-relation-out:
# for each held-out relation, train the compressor on the OTHER relations, then a few examples of
# the held-out relation produce a prefix the frozen model applies to held-out words.
class Compressor(nn.Module):
    """K example features (frozen LLM activations of x and R(x)) -> a soft prefix [m, H].
    Permutation-invariant over the K examples (mean), like sidecar_semantic's write.
    CRITICAL (diagnosed from a first run where generated prefixes collapsed to norm ~0.1 and the
    model's free-gen output went to a degenerate ' ' token): we EXPLICITLY scale each generated
    prefix vector to `target_norm` - matched to where independently-optimized stage-1 prefixes live
    (~real-token-embedding norm) - so the compressor starts in a regime that can actually drive the
    frozen model, instead of a near-zero basin the optimizer never escapes. A learnable per-output
    gain around that target allows fine adjustment. This is the fair, good-faith strong version."""
    def __init__(self, H, m, hidden=512, proj=256, target_norm=0.45):
        super().__init__()
        self.m, self.H, self.target_norm = m, H, target_norm
        self.proj = nn.Linear(H, proj)
        self.enc = nn.Sequential(nn.Linear(2 * proj, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, m * H))
        self.gain = nn.Parameter(torch.ones(m))          # per-prefix-vector learnable gain about target
    def forward(self, xf, yf):                            # xf,yf: [B, K, H] frozen example feats
        e = self.enc(torch.cat([self.proj(xf), self.proj(yf)], -1)).mean(1)   # [B, hidden]
        raw = self.dec(e).view(-1, self.m, self.H)                            # [B, m, H]
        unit = F.normalize(raw, dim=-1)                                       # direction only
        return unit * (self.target_norm * self.gain)[None, :, None]           # scaled to useful norm

@torch.no_grad()
def harvest_word_feature(tok, model, word, layer):
    """Frozen residual feature for a word at its position in CARRIER (the sink-fix), for stage 2."""
    tid = single_token_id(tok, word)
    ids = tok.encode(CARRIER.format(w=word), add_special_tokens=False)
    pos = max(i for i, t in enumerate(ids) if t == tid)
    out = model(torch.tensor(ids, device=DEV)[None, :], output_hidden_states=True)
    return out.hidden_states[layer][0, pos, :].float()

def run_stage2(tok, model, words, train_pairs, test_pairs, q_emb_cache, answer_tok, menu_ids,
               out_set_idx, m, layer, steps, K, lr, seed, feat_cache):
    """Leave-one-relation-out meta-consolidation. Returns per-held-out-relation menu/free apply acc
    using a compressor-generated prefix (trained on the OTHER relations only)."""
    H = model.config.hidden_size
    results = {}
    rels = REL_NAMES
    for held in rels:
        torch.manual_seed(seed)
        comp = Compressor(H, m).to(DEV)
        opt = torch.optim.Adam(comp.parameters(), lr)
        train_rels = [r for r in rels if r != held]
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        comp.train()
        for step in range(steps):
            # sample a train relation + K teaching pairs (train) + a batch of train query words
            r = train_rels[int(torch.randint(0, len(train_rels), (1,), generator=g, device=DEV))]
            tp = train_pairs[r]
            kk = min(K, tp.shape[0])
            ti = torch.randperm(tp.shape[0], generator=g, device=DEV)[:kk]
            xi = tp[ti, 0]; yi = tp[ti, 1]
            xf = torch.stack([feat_cache[int(j)] for j in xi])[None]      # [1,K,H]
            yf = torch.stack([feat_cache[int(j)] for j in yi])[None]      # [1,K,H]
            pre = comp(xf, yf)                                            # [1,m,H]
            # query: train pairs of r HELD OUT of the K teaching set (forces the compressor to make
            # a prefix that GENERALIZES the relation, not one that echoes the taught answers). Falls
            # back to all train pairs if the relation has too few to spare.
            taught_x = set(int(j) for j in xi)
            qsel = [(a, b) for (a, b) in tp.tolist() if a not in taught_x]
            if len(qsel) < 2:
                qsel = tp.tolist()
            qsel = qsel[:8]
            xs = [words[a] for (a, b) in qsel]; ys = [answer_tok[words[b]] for (a, b) in qsel]
            padded, mask = batch_pack([q_emb_cache[x] for x in xs])
            pm = SoftPrefix(m, H).to(DEV)            # thin wrapper to reuse forward path
            pm.prefix = nn.Parameter(pre[0])         # generated (carries grad to comp)
            logits = forward_with_prefix(model, pm, padded, mask)
            loss = F.cross_entropy(logits, torch.tensor(ys, device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        comp.eval()
        # EVAL on the HELD-OUT relation: K teaching from its TRAIN pairs -> prefix -> held-out TEST words
        with torch.no_grad():
            tp = train_pairs[held]; kk = min(K, tp.shape[0])
            gg = torch.Generator(device=DEV).manual_seed(seed + 5)
            ti = torch.randperm(tp.shape[0], generator=gg, device=DEV)[:kk]
            xf = torch.stack([feat_cache[int(j)] for j in tp[ti, 0]])[None]
            yf = torch.stack([feat_cache[int(j)] for j in tp[ti, 1]])[None]
            pre = comp(xf, yf)
            pm = SoftPrefix(m, H).to(DEV); pm.prefix = nn.Parameter(pre[0])
            gen_norm = float(pre[0].norm(dim=-1).mean().item())   # diagnose prefix strength
            te = test_pairs[held].tolist()
            if te:
                xs = [words[a] for (a, b) in te]
                ytok = torch.tensor([answer_tok[words[b]] for (a, b) in te], device=DEV)
                ymenu = torch.tensor([out_set_idx[words[b]] for (a, b) in te], device=DEV)
                padded, mask = batch_pack([q_emb_cache[x] for x in xs])
                lg = forward_with_prefix(model, pm, padded, mask)
                free = (lg.argmax(-1) == ytok).float().mean().item()
                menu = (lg[:, menu_ids].argmax(-1) == ymenu).float().mean().item()
            else:
                free = menu = float("nan")
        results[held] = dict(menu=menu, free=free, gen_prefix_norm=gen_norm)
    return results

# ----------------------------------------------------------------------------------------
def svg_stage1(path, rels, soft_menu, soft_free, icl, readmlp, null_menu, chance, title):
    """Grouped bars per relation: soft-prefix (menu + free) vs ICL ceiling vs read-MLP vs null,
    + an AGGREGATE group. The whole verdict in one picture."""
    groups = list(rels) + ["AGG"]
    W, Hh, ml, mr, mt, mb = 860, 380, 52, 188, 44, 86
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(groups); bw = (x1 - x0) / n
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    Ych = Yc(chance)
    p.append(f'<line x1="{x0}" y1="{Ych:.1f}" x2="{x1}" y2="{Ych:.1f}" stroke="{MUT}" stroke-dasharray="4 3"/>')
    def bar(cx, frac, v, col):
        if v != v: return ""   # nan
        return f'<rect x="{cx:.1f}" y="{Yc(v):.1f}" width="{bw*0.17:.1f}" height="{(y0-Yc(v)):.1f}" fill="{col}"/>'
    for i, gname in enumerate(groups):
        cx = x0 + (i + 0.5) * bw
        sm = soft_menu.get(gname, float("nan")); sf = soft_free.get(gname, float("nan"))
        ic = icl.get(gname, float("nan")); rm = readmlp.get(gname, float("nan")); nu = null_menu.get(gname, float("nan"))
        p.append(bar(cx - bw*0.42, 1, sm, TEAL))
        p.append(bar(cx - bw*0.23, 1, sf, PINK))
        p.append(bar(cx - bw*0.04, 1, ic, GOLD))
        p.append(bar(cx + bw*0.15, 1, rm, LILAC))
        p.append(bar(cx + bw*0.34, 1, nu, SLATE))
        lab = gname if gname != "AGG" else "AGGREGATE"
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" transform="rotate(20 {cx:.1f} {y0+14})">{lab}</text>')
    ly = mt + 12
    for col, lab in [(TEAL, "soft-prefix (menu)"), (PINK, "soft-prefix (free-gen)"), (GOLD, "ICL ceiling"),
                     (LILAC, "read-MLP (p18)"), (SLATE, "null (random prefix)"), (MUT, f"chance {chance:.3f}")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_stage2(path, rels, s2_menu, icl, readmlp, chance, title):
    W, Hh, ml, mr, mt, mb = 760, 360, 52, 180, 44, 86
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(rels); bw = (x1 - x0) / n
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    for i, r in enumerate(rels):
        cx = x0 + (i + 0.5) * bw
        for off, key, col in [(-0.30, s2_menu, TEAL), (-0.04, icl, GOLD), (0.22, readmlp, LILAC)]:
            v = key.get(r, float("nan"))
            if v == v:
                p.append(f'<rect x="{cx+off*bw:.1f}" y="{Yc(v):.1f}" width="{bw*0.24:.1f}" height="{(y0-Yc(v)):.1f}" fill="{col}"/>')
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" transform="rotate(20 {cx:.1f} {y0+14})">{r}</text>')
    ly = mt + 12
    for col, lab in [(TEAL, "meta-consolidated prefix"), (GOLD, "ICL ceiling"), (LILAC, "read-MLP")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--m", type=int, default=8)          # soft-prefix length
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--icl_K", type=int, default=5)
    ap.add_argument("--icl_episodes", type=int, default=200)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=12)     # stage-2 feature layer (= read-MLP best L12)
    ap.add_argument("--stage2_steps", type=int, default=1500)
    ap.add_argument("--stage2_K", type=int, default=5)
    ap.add_argument("--stage2_lr", type=float, default=1e-3)
    ap.add_argument("--force_stage2", action="store_true")
    ap.add_argument("--no_stage2", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--ms", default="")                  # optional prefix-length sweep "4,8" (reported, not cherry-picked)
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(","))
    t_start = time.time()

    words, widx, out_words, out_ids = build_vocab()
    train_pairs, test_pairs = split_relations(words, widx, test_frac=args.test_frac, seed=args.split_seed)
    chance = 1.0 / len(out_ids)
    print(f"device={DEV}  model={args.model}  m={args.m}  steps={args.steps}  seeds={seeds}")
    print(f"|words|={len(words)}  |menu V|={len(out_ids)}  chance=1/|V|={chance:.4f}")
    print("relation TRAIN/TEST pair counts (identical split to the read-MLP rung):")
    for r in REL_NAMES:
        print(f"  {r:12s}  train={train_pairs[r].shape[0]:2d}  test={test_pairs[r].shape[0]:2d}")

    print("\nloading FROZEN LLM, caching query embeddings...")
    tok, model = load_llm(args.model, dtype=getattr(torch, args.dtype))
    answer_tok = {w: single_token_id(tok, w) for w in words}            # word -> answer token id
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)   # candidate menu token ids
    out_set_idx = {w: j for j, w in enumerate(out_words)}
    menu_rel = build_menu_rel(out_ids, widx)            # [|menu|, R] is menu word j an answer of relation r?
    q_emb_cache = cache_query_embeds(tok, model, words)
    H = model.config.hidden_size
    # per-relation query-word pools (held-out test x's) for the cross-relation control
    test_xwords = {r: [words[xi] for (xi, yi) in test_pairs[r].tolist()] for r in REL_NAMES}

    readmlp = load_readmlp_baseline()
    if readmlp is None:
        print("  WARNING: sidecar_semantic_0p5b.json not found; read-MLP baseline will be NaN.")

    report = dict(model=args.model, device=DEV, m=args.m, steps=args.steps, lr=args.lr,
                  seeds=list(seeds), icl_K=args.icl_K, test_frac=args.test_frac,
                  split_seed=args.split_seed, menu_size=len(out_ids), chance=chance,
                  query_template=QUERY_TMPL,
                  env="cloze/.venv (torch cu128, RTX 5080)", frozen_backbone=True,
                  readmlp_baseline=readmlp, stage1={}, stage2=None, legibility={})

    # ----- prefix-length sweep (optional; if given, all m reported - no cherry-pick) -----
    ms = [int(x) for x in args.ms.split(",")] if args.ms else [args.m]

    # ======================= STAGE 1 =======================
    print("\n" + "=" * 78)
    print("STAGE 1 - learn a SOFT PREFIX per relation; frozen model applies it to HELD-OUT words")
    print("=" * 78)

    # ICL ceiling + null are independent of m / seed-of-prefix; compute once.
    print("\nnative-ICL ceiling (frozen model, K text pairs in prompt; retrieval-scored over V):")
    icl_per = {}
    for r in REL_NAMES:
        icl_per[r] = icl_ceiling_rel(tok, model, r, train_pairs, test_pairs, words, menu_ids,
                                     out_set_idx, K=args.icl_K, n_episodes=args.icl_episodes)
        print(f"  {r:12s}  ICL = {icl_per[r]:.3f}")
    icl_agg = float(sum(icl_per[r] for r in REL_NAMES) / len(REL_NAMES))

    stage1_by_m = {}
    best_m, best_agg_menu = ms[0], -1.0
    prefix_store = {}   # for legibility: {m: {rel: [n_seeds, m*H]}}
    for m_len in ms:
        print(f"\n--- prefix length m={m_len} ---")
        # accumulate per-relation over seeds
        soft_menu_seed = defaultdict(list); soft_free_seed = defaultdict(list)
        null_menu_seed = defaultdict(list); null_free_seed = defaultdict(list)
        nonejust_menu = defaultdict(list)
        train_acc_seed = defaultdict(list)
        xrel_seed = defaultdict(list)        # cross-relation: R's prefix on OTHER relations' words -> R-type?
        prefix_vecs = defaultdict(list)
        for sd in seeds:
            for r in REL_NAMES:
                pre = train_soft_prefix_for_relation(
                    tok, model, r, train_pairs, words, q_emb_cache, answer_tok,
                    m=m_len, steps=args.steps, lr=args.lr, seed=sd)
                mn, fr, ntest = eval_soft_prefix(model, pre, r, test_pairs, words, q_emb_cache,
                                                 answer_tok, menu_ids, out_set_idx)
                soft_menu_seed[r].append(mn); soft_free_seed[r].append(fr)
                prefix_vecs[r].append(pre.prefix.detach().flatten().cpu())
                # cross-relation control: apply R's prefix to other relations' held-out words
                foreign = [w for rr in REL_NAMES if rr != r for w in test_xwords[rr]]
                xrel_seed[r].append(cross_relation_apply(model, pre, r, foreign, words, q_emb_cache,
                                                         menu_ids, out_words, menu_rel))
                # train-fit sanity (does the prefix even fit TRAIN?) - reported, catches under-fit
                with torch.no_grad():
                    trp = train_pairs[r].tolist()
                    xs = [words[a] for (a, b) in trp]
                    ytok = torch.tensor([answer_tok[words[b]] for (a, b) in trp], device=DEV)
                    pad, msk = batch_pack([q_emb_cache[x] for x in xs])
                    tacc = (forward_with_prefix(model, pre, pad, msk).argmax(-1) == ytok).float().mean().item()
                train_acc_seed[r].append(tacc)
                # nulls (random prefix + no prefix), same held-out words
                nm, nf = eval_null_prefix(model, r, test_pairs, words, q_emb_cache, answer_tok,
                                          menu_ids, out_set_idx, m_len, sd, mode="random")
                null_menu_seed[r].append(nm); null_free_seed[r].append(nf)
                nem, _ = eval_null_prefix(model, r, test_pairs, words, q_emb_cache, answer_tok,
                                          menu_ids, out_set_idx, m_len, sd, mode="none")
                nonejust_menu[r].append(nem)
            print(f"  seed {sd}: done all {len(REL_NAMES)} relations "
                  f"(elapsed {time.time()-t_start:.0f}s)")
        # aggregate
        def mean_over_seeds(d, r): return float(sum(d[r]) / len(d[r]))
        soft_menu = {r: mean_over_seeds(soft_menu_seed, r) for r in REL_NAMES}
        soft_free = {r: mean_over_seeds(soft_free_seed, r) for r in REL_NAMES}
        null_menu = {r: mean_over_seeds(null_menu_seed, r) for r in REL_NAMES}
        null_free = {r: mean_over_seeds(null_free_seed, r) for r in REL_NAMES}
        none_menu = {r: mean_over_seeds(nonejust_menu, r) for r in REL_NAMES}
        train_fit = {r: mean_over_seeds(train_acc_seed, r) for r in REL_NAMES}
        xrel = {r: mean_over_seeds(xrel_seed, r) for r in REL_NAMES}
        agg = lambda d: float(sum(d[r] for r in REL_NAMES) / len(REL_NAMES))
        sm_std = {r: float(torch.tensor(soft_menu_seed[r]).std().item()) if len(seeds) > 1 else 0.0
                  for r in REL_NAMES}
        rec = dict(soft_menu=soft_menu, soft_free=soft_free, null_menu=null_menu, null_free=null_free,
                   none_menu=none_menu, train_fit=train_fit, soft_menu_std=sm_std, cross_rel=xrel,
                   agg_soft_menu=agg(soft_menu), agg_soft_free=agg(soft_free),
                   agg_null_menu=agg(null_menu), agg_none_menu=agg(none_menu),
                   agg_train_fit=agg(train_fit), agg_cross_rel=agg(xrel))
        stage1_by_m[str(m_len)] = rec
        prefix_store[m_len] = {r: prefix_vecs[r] for r in REL_NAMES}
        # print per-relation table
        print(f"\n  m={m_len} HELD-OUT apply accuracy (model's OWN output), per relation:")
        print("  relation      | soft(menu) soft(free) |  ICL   read-MLP  null  none | train-fit  x-rel")
        rm_per = readmlp["per_relation"] if readmlp else {}
        for r in REL_NAMES:
            print(f"  {r:12s}  |   {soft_menu[r]:.3f}     {soft_free[r]:.3f}   | "
                  f"{icl_per[r]:.3f}   {rm_per.get(r, float('nan')):.3f}   {null_menu[r]:.3f} {none_menu[r]:.3f} | "
                  f"  {train_fit[r]:.3f}   {xrel[r]:.3f}")
        rm_agg = readmlp["aggregate"] if readmlp else float("nan")
        print(f"  {'AGGREGATE':12s}  |   {rec['agg_soft_menu']:.3f}     {rec['agg_soft_free']:.3f}   | "
              f"{icl_agg:.3f}   {rm_agg:.3f}   {rec['agg_null_menu']:.3f} {rec['agg_none_menu']:.3f} | "
              f"  {rec['agg_train_fit']:.3f}   {rec['agg_cross_rel']:.3f}")
        print("  (x-rel = R's prefix applied to OTHER relations' words -> fraction landing on an R-TYPE "
              "answer; high => the prefix transports the RELATION, not a memorized answer set.)")
        if rec["agg_soft_menu"] > best_agg_menu:
            best_agg_menu, best_m = rec["agg_soft_menu"], m_len

    report["stage1"] = dict(by_m=stage1_by_m, icl_per_relation=icl_per, icl_aggregate=icl_agg,
                            best_m=best_m, prefix_length_sweep=ms)

    # ----- LEGIBILITY of the learned prefixes (at best_m) -----
    leg_acc, n_rel_leg = legibility_probe(prefix_store[best_m])
    leg_chance = 1.0 / n_rel_leg
    leg_legible = (leg_acc == leg_acc) and leg_acc > 2 * leg_chance
    leg_note = ("legible: same-relation prefixes cluster" if leg_legible else
                "NOT legible at chance level - independently-optimized prefixes for the SAME relation "
                "do not align in parameter space (different inits -> different equally-valid solutions "
                "of the same behavior). Contrast sidecar_semantic's shared-encoder state (probe 1.000): "
                "legibility there came from a SHARED encoder, absent here. Per-relation soft prefixes "
                "are functionally relation-specific but parametrically not relation-legible.")
    report["legibility"] = dict(prefix_relation_probe_acc=leg_acc, chance=leg_chance, legible=leg_legible,
                                method="LOO nearest-centroid over centered (rel,seed) prefixes",
                                n_relations=n_rel_leg, n_prefixes=n_rel_leg * len(seeds), note=leg_note)
    print(f"\nLEGIBILITY: learned-prefix relation identity (LOO nearest-centroid, centered) = {leg_acc:.3f}  "
          f"(chance 1/{n_rel_leg} = {leg_chance:.3f}; n={n_rel_leg*len(seeds)} prefixes)")
    print(f"  -> {leg_note}")

    # ----- STAGE 1 VERDICT -----
    best = stage1_by_m[str(best_m)]
    sm = best["agg_soft_menu"]; nu = best["agg_null_menu"]; tf = best["agg_train_fit"]
    rm_agg = readmlp["aggregate"] if readmlp else float("nan")
    # the five 1-to-1 relations the read-MLP scored 0.000 on - the real test
    one_to_one = ["antonym", "plural", "past", "comparative", "capital"]
    soft_121 = float(sum(best["soft_menu"][r] for r in one_to_one) / len(one_to_one))
    icl_121 = float(sum(icl_per[r] for r in one_to_one) / len(one_to_one))
    rm_121 = float(sum((readmlp["per_relation"].get(r, 0.0) if readmlp else 0.0) for r in one_to_one) / len(one_to_one))
    # gate: soft-prefix must (a) crush the read-MLP on the 1-to-1 relations specifically, (b) be
    # well above the null, (c) reach a meaningful fraction of the ICL ceiling. Conservative.
    beats_readmlp_121 = soft_121 > rm_121 + 0.15
    beats_null = sm > nu + 0.15
    frac_icl = (sm / icl_agg) if icl_agg > 1e-6 else 0.0
    stage1_pass = bool(beats_readmlp_121 and beats_null and frac_icl > 0.5)
    if stage1_pass:
        verdict1 = (f"STAGE 1 PASS: a learned soft-prefix injection makes the FROZEN model APPLY relations "
                    f"to HELD-OUT words. 1-to-1 relations (read-MLP=0.000) reach soft={soft_121:.3f} "
                    f"(ICL={icl_121:.3f}); aggregate soft={sm:.3f} >> null={nu:.3f}, "
                    f"{100*frac_icl:.0f}% of the ICL ceiling. The inject-and-model-applies architecture is viable.")
    else:
        why = []
        if not beats_readmlp_121: why.append(f"does NOT beat read-MLP on 1-to-1 (soft={soft_121:.3f} vs read-MLP={rm_121:.3f})")
        if not beats_null: why.append(f"not clear of null (soft={sm:.3f} vs null={nu:.3f})")
        if frac_icl <= 0.5: why.append(f"only {100*frac_icl:.0f}% of ICL ceiling")
        diag = ("train-fit is LOW (prefix cannot even memorize TRAIN -> capacity/optimization limit) "
                if tf < 0.6 else
                "train-fit is HIGH but held-out is LOW (prefix MEMORIZES train pairs, does not GENERALIZE "
                "the relation - the soft prefix encodes the seen answers, not an applicable rule)")
        verdict1 = ("STAGE 1 NEGATIVE: a learned per-relation soft-prefix does NOT let the frozen model apply "
                    "the relation to held-out words [" + "; ".join(why) + "]. Diagnosis: " + diag +
                    f". (agg train-fit={tf:.3f}, held-out menu={sm:.3f}, ICL={icl_agg:.3f}.) "
                    "This kills the simplest version of the frontier; reported plainly.")
    report["stage1"]["verdict"] = dict(text=verdict1, pass_=stage1_pass, best_m=best_m,
                                       agg_soft_menu=sm, agg_soft_free=best["agg_soft_free"],
                                       agg_null_menu=nu, agg_train_fit=tf, icl_aggregate=icl_agg,
                                       frac_of_icl=frac_icl,
                                       one_to_one_soft=soft_121, one_to_one_icl=icl_121,
                                       one_to_one_readmlp=rm_121,
                                       beats_readmlp_one_to_one=beats_readmlp_121, beats_null=beats_null)
    print("\n" + "-" * 78)
    print("STAGE 1 VERDICT:", verdict1)
    print("-" * 78)

    # SVG for stage 1 (best m)
    soft_menu_plot = dict(best["soft_menu"]); soft_menu_plot["AGG"] = sm
    soft_free_plot = dict(best["soft_free"]); soft_free_plot["AGG"] = best["agg_soft_free"]
    icl_plot = dict(icl_per); icl_plot["AGG"] = icl_agg
    rm_plot = dict((readmlp["per_relation"] if readmlp else {})); rm_plot["AGG"] = rm_agg
    null_plot = dict(best["null_menu"]); null_plot["AGG"] = nu
    svg_stage1(os.path.join(RUNS, f"frontier_apply_stage1{args.tag}.svg"), REL_NAMES,
               soft_menu_plot, soft_free_plot, icl_plot, rm_plot, null_plot, chance,
               f"Stage 1: can the frozen model apply a learned injection? (Qwen2.5-0.5B-Instruct, m={best_m})")

    # ======================= STAGE 2 (gated) =======================
    do_stage2 = (stage1_pass or args.force_stage2) and not args.no_stage2
    if do_stage2:
        print("\n" + "=" * 78)
        print("STAGE 2 - meta-learn a COMPRESSOR (K example activations -> soft prefix), LORO")
        print("=" * 78)
        print("harvesting frozen word features for stage-2 compressor input...")
        feat_cache = {}
        for w in words:
            feat_cache[widx[w]] = harvest_word_feature(tok, model, w, args.layer)
        s2 = run_stage2(tok, model, words, train_pairs, test_pairs, q_emb_cache, answer_tok,
                        menu_ids, out_set_idx, m=best_m, layer=args.layer, steps=args.stage2_steps,
                        K=args.stage2_K, lr=args.stage2_lr, seed=seeds[0], feat_cache=feat_cache)
        s2_menu = {r: s2[r]["menu"] for r in REL_NAMES}
        s2_free = {r: s2[r]["free"] for r in REL_NAMES}
        agg2_menu = float(sum(s2_menu[r] for r in REL_NAMES) / len(REL_NAMES))
        agg2_free = float(sum(s2_free[r] for r in REL_NAMES) / len(REL_NAMES))
        agg2_norm = float(sum(s2[r]["gen_prefix_norm"] for r in REL_NAMES) / len(REL_NAMES))
        print(f"  (mean generated-prefix norm = {agg2_norm:.3f}; real-token-emb norm ~0.45; "
              f"stage-1 prefixes reach a comparable norm. Collapsed norm => weak/ineffective injection.)")
        print("\n  STAGE 2 leave-one-relation-out held-out apply (compressor-generated prefix):")
        print("  held-out rel  | meta(menu) meta(free) |  ICL   read-MLP")
        for r in REL_NAMES:
            print(f"  {r:12s}  |   {s2_menu[r]:.3f}     {s2_free[r]:.3f}   | {icl_per[r]:.3f}   "
                  f"{(readmlp['per_relation'].get(r, float('nan')) if readmlp else float('nan')):.3f}")
        print(f"  {'AGGREGATE':12s}  |   {agg2_menu:.3f}     {agg2_free:.3f}   | {icl_agg:.3f}")
        # HONEST gate: stage 2 only counts as the model genuinely APPLYING the relation if FREE-GEN
        # (the model's true argmax) works - a high menu-restricted score with ~0 free-gen is a menu
        # artifact (the right token only wins after discarding the 100s of tokens the model prefers),
        # NOT the model applying its clay. Stage 1 had free ~= menu (genuine); we hold stage 2 to the
        # same bar. menu reported alongside for completeness, but the verdict keys on FREE.
        s2_genuine = agg2_free > (rm_agg + 0.10) and agg2_free > 0.3
        s2_menu_only = agg2_menu > (rm_agg + 0.10) and agg2_menu > 0.3
        if s2_genuine:
            verdict2 = (f"STAGE 2 PASS: a meta-learned compressor produces a prefix the FROZEN model genuinely "
                        f"applies to held-out words of a NEW relation: free-gen={agg2_free:.3f} (menu={agg2_menu:.3f}) "
                        f"vs read-MLP {rm_agg:.3f}, ICL {icl_agg:.3f}. The full consolidate-a-rule-the-model-applies "
                        f"loop works.")
        elif s2_menu_only:
            verdict2 = (f"STAGE 2 NEGATIVE (menu-only mirage): the compressor's prefix scores menu={agg2_menu:.3f} "
                        f"but FREE-GEN={agg2_free:.3f} - the model's TRUE output is NOT the answer (the menu-restricted "
                        f"number only survives by discarding the tokens the model actually prefers). The generated "
                        f"prefix is too weak (collapsed norm) to drive the frozen model the way an independently-"
                        f"optimized stage-1 prefix does. Meta-learning the injection across relations FAILS at the "
                        f"honest (free-gen) bar, even though per-relation prefixes (stage 1) succeed. Reported plainly.")
        else:
            verdict2 = (f"STAGE 2 NEGATIVE: meta-consolidated prefix does not apply the relation (free-gen={agg2_free:.3f}, "
                        f"menu={agg2_menu:.3f}) vs read-MLP {rm_agg:.3f}. The compressor does not learn a prefix the "
                        f"frozen model can use on a new relation. Reported plainly.")
        report["stage2"] = dict(per_relation_menu=s2_menu, per_relation_free=s2_free,
                                aggregate_menu=agg2_menu, aggregate_free=agg2_free,
                                gen_prefix_norm=agg2_norm,
                                pass_=bool(s2_genuine), menu_only_mirage=bool(s2_menu_only and not s2_genuine),
                                verdict=verdict2, K=args.stage2_K, layer=args.layer, steps=args.stage2_steps)
        print("\nSTAGE 2 VERDICT:", verdict2)
        svg_stage2(os.path.join(RUNS, f"frontier_apply_stage2{args.tag}.svg"), REL_NAMES, s2_menu,
                   icl_per, (readmlp["per_relation"] if readmlp else {}), chance,
                   f"Stage 2: meta-consolidated prefix, leave-one-relation-out (m={best_m}, K={args.stage2_K})")
    else:
        why = "stage 1 did not pass" if not (stage1_pass or args.force_stage2) else "disabled by flag"
        print(f"\nSTAGE 2 skipped ({why}).")
        report["stage2"] = dict(skipped=True, reason=why)

    # ----- final report -----
    report["wall_time_s"] = round(time.time() - t_start, 1)
    json.dump(report, open(os.path.join(RUNS, f"frontier_apply{args.tag}.json"), "w"), indent=2)
    print(f"\nwrote runs/frontier_apply{args.tag}.json (+ stage1 SVG"
          + (", stage2 SVG" if do_stage2 else "") + f")  [{report['wall_time_s']}s]")

    # ----- FINAL HONEST VERDICT -----
    print("\n" + "#" * 78)
    print("# FRONTIER VERDICT - can the frozen model apply its OWN consolidated clay?")
    print("#" * 78)
    print("# " + verdict1.replace("\n", "\n# "))
    if report["stage2"] and not report["stage2"].get("skipped"):
        print("# " + report["stage2"]["verdict"].replace("\n", "\n# "))
    print(f"# legibility (per-relation prefix -> relation): {leg_acc:.3f} (chance {leg_chance:.3f}) "
          f"-> {'legible' if leg_legible else 'NOT legible (independent inits do not align; see note)'}")
    print("#" * 78)

if __name__ == "__main__":
    main()
