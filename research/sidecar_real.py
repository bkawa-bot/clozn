"""
sidecar_real.py — Experiment 8b: the consolidation SIDECAR on a REAL frozen LLM.

The toy (research/sidecar.py) showed a meta-learned write/read module can CONSOLIDATE
K teaching pairs of a hidden cipher  y = (x + b) mod N  into the underlying RULE b:
it generalizes to UNTAUGHT inputs, beats a lookup memory, and is legible (a linear probe
reads b out of the state; b lays on a circle). But its tokens were a toy nn.Embedding.

This file asks the smallest-thing-that-could-kill-it question: does the SAME consolidation
work when feat(token) is a REAL FROZEN LLM's mid-layer activation instead of a toy embedding?

  feat(t)  = frozen Qwen2.5 residual-stream activation for number-word token t (cached once)
  WRITE :  s = mean_i  write_mlp([feat(x_i), feat(y_i)])      (permutation-invariant accumulate)
  READ  :  logits = read_mlp([feat(x_query), s])  over N classes
  write/read are META-LEARNED across random-b episodes; the LLM stays FROZEN.

Tractability: the backbone is frozen, so each token's representation is fixed. We HARVEST
feat(t) for all N vocab tokens ONCE (per tapped layer), cache it, then meta-train the small
sidecar on the cached features — fast, the expensive LLM forwards happen once.

Honesty (controls beside every number):
  - chance = 1/N
  - LOOKUP baseline (store the K pairs; untaught -> chance) — the contrast that proves consolidation
  - NATIVE ICL ceiling: put the K pairs in Qwen's PROMPT as text, let the FROZEN model itself
    answer the untaught query (no sidecar). Does the sidecar approach what the model can do in-context?
We aggregate over thousands of held-out rule instances and report the FULL layer sweep (no
cherry-picking K or layer).

Tasks:
  PRIMARY     : mod-N cipher over number-word tokens  " zero".." eleven"  (single-token; N=12, mirrors the toy)
  ROBUSTNESS  : permutation cipher over a small set of common single-token words

Outputs (research/runs/): sidecar_real.json, sidecar_real_genK.svg, sidecar_real_circle.svg
"""
import os, sys, json, time, argparse, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.set_float32_matmul_precision("high")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

# Maiko palette
BG, TEAL, PINK, TXT, MUT, GRID = "#1A1F4A", "#6FD6C9", "#FF8FB3", "#F4F0E8", "#8784b3", "#2c2f5e"
GOLD = "#E8C977"  # third series (ICL ceiling)

# ----------------------------------------------------------------------------------------
# Task vocab. PRIMARY = number words (single-token w/ leading space in Qwen2.5 BPE).
NUMBER_WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven"]
# A small set of common single-token words for the permutation task.
PERM_WORDS = ["cat","dog","red","blue","sun","moon","sea","fire","tree","star","king","gold"]
CARRIER = "The number {w}"           # carrier context (escapes Qwen's position-0 sink artifact)
CARRIER_PERM = "The word {w}"

# ----------------------------------------------------------------------------------------
def load_llm(model_name, dtype=torch.float32):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(DEV).eval()
    return tok, model

@torch.no_grad()
def harvest_features(tok, model, words, carrier, layers):
    """Return {L: tensor[N, H]} of frozen residual-stream features for each word token,
    tapped at the word's position inside `carrier`. Done ONCE; the LLM never trains."""
    wid = [tok.encode(" " + w, add_special_tokens=False) for w in words]
    for w, ids in zip(words, wid):
        assert len(ids) == 1, f"word {w!r} is not single-token: {ids}"
    wid = [i[0] for i in wid]
    feats = {L: [] for L in layers}
    for w, tid in zip(words, wid):
        ids = tok.encode(carrier.format(w=w), add_special_tokens=False)
        pos = max(i for i, t in enumerate(ids) if t == tid)   # the word-token position
        out = model(torch.tensor(ids, device=DEV)[None, :], output_hidden_states=True)
        for L in layers:
            feats[L].append(out.hidden_states[L][0, pos, :].float())
    return {L: torch.stack(v) for L, v in feats.items()}, wid

# ----------------------------------------------------------------------------------------
class Sidecar(nn.Module):
    """Mirror of research/sidecar.py's Sidecar, but feat() is a FROZEN cached LLM feature
    (registered as a buffer, never trained) instead of a learned nn.Embedding."""
    def __init__(self, feats, n, proj=64, hs=128, feat_noise=0.0):
        super().__init__()
        self.register_buffer("feat", feats)          # [N, H] frozen LLM features
        H = feats.shape[1]
        self.n = n
        self.feat_noise = feat_noise                 # additive train-time feature noise (robustness probe)
        self.proj = nn.Linear(H, proj)               # learned projection of the frozen feature
        self.write = nn.Sequential(nn.Linear(2 * proj, hs), nn.ReLU(), nn.Linear(hs, hs))
        self.read  = nn.Sequential(nn.Linear(proj + hs, hs), nn.ReLU(), nn.Linear(hs, n))
    def fe(self, idx):
        f = self.feat[idx]
        if self.training and self.feat_noise > 0:
            f = f + self.feat_noise * torch.randn_like(f)
        return self.proj(f)
    def state(self, xp, yp):                          # xp,yp: [B,K]
        pe = self.write(torch.cat([self.fe(xp), self.fe(yp)], -1))   # [B,K,hs]
        return pe.mean(1)                             # [B,hs]
    def answer(self, xq, s):                          # xq:[B,Q], s:[B,hs]
        Q = xq.shape[1]
        return self.read(torch.cat([self.fe(xq), s[:, None, :].expand(-1, Q, -1)], -1))

def episode_mod(B, K, N, g, noise=0.0):
    b  = torch.randint(0, N, (B,), generator=g, device=DEV)
    xp = torch.randint(0, N, (B, K), generator=g, device=DEV)
    yp = (xp + b[:, None]) % N
    if noise > 0:
        bad = torch.rand(B, K, generator=g, device=DEV) < noise
        yp = torch.where(bad, torch.randint(0, N, (B, K), generator=g, device=DEV), yp)
    return xp, yp, b

def apply_rule_mod(allx, b, N):
    return (allx + b[:, None]) % N

def train_sidecar(feats, N, episode_fn, apply_rule, steps=12000, B=256, lr=1e-3,
                  seed=0, kmax=4, proj=64, hs=128, feat_noise=0.0):
    torch.manual_seed(seed)
    m = Sidecar(feats, N, proj=proj, hs=hs, feat_noise=feat_noise).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    allx = torch.arange(N, device=DEV)[None, :].expand(B, N)
    m.train()
    for step in range(steps):
        K = int(torch.randint(1, kmax, (1,), generator=g, device=DEV))
        xp, yp, b = episode_fn(B, K, N, g)
        s = m.state(xp, yp)
        logits = m.answer(allx, s)
        yq = apply_rule(allx, b, N)
        loss = F.cross_entropy(logits.reshape(-1, N), yq.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m

@torch.no_grad()
def untaught_acc(m, K, N, episode_fn, apply_rule, g, noise=0.0, B=4000):
    """Sidecar accuracy on UNTAUGHT inputs only (inputs not among the K teaching pairs)."""
    xp, yp, b = episode_fn(B, K, N, g, noise=noise)
    allx = torch.arange(N, device=DEV)[None, :].expand(B, N)
    pred = m.answer(allx, m.state(xp, yp)).argmax(-1)
    yq = apply_rule(allx, b, N)
    taught = torch.zeros(B, N, dtype=torch.bool, device=DEV).scatter_(1, xp, True)
    mask = ~taught
    sc = ((pred == yq) & mask).sum().item() / max(1, mask.sum().item())
    # also taught-input accuracy (sanity; sidecar should also get these)
    tc = ((pred == yq) & taught).sum().item() / max(1, taught.sum().item())
    return sc, tc

# ----------------------------------------------------------------------------------------
# NATIVE ICL ceiling: frozen model answers the untaught query from K pairs in its PROMPT.
@torch.no_grad()
def icl_ceiling_mod(tok, model, words, N, K, g, n_episodes=300, seed=999):
    """For each episode: random b, K teaching pairs as text, ask the FROZEN model to map an
    UNTAUGHT x -> y. Score = does argmax over the N number-word class tokens equal (x+b)%N.
    This is what the model can already do in-context, with NO sidecar."""
    gg = torch.Generator().manual_seed(seed)
    # class token ids (leading-space word forms), used to read the model's distribution over the N classes.
    class_ids = torch.tensor([tok.encode(" " + w, add_special_tokens=False)[0] for w in words], device=DEV)
    correct = 0; total = 0
    for _ in range(n_episodes):
        b = int(torch.randint(0, N, (1,), generator=gg).item())
        # sample K distinct teaching x, and an untaught query x not among them
        perm = torch.randperm(N, generator=gg).tolist()
        teach = perm[:K]
        untaught = [x for x in perm if x not in teach]
        if not untaught:
            continue
        xq = untaught[torch.randint(0, len(untaught), (1,), generator=gg).item()]
        # build a clean few-shot prompt mapping word->word
        lines = [f"{words[x]} -> {words[(x + b) % N]}" for x in teach]
        prompt = "Apply the same rule.\n" + "\n".join(lines) + f"\n{words[xq]} ->"
        ids = tok.encode(prompt, add_special_tokens=False)
        # the model's next-token: prefer the continuation that starts with a leading space (" word")
        logits = model(torch.tensor(ids, device=DEV)[None, :]).logits[0, -1]   # [V]
        # restrict to the N class tokens (leading-space forms), argmax among them
        cls_logits = logits[class_ids]
        pred = int(cls_logits.argmax().item())
        if pred == (xq + b) % N:
            correct += 1
        total += 1
    return correct / max(1, total)

# ----------------------------------------------------------------------------------------
# LEGIBILITY: linear probe reads secret b out of the sidecar state s.
def probe_b(m, N, episode_fn, g, B=8000):
    xp, yp, b = episode_fn(B, 3, N, g)
    with torch.no_grad():
        s = m.state(xp, yp)
    ntr = B // 2
    probe = nn.Linear(s.shape[1], N).to(DEV)
    opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(600):
        loss = F.cross_entropy(probe(s[:ntr]), b[:ntr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(s[ntr:]).argmax(1) == b[ntr:]).float().mean().item()
    return acc

@torch.no_grad()
def mean_state_per_b(m, N, episode_fn, g, B=12000):
    xp, yp, b = episode_fn(B, 3, N, g)
    s = m.state(xp, yp)
    return torch.stack([s[b == v].mean(0) for v in range(N)])   # [N, hs]

# ----------------------------------------------------------------------------------------
# Permutation task: per-episode secret permutation p; y = p[x]. Generalization = untaught x.
def make_perm_episode():
    def episode_perm(B, K, N, g, noise=0.0):
        # secret permutation per row
        perms = torch.argsort(torch.rand(B, N, generator=g, device=DEV), dim=1)   # [B,N]
        xp = torch.randint(0, N, (B, K), generator=g, device=DEV)
        yp = torch.gather(perms, 1, xp)
        if noise > 0:
            bad = torch.rand(B, K, generator=g, device=DEV) < noise
            yp = torch.where(bad, torch.randint(0, N, (B, K), generator=g, device=DEV), yp)
        # stash perms on the tensors so apply_rule can use them — return via b slot as the full perm
        return xp, yp, perms
    def apply_rule_perm(allx, perms, N):
        return torch.gather(perms, 1, allx)
    return episode_perm, apply_rule_perm

# ----------------------------------------------------------------------------------------
def run_layer(tag, feats_L, N, words, tok, model, episode_fn, apply_rule, do_icl=True,
              steps=12000, Ks=(1,2,3,5), legible=True, seeds=(0,), feat_noise=0.0):
    """Train `len(seeds)` sidecars; report untaught-acc per K aggregated (mean, std) across seeds.
    `legible` gates the scalar-b probe (only defined for the mod cipher, not permutations)."""
    g = torch.Generator(device=DEV).manual_seed(123)
    per_seed = []          # list of {K: (untaught, taught)}
    ms = []
    for sd in seeds:
        m = train_sidecar(feats_L, N, episode_fn, apply_rule, steps=steps, seed=sd, feat_noise=feat_noise)
        ms.append(m)
        d = {K: untaught_acc(m, K, N, episode_fn, apply_rule, g) for K in Ks}
        per_seed.append(d)
    genK = {}
    for K in Ks:
        us = [d[K][0] for d in per_seed]; ts = [d[K][1] for d in per_seed]
        genK[K] = dict(sidecar_untaught=float(sum(us)/len(us)),
                       sidecar_untaught_std=float(torch.tensor(us).std().item()) if len(us) > 1 else 0.0,
                       sidecar_taught=float(sum(ts)/len(ts)),
                       lookup_untaught=1.0 / N, chance=1.0 / N, n_seeds=len(seeds))
    m = ms[0]   # representative model for legibility / circle / noise
    noise = {}
    for eps in (0.0, 0.2, 0.4):
        sc, _ = untaught_acc(m, 5, N, episode_fn, apply_rule, g, noise=eps)
        noise[f"{eps}"] = sc
    leg = probe_b(m, N, episode_fn, g) if legible else None
    icl = {}
    if do_icl:
        for K in Ks:
            icl[K] = icl_ceiling_mod(tok, model, words, N, K, g)
    return dict(genK=genK, noise=noise, legibility=leg, chance=1.0 / N, icl=icl), m

# ----------------------------------------------------------------------------------------
def svg_genK(path, genK, icl, title, N):
    W, Hh, ml, mr, mt, mb = 600, 340, 56, 196, 40, 50
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    KS = sorted(genK.keys())
    Xc = lambda i: x0 + (i / (len(KS) - 1) * (x1 - x0) if len(KS) > 1 else 0)
    Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    # chance line
    Ych = Yc(1.0 / N)
    p.append(f'<line x1="{x0}" y1="{Ych:.1f}" x2="{x1}" y2="{Ych:.1f}" stroke="{MUT}" stroke-dasharray="4 3"/>')
    p.append(f'<text x="{x1-2}" y="{Ych-4:.1f}" fill="{MUT}" font-size="9" text-anchor="end">chance 1/{N}</text>')
    for i, K in enumerate(KS):
        p.append(f'<text x="{Xc(i):.1f}" y="{y0+18}" fill="{MUT}" font-size="11" text-anchor="middle">{K}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-10}" fill="#B8B3D6" font-size="11" text-anchor="middle"># teaching pairs shown (K)</text>')
    series = [("sidecar_untaught", TEAL, "sidecar (untaught)"),
              ("lookup_untaught", PINK, "lookup (untaught)")]
    for key, col, lab in series:
        pts = " ".join(f"{Xc(i):.1f},{Yc(genK[K][key]):.1f}" for i, K in enumerate(KS))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for i, K in enumerate(KS):
            p.append(f'<circle cx="{Xc(i):.1f}" cy="{Yc(genK[K][key]):.1f}" r="3.5" fill="{col}"/>')
    if icl:
        pts = " ".join(f"{Xc(i):.1f},{Yc(icl[K]):.1f}" for i, K in enumerate(KS))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{GOLD}" stroke-width="2.0" stroke-dasharray="6 3"/>')
        for i, K in enumerate(KS):
            p.append(f'<circle cx="{Xc(i):.1f}" cy="{Yc(icl[K]):.1f}" r="3.0" fill="{GOLD}"/>')
    ly = mt + 12
    legend = [(TEAL, "sidecar (untaught)"), (PINK, "lookup (untaught)")]
    if icl: legend.append((GOLD, "native-ICL ceiling"))
    legend.append((MUT, f"chance = 1/{N}"))
    for col, lab in legend:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_circle(path, means, N, title):
    e = means - means.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(e, full_matrices=False)
    P = (e @ Vh[:2].T).cpu(); P = P / (P.abs().max() + 1e-9)
    Wd = 460; R = 165; cx = Wd / 2; cy = Wd / 2 + 6
    Xf = lambda i: cx + P[i, 0].item() * R; Yf = lambda i: cy - P[i, 1].item() * R
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Wd+30}" font-family="Inconsolata,monospace">',
         f'<rect width="{Wd}" height="{Wd+30}" fill="{BG}"/>',
         f'<text x="{Wd/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    ring = " ".join(f"{Xf(i):.1f},{Yf(i):.1f}" for i in range(N)) + f" {Xf(0):.1f},{Yf(0):.1f}"
    p.append(f'<polyline points="{ring}" fill="none" stroke="#3a3f6e" stroke-width="1.2"/>')
    for i in range(N):
        p.append(f'<circle cx="{Xf(i):.1f}" cy="{Yf(i):.1f}" r="6" fill="hsl({int(360*i/N)},70%,64%)"/>')
        p.append(f'<text x="{Xf(i):.1f}" y="{Yf(i)-9:.1f}" fill="{TXT}" font-size="9" text-anchor="middle">{i}</text>')
    p.append(f'<text x="{Wd/2}" y="{Wd+24}" fill="{MUT}" font-size="10" text-anchor="middle">mean sidecar state per secret shift b, projected to 2D (real LLM features)</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layers", default="4,8,12,16,20")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--icl_episodes", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2")     # aggregate sidecar over seeds for spread
    ap.add_argument("--tag", default="")            # suffix for output files (e.g. "_3b")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--skip_controls", action="store_true")
    ap.add_argument("--skip_perm", action="store_true")
    args = ap.parse_args()

    N = len(NUMBER_WORDS)
    layers = [int(x) for x in args.layers.split(",")]
    seeds = tuple(int(x) for x in args.seeds.split(","))
    print(f"device={DEV}  model={args.model}  layers={layers}  N={N}  seeds={seeds}")
    print("loading frozen LLM (number-word features harvested ONCE)...")
    t0 = time.time()
    tok, model = load_llm(args.model, dtype=getattr(torch, args.dtype))
    feats, wid = harvest_features(tok, model, NUMBER_WORDS, CARRIER, layers)
    H = int(feats[layers[0]].shape[1])
    print(f"  harvested in {time.time()-t0:.1f}s; feature dim H={H}")

    report = dict(model=args.model, device=DEV, layers=layers, N=N, feature_dim=H,
                  carrier=CARRIER, seeds=list(seeds),
                  env="cloze/.venv (torch 2.11+cu128, RTX 5080)", primary={}, perm={}, controls={})

    # ----- PRIMARY: mod-N cipher over number words; sweep layers (real features) -----
    print("\n=== PRIMARY: mod-N cipher over number words, sidecar on FROZEN real features ===")
    print("layer | K | sidecar(untaught)+-std | lookup | chance | ICL-ceiling | legibility")
    best_layer, best_score, best_m = None, -1, None
    for L in layers:
        res, m = run_layer(f"L{L}", feats[L], N, NUMBER_WORDS, tok, model,
                           episode_mod, apply_rule_mod, do_icl=True, steps=args.steps, seeds=seeds)
        report["primary"][L] = res
        for K in sorted(res["genK"].keys()):
            gg = res["genK"][K]; iclK = res["icl"].get(K, float("nan"))
            print(f"  {L:3d} | {K} |     {gg['sidecar_untaught']:.3f}+-{gg['sidecar_untaught_std']:.3f}     "
                  f"| {gg['lookup_untaught']:.3f}  | {res['chance']:.3f} |    {iclK:.3f}    |   {res['legibility']:.3f}")
        score = sum(res["genK"][K]["sidecar_untaught"] for K in res["genK"]) / len(res["genK"])
        if score > best_score:
            best_layer, best_score, best_m = L, score, m
    report["best_layer"] = best_layer
    print(f"\nbest layer by mean untaught acc: L{best_layer} (mean {best_score:.3f})")

    # ----- CONTROLS at the best layer: what does the win actually depend on? -----
    # Honesty: the mod task may be solvable by ANY distinct code. Compare real features to
    # random-gaussian / one-hot / collapsed feature sets, and add a PERMUTATION negative control.
    if not args.skip_controls:
        print("\n=== CONTROLS @ L{} (does the win need REAL features, or just distinct ones?) ===".format(best_layer))
        rf = feats[best_layer]
        ctrl_sets = {
            "real":      rf,
            "gaussian":  torch.randn(N, H, device=DEV),
            "one_hot":   F.pad(torch.eye(N, device=DEV), (0, H - N)) if H >= N else torch.eye(N, device=DEV),
            "collapsed": rf.mean(0, keepdim=True).expand(N, -1) + 1e-3 * torch.randn(N, H, device=DEV),
        }
        print("  feature-set | K1 | K2 | K3 | K5   (mod cipher untaught acc)")
        for name, fs in ctrl_sets.items():
            res_c, _ = run_layer(f"ctrl_{name}", fs.contiguous(), N, NUMBER_WORDS, tok, model,
                                 episode_mod, apply_rule_mod, do_icl=False, steps=args.steps, seeds=(0,))
            report["controls"][name] = res_c["genK"]
            print("  {:10s} | ".format(name) + " | ".join(f"{res_c['genK'][K]['sidecar_untaught']:.3f}" for K in (1,2,3,5)))
        # feature-noise robustness: real vs gaussian (reveals real features' anisotropy weakness)
        print("\n  feature-noise robustness (train-time additive noise; real vs gaussian):")
        for name, fs in [("real", rf), ("gaussian", ctrl_sets["gaussian"])]:
            res_n, _ = run_layer(f"fn_{name}", fs.contiguous(), N, NUMBER_WORDS, tok, model,
                                 episode_mod, apply_rule_mod, do_icl=False, steps=args.steps,
                                 seeds=(0,), feat_noise=2.0)
            report["controls"][f"{name}_featnoise2"] = res_n["genK"]
            print("  {:10s} (noise=2.0) | ".format(name) + " | ".join(f"{res_n['genK'][K]['sidecar_untaught']:.3f}" for K in (1,2,3,5)))

    # legibility detail + circle on the best layer (representative seed-0 model)
    g = torch.Generator(device=DEV).manual_seed(7)
    means = mean_state_per_b(best_m, N, episode_mod, g)
    svg_genK(os.path.join(RUNS, f"sidecar_real_genK{args.tag}.svg"),
             report["primary"][best_layer]["genK"], report["primary"][best_layer]["icl"],
             f"Consolidation on a REAL frozen LLM ({args.model.split('/')[-1]}, layer {best_layer})", N)
    svg_circle(os.path.join(RUNS, f"sidecar_real_circle{args.tag}.svg"), means, N,
               f"Secret shift b read out of the sidecar (real LLM features, L{best_layer})")

    # ----- PERMUTATION negative control: untaught generalization is impossible -> ~chance -----
    if not args.skip_perm:
        print("\n=== NEGATIVE CONTROL: permutation cipher over common words (no algebraic rule) ===")
        print("  (untaught outputs are unconstrained by taught ones -> generalization must be ~chance)")
        ep_perm, ap_perm = make_perm_episode()
        pfeats, _ = harvest_features(tok, model, PERM_WORDS, CARRIER_PERM, [best_layer])
        res_p, _ = run_layer(f"perm_L{best_layer}", pfeats[best_layer], len(PERM_WORDS), PERM_WORDS,
                             tok, model, ep_perm, ap_perm, do_icl=False, steps=args.steps,
                             seeds=(0,), legible=False)
        report["perm"][best_layer] = res_p
        print("  K | sidecar(untaught) | lookup | chance")
        for K in sorted(res_p["genK"].keys()):
            gg = res_p["genK"][K]
            print(f"  {K} |     {gg['sidecar_untaught']:.3f}       | {gg['lookup_untaught']:.3f}  | {res_p['chance']:.3f}")

    json.dump(report, open(os.path.join(RUNS, f"sidecar_real{args.tag}.json"), "w"), indent=2)
    print(f"\nwrote runs/sidecar_real{args.tag}.json, sidecar_real_genK{args.tag}.svg, sidecar_real_circle{args.tag}.svg")

if __name__ == "__main__":
    main()
