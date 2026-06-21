"""
crux.py — the open crux from HANDOFF.md: sparse != interpretable?

Step 1 (reproduce): train the top-k bottleneck char model across k, report held-out
                    bits/char. Confirm the plateau-then-cliff from session 1.
Step 2 (the crux):  take the k=8 model. For each of the 256 bottleneck units,
                    (a) show its max-activating 8-char contexts, and
                    (b) quantify how much of its activation variance is explained by
                        SIMPLE, NAMEABLE features (last char, last-2 chars, char-class)
                        vs the full-input linear ceiling.
                    A unit explained by "last char is a space/vowel/..." is nameable.
                    Sparse-but-inscrutable units (low R^2 even from the full input) are
                    the result that would put the legibility bet in trouble.

Honesty rule (from the handoff): state caveats louder than wins.
"""
import os, sys, math, json, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")    # Windows console defaults to cp1252
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)
CTX, EMB, WIDTH, SEED = 8, 16, 256, 1234

# ----------------------------------------------------------------------------- data
def load_text():
    p = os.path.join(DATA, "input.txt")
    if not os.path.exists(p):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("downloading tinyshakespeare ...")
        urllib.request.urlretrieve(url, p)
    return open(p, "r", encoding="utf-8").read()

text = load_text()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
ntr = int(len(data) * 0.9)
train_ids, test_ids = data[:ntr], data[ntr:]

def make_ctx(ids):
    w = ids.unfold(0, CTX + 1, 1)          # [M, CTX+1]
    return w[:, :CTX].contiguous(), w[:, CTX].contiguous()

Xtr, Ytr = (t.to(DEV) for t in make_ctx(train_ids))
Xte, Yte = (t.to(DEV) for t in make_ctx(test_ids))
print(f"device={DEV}  vocab={V}  chars(train ctx)={len(Xtr):,}  test ctx={len(Xte):,}")

# ---------------------------------------------------------------------------- model
def topk_mask(z, k):                        # straight-through hard top-k on |z|
    if k >= z.shape[-1]:
        return z
    _, idx = z.abs().topk(k, dim=-1)
    m = torch.zeros_like(z).scatter_(-1, idx, 1.0)
    return z * m                            # grad flows to kept units only

class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.fc1 = nn.Linear(CTX * EMB, WIDTH)
        self.bottleneck = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, V)
        self.k = k
    def forward(self, x, return_z=False):
        h = self.emb(x).flatten(1)
        h = F.relu(self.fc1(h))
        z = self.bottleneck(h)              # pre-mask bottleneck activations
        h2 = F.relu(self.fc2(topk_mask(z, self.k)))
        logits = self.head(h2)
        return (logits, z) if return_z else logits

@torch.no_grad()
def bits_per_char(model, X, Y, bs=16384):
    model.eval(); tot = 0.0
    for i in range(0, len(X), bs):
        ce = F.cross_entropy(model(X[i:i+bs]), Y[i:i+bs], reduction="sum")
        tot += ce.item()
    return tot / len(X) / math.log(2)

def train_one(k, steps=3000, bs=1024, lr=2e-3):
    torch.manual_seed(SEED)                 # identical init across k (controlled)
    model = Net(k).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(SEED + 1)
    M = len(Xtr); model.train()
    for _ in range(steps):
        idx = torch.randint(0, M, (bs,), generator=g, device=DEV)
        loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# ------------------------------------------------------------------ step 1: k-sweep
KS = [256, 64, 32, 16, 8, 4, 2, 1]
REF = {256: 2.585, 64: 2.525, 32: 2.562, 16: 2.569, 8: 2.630, 4: 2.706, 2: 2.822, 1: 3.216}
print("\n=== STEP 1 — held-out bits/char vs k (lower = better compression) ===")
res, model_k8 = {}, None
for k in KS:
    t = time.time(); m = train_one(k); bpc = bits_per_char(m, Xte, Yte)
    res[k] = bpc
    print(f"  k={k:<4} ours={bpc:.3f}  handoff={REF[k]:.3f}  ({time.time()-t:.1f}s)")
    if k == 8:
        model_k8 = m
json.dump(res, open(os.path.join(RUNS, "ksweep.json"), "w"), indent=2)
best = min(res, key=res.get)
print(f"  best k = {best} ({res[best]:.3f}).  refs: 4-gram=2.534, unigram=4.779, human~1.0, LLM~0.7")

# --------------------------------------------------------------- step 2: the crux
print("\n=== STEP 2 — the crux: are the k=8 bottleneck units nameable? ===")
model_k8.eval()
with torch.no_grad():
    Z = torch.cat([model_k8(Xte[i:i+16384], return_z=True)[1] for i in range(0, len(Xte), 16384)])
N = len(Z)

# how often each unit is actually selected into the top-8
with torch.no_grad():
    sel = torch.zeros(WIDTH, device=DEV)
    for i in range(0, N, 16384):
        zz = Z[i:i+16384]
        idx = zz.abs().topk(8, dim=-1).indices
        sel += torch.zeros_like(zz).scatter_(-1, idx, 1.0).sum(0)
selfreq = (sel / N).cpu()                   # fraction of contexts where unit is in top-8

# interpretable feature sets over the test contexts
def onehot(col):
    return F.one_hot(col, V).float()

vowels = set("aeiouAEIOU")
_cls = torch.tensor([[float(itos[i] == " "), float(itos[i] == "\n"),
                      float(itos[i].isalpha()), float(itos[i].isupper()),
                      float(itos[i].islower()), float(itos[i] in vowels),
                      float(itos[i].isdigit()),
                      float((not itos[i].isalnum()) and (not itos[i].isspace()))]
                     for i in range(V)], device=DEV)
def feats(name):
    if name == "class": return _cls[Xte[:, 7]]
    if name == "last":  return onehot(Xte[:, 7])
    if name == "last2": return torch.cat([onehot(Xte[:, 7]), onehot(Xte[:, 6])], 1)
    if name == "full":  return torch.cat([onehot(Xte[:, p]) for p in range(CTX)], 1)

@torch.no_grad()
def r2_and_weights(Xf, ridge=1e-3):
    Xb = torch.cat([Xf, torch.ones(len(Xf), 1, device=Xf.device)], 1)
    D = Xb.shape[1]
    Wt = torch.linalg.solve(Xb.T @ Xb + ridge * torch.eye(D, device=Xf.device), Xb.T @ Z)
    ss_res = ((Z - Xb @ Wt) ** 2).sum(0)
    ss_tot = ((Z - Z.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    return (1 - ss_res / ss_tot).cpu(), Wt.cpu()

R2, WT = {}, {}
for s in ["class", "last", "last2", "full"]:
    R2[s], WT[s] = r2_and_weights(feats(s))

# active = units selected meaningfully often (uniform baseline would be 8/256 = 3.1%)
active = (selfreq > 0.005).nonzero().flatten().tolist()
def disp(s):  # printable chars
    return s.replace("\n", "⏎").replace(" ", "·")
def med(xs):
    xs = sorted(xs); n = len(xs)
    return 0.0 if n == 0 else (xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2)

am = [R2["last"][u].item() for u in active]
af = [R2["full"][u].item() for u in active]
ac = [R2["class"][u].item() for u in active]
print(f"  {len(active)}/{WIDTH} units are 'active' (selected in >0.5% of contexts).")
print(f"  median R^2 among active units:  char-class={med(ac):.2f}  last-char={med(am):.2f}"
      f"  last-2={med([R2['last2'][u].item() for u in active]):.2f}  full-input(ceiling)={med(af):.2f}")
for thr in (0.5, 0.7, 0.9):
    print(f"  active units with R^2(last-char) >= {thr}: "
          f"{sum(v>=thr for v in am)}/{len(active)}   |  R^2(full) >= {thr}: {sum(v>=thr for v in af)}/{len(active)}")

# per-unit detail for the most-used units
order = sorted(active, key=lambda u: -selfreq[u].item())
@torch.no_grad()
def triggers(u, m=4):                       # which last-char shifts this unit most (mean deviation)
    z = Z[:, u]; base = z.mean().item(); last = Xte[:, 7]; dev = torch.zeros(V)
    for c in range(V):
        mk = last == c
        if mk.any(): dev[c] = z[mk].mean().item() - base
    return ", ".join(f"'{disp(itos[i])}'({dev[i].item():+.2f})"
                     for i in dev.abs().topk(m).indices.tolist())
@torch.no_grad()
def top_ctx(u, m=5):
    vals, idx = Z[:, u].topk(m)
    return "  ".join(f"[{disp(''.join(itos[t] for t in Xte[j].tolist()))}]"
                     for j in idx.tolist())
print("\n  most-used units  (sel% | R2 last/full | strongest last-char triggers):")
for u in order[:14]:
    print(f"   u{u:<3} sel={selfreq[u]*100:4.1f}%  R2 last={R2['last'][u]:.2f} full={R2['full'][u]:.2f}"
          f"  after {triggers(u)}")
    print(f"        max-activating: {top_ctx(u)}")

torch.save({"state": model_k8.state_dict(), "vocab": chars, "k": 8}, os.path.join(HERE, "model_k8.pt"))
print("\n  saved k=8 model -> model_k8.pt ; sweep -> runs/ksweep.json")
print("\n  read: high last-char/class R^2 => the unit is a nameable detector.")
print("        low full-input R^2 on an active unit => sparse but NOT linearly nameable (the worrying case).")
