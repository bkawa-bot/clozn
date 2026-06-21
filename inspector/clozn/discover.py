"""
clozn.discover — unsupervised feature DISCOVERY on the hidden state (the #1 gap vs SOTA).

Everything so far probes features we NAME. This finds features the model REVEALS: collect the
state over a corpus, then surface recurring directions two ways and let each describe itself by
its top-activating tokens ("what is this one about?"), Neuronpedia-style, scaled down.

  PCA      — the simple baseline: dominant axes of variation (often polysemantic).
  TinySAE  — a minimal sparse autoencoder: a learned overcomplete-ish dictionary, the SAE idea
             meant to beat PCA on monosemanticity.

Honesty-first (the eval-rigor culture the field has pivoted to): we don't assume the fancy method
wins — we show both and let the top-activating tokens be the judge. A minimal SAE is the *simplest*
discovery method (the field has since moved to transcoders, and SAEs have known limits at small
scale); on a substrate with no off-the-shelf tooling it's the right first step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A themed corpus: recurring concepts, so discovery has something coherent to find (and we can
# check whether it rediscovers the themes we seeded — without ever telling it the labels).
THEMES: dict[str, list[str]] = {
    "color": [
        "The sky turned a deep shade of blue", "She painted the old wall bright red",
        "His favorite color has always been green", "The autumn leaves faded to yellow",
        "A single purple flower bloomed by the gate", "They chose white paint for the kitchen",
        "The sea looked grey under the clouds", "Her dress was a brilliant shade of pink",
        "Orange and red lit up the evening sky", "He drew the ocean in dark blue ink",
    ],
    "animal": [
        "The dog chased the cat across the yard", "A lion roared somewhere in the distance",
        "Birds and fish filled the little zoo", "The horse galloped over the open field",
        "An elephant is a remarkably large animal", "A small mouse hid behind the shelf",
        "The owl watched silently from the tree", "Sheep and cows grazed on the hill",
        "A snake slid quietly through the grass", "The bear wandered down to the river",
    ],
    "number": [
        "She counted one two three four and five", "He bought exactly a dozen brown eggs",
        "The total came to forty seven dollars", "Add seven and eight to get fifteen",
        "There were nearly a hundred people there", "We waited for almost twenty minutes",
        "The recipe needs three cups of flour", "They scored ten points in the first half",
        "I read the first two hundred pages", "Divide sixty by four to get fifteen",
    ],
    "food": [
        "We ate pizza and pasta for dinner", "The bread was warm and very fresh",
        "She baked a rich chocolate cake", "Add some salt and pepper to the soup",
        "Apples and oranges are healthy fruit", "He grilled the fish with lemon",
        "The rice and beans smelled wonderful", "They shared a bowl of cold ice cream",
        "I fried the eggs in a little butter", "The cheese melted over the hot pasta",
    ],
    "place": [
        "They traveled to Paris and then Rome", "The mountain stood tall above the valley",
        "We drove for hours through the desert", "The city streets were noisy and crowded",
        "A wide river runs through the forest", "She moved to a small town near the coast",
        "The island was far out in the ocean", "We hiked up the steep rocky hill",
        "The village sat quietly in the valley", "London and Tokyo are huge busy cities",
    ],
    "emotion": [
        "She felt happy and full of warm joy", "He was angry and deeply frustrated",
        "A wave of sadness washed over her", "They were excited for the long trip",
        "Fear gripped him in the dark hallway", "The good news filled them with hope",
        "He grew anxious as the time passed", "A calm peace settled over the room",
        "She cried out of pure relief and joy", "The loss left him grieving for weeks",
    ],
}
THEME_WORDS: dict[str, set[str]] = {
    "color": {"blue", "red", "green", "yellow", "purple", "white", "grey", "pink", "orange", "color"},
    "animal": {"dog", "cat", "lion", "birds", "fish", "horse", "elephant", "mouse", "owl", "sheep",
               "cows", "snake", "bear", "animal"},
    "number": {"one", "two", "three", "four", "five", "dozen", "seven", "eight", "fifteen", "hundred",
               "twenty", "ten", "sixty", "forty"},
    "food": {"pizza", "pasta", "bread", "chocolate", "cake", "soup", "apples", "oranges", "fruit",
             "fish", "rice", "beans", "cream", "eggs", "butter", "cheese"},
    "place": {"paris", "rome", "mountain", "valley", "desert", "city", "river", "forest", "town",
              "coast", "island", "ocean", "hill", "village", "london", "tokyo"},
    "emotion": {"happy", "joy", "angry", "frustrated", "sadness", "excited", "fear", "hope",
                "anxious", "calm", "peace", "relief", "grieving"},
}


def corpus() -> list[str]:
    return [s for group in THEMES.values() for s in group]


def collect_token_states(source, texts, component: str = "att_num"):
    """Feed each text, returning (vectors[N,d], tokens[N], text_id[N]) — one row per token."""
    vecs, toks, tid = [], [], []
    for i, t in enumerate(texts):
        source.reset()
        for st in source.feed(t):
            vecs.append(st.state[component][0].mean(axis=1))
            toks.append(st.meta["token"])
            tid.append(i)
    return np.stack(vecs), toks, np.array(tid)


def _as_vec(z):
    if isinstance(z, (tuple, list)):
        z = z[0]
    return z.detach().cpu().numpy().reshape(-1)


def collect_block_io(source, texts, layer: int = 9, block: str = "feed_forward"):
    """Hook a block (default the channel-mix / RWKV's MLP) and capture its (input, output) per
    token over the corpus — the (x_in, y_out) pairs a TRANSCODER is trained on. Returns
    (inputs[N,d], outputs[N,d], tokens[N], text_id[N]). Needs source.model (the HF RWKV)."""
    mod = getattr(source.model.rwkv.blocks[layer], block)
    cap: list[tuple] = []
    h = mod.register_forward_hook(lambda m, inp, out: cap.append((_as_vec(inp[0]), _as_vec(out))))
    ins, outs, toks, tid = [], [], [], []
    try:
        for i, t in enumerate(texts):
            source.reset()
            cap.clear()
            steps = source.feed(t)                       # fires the hook once per token
            for k, st in enumerate(steps):
                ins.append(cap[k][0]); outs.append(cap[k][1])
                toks.append(st.meta["token"]); tid.append(i)
    finally:
        h.remove()
    return np.stack(ins), np.stack(outs), toks, np.array(tid)


def standardize(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


@dataclass
class Feature:
    idx: int
    kind: str                       # "pca" | "sae"
    top_tokens: list[str]
    fires_on: float = 0.0           # fraction of tokens it activates (sae) / variance share (pca)
    theme: str = ""                 # best-matching seeded theme (eval only)
    purity: float = 0.0             # fraction of top tokens in that theme (eval only)


def _norm_tok(t: str) -> str:
    return t.strip().lower()


def _label_theme(top_tokens: list[str]) -> tuple[str, float]:
    """Eval helper: which seeded theme do these top tokens best match, and how purely?"""
    best, best_frac = "", 0.0
    norm = [_norm_tok(t) for t in top_tokens]
    for theme, words in THEME_WORDS.items():
        frac = sum(t in words for t in norm) / max(len(norm), 1)
        if frac > best_frac:
            best, best_frac = theme, frac
    return best, best_frac


def pca_features(Xs, toks, k: int = 12, topn: int = 8) -> list[Feature]:
    """Top-k principal axes; describe each by the tokens at its positive extreme."""
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    proj = Xs @ Vt[:k].T                                  # [N, k]
    out = []
    for j in range(k):
        order = np.argsort(proj[:, j])[::-1][:topn]
        tt = [toks[i] for i in order]
        theme, purity = _label_theme(tt)
        out.append(Feature(j, "pca", tt, float(var[j]), theme, purity))
    return out


def unlabeled_features(Xs, toks, m: int = 256, l1: float = 4e-2, steps: int = 800, seed: int = 0,
                       min_fire: float = 0.005, max_fire: float = 0.40, keep: int = 20, topn: int = 8):
    """Open-ended discovery on an UN-seeded corpus: no themes to score against, so rank live features
    by *selectivity* (how concentrated each feature's activation is on its top tokens) and return the
    most concept-like ones, described by their top-activating tokens. This is the honest 'what comes
    up' view — the model reveals whatever it uses, not what we planted."""
    sae = TinySAE(Xs.shape[1], m, l1=l1, seed=seed).fit(Xs, steps=steps)
    C = sae.codes(Xs)
    fire = (C > 1e-6).mean(0)
    scored = []
    for j in range(m):
        if not (min_fire <= fire[j] <= max_fire):
            continue
        col = C[:, j]
        order = np.argsort(col)[::-1]
        pos = col[col > 1e-6]
        sel = float(col[order[:topn]].mean() / (pos.mean() + 1e-9)) if len(pos) else 0.0
        scored.append((sel, Feature(int(j), "sae", [toks[i] for i in order[:topn]], float(fire[j]))))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:keep]], sae


def describe_sae(sae, Xs, toks, min_fire: float = 0.002, max_fire: float = 0.4,
                 keep: int = 24, topn: int = 10):
    """Describe an ALREADY-trained SAE's features: rank its live features by selectivity and list
    each one's top-activating tokens. (unlabeled_features = train + describe; this is just describe,
    for when training is done separately — e.g. a minibatched SAE on real activations.)"""
    C = sae.codes(Xs)
    fire = (C > 1e-6).mean(0)
    scored = []
    for j in range(C.shape[1]):
        if not (min_fire <= fire[j] <= max_fire):
            continue
        col = C[:, j]
        order = np.argsort(col)[::-1]
        pos = col[col > 1e-6]
        sel = float(col[order[:topn]].mean() / (pos.mean() + 1e-9)) if len(pos) else 0.0
        scored.append((sel, Feature(int(j), "sae", [toks[i] for i in order[:topn]], float(fire[j]))))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:keep]]


class TinySAE:
    """A minimal sparse autoencoder: f = relu(x·We + be); x̂ = f·Wd + bd. Trained to reconstruct
    with an L1 sparsity penalty + unit-norm decoder rows (the standard anti-shrinkage constraint)."""

    def __init__(self, d: int, m: int, l1: float = 2e-3, seed: int = 0):
        import torch
        self.torch = torch
        g = torch.Generator().manual_seed(seed)
        self.We = torch.nn.Parameter(torch.randn(d, m, generator=g) * 0.1)
        self.be = torch.nn.Parameter(torch.zeros(m))
        self.Wd = torch.nn.Parameter(torch.randn(m, d, generator=g) * 0.1)
        self.bd = torch.nn.Parameter(torch.zeros(d))
        self.l1 = l1

    def _encode(self, x):
        return (x @ self.We + self.be).clamp(min=0)

    def fit(self, X, steps: int = 600, lr: float = 3e-3, Y=None, batch_size=None, epochs: int = 10):
        """Reconstruct target Y from input X through the sparse bottleneck. Y=None ⇒ Y=X (an SAE,
        a snapshot of activations). Y=a component's output ⇒ a TRANSCODER. batch_size set ⇒
        minibatch SGD over `epochs` (scales to large activation sets); else full-batch `steps`."""
        torch = self.torch
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = Xt if Y is None else torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam([self.We, self.be, self.Wd, self.bd], lr=lr)

        def _step(xb, yb):
            f = self._encode(xb)
            recon = f @ self.Wd + self.bd
            loss = ((recon - yb) ** 2).mean() + self.l1 * f.abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                self.Wd.data /= (self.Wd.data.norm(dim=1, keepdim=True) + 1e-8)

        if batch_size:
            torch.manual_seed(0)
            n = Xt.shape[0]
            for _ in range(epochs):
                perm = torch.randperm(n)
                for s in range(0, n, batch_size):
                    idx = perm[s:s + batch_size]
                    _step(Xt[idx], Yt[idx])
        else:
            for _ in range(steps):
                _step(Xt, Yt)
        return self

    def codes(self, X):
        torch = self.torch
        with torch.no_grad():
            return self._encode(torch.tensor(X, dtype=torch.float32)).numpy()

    def recon_error(self, X, Y=None):
        torch = self.torch
        target = X if Y is None else Y
        with torch.no_grad():
            f = self._encode(torch.tensor(X, dtype=torch.float32))
            recon = (f @ self.Wd + self.bd).numpy()
        return float(np.mean((recon - target) ** 2))

    def decoder_direction(self, j: int, sd) -> np.ndarray:
        """Discovered feature j's write direction, un-standardized back to raw state units —
        i.e. the steering vector for that feature. Closes discover → steer → verify."""
        wd = self.Wd.detach().cpu().numpy()[j] * np.asarray(sd)
        return wd / (np.linalg.norm(wd) + 1e-9)


def sae_features(Xs, toks, m: int = 128, topn: int = 8, l1: float = 2e-3,
                 steps: int = 600, seed: int = 0, min_fire: float = 0.004,
                 max_fire: float = 0.5, keep: int = 12, Y=None):
    """Train a TinySAE (Y=None) or a transcoder (Y=block output) and surface its most coherent live
    features. We drop only dead features (fire < min_fire) and ubiquitous ones (fire > max_fire),
    then rank what's left by purity and keep the top `keep` — a too-dense/sparse run is reported
    honestly, never as empty. Features are described by what activates the encoder on the INPUT."""
    sae = TinySAE(Xs.shape[1], m, l1=l1, seed=seed).fit(Xs, steps=steps, Y=Y)
    C = sae.codes(Xs)                                    # [N, m]
    fire = (C > 1e-6).mean(0)                            # how often each feature activates
    feats = []
    for j in range(m):
        if not (min_fire <= fire[j] <= max_fire):
            continue
        order = np.argsort(C[:, j])[::-1][:topn]
        tt = [toks[i] for i in order]
        theme, purity = _label_theme(tt)
        feats.append(Feature(j, "sae", tt, float(fire[j]), theme, purity))
    feats.sort(key=lambda f: -f.purity)                  # most coherent first
    n_dead = int((fire < min_fire).sum())
    n_dense = int((fire > max_fire).sum())
    return feats[:keep], sae, {"dead": n_dead, "dense": n_dense, "live": len(feats)}
