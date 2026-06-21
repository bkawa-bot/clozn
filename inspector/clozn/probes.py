"""
clozn.probes — fit a concept direction out of a source's state, and causally verify it.

Packages the M3 method as reusable functions so the inspector and the spikes share ONE
implementation (the seam discipline: logic lives here, scripts stay thin). Honesty-first: a
ConceptResult always carries BOTH decodability (what's linearly readable) and a causal
dose-response (what the model actually uses) — never one without the other.

Targets a logit-readable recurrent source (the RwkvStateSource interface: reset/feed/encode,
get_state/set_state, .tok, ._last_logits). The diffusion substrate will get its own readout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ops import verify_causal, LinearProbe

# parallel polar corpora — same frames, opposite valence (the M3 default concept: sentiment)
DEFAULT_POS = [
    "I love this", "what a wonderful day", "this is great", "absolutely fantastic",
    "I feel happy", "such a beautiful gift", "the best news ever", "I am delighted",
    "this made me smile", "pure joy", "an amazing experience", "I adore it",
    "wonderful and kind", "a brilliant success", "everything is perfect", "I am so grateful",
    "this is delightful", "a lovely surprise", "feeling cheerful today", "what a great win",
    "truly excellent work", "I really enjoyed that", "a heartwarming story", "incredibly fun",
]
DEFAULT_NEG = [
    "I hate this", "what a terrible day", "this is awful", "absolutely disgusting",
    "I feel miserable", "such a cruel insult", "the worst news ever", "I am furious",
    "this made me cry", "pure misery", "a horrible experience", "I despise it",
    "nasty and cruel", "a complete failure", "everything is ruined", "I am so resentful",
    "this is dreadful", "an awful surprise", "feeling gloomy today", "what a bad loss",
    "truly atrocious work", "I really hated that", "a heartbreaking story", "incredibly boring",
]
DEFAULT_POS_WORDS = [" good", " great", " happy", " love", " wonderful", " best", " nice",
                     " amazing", " excellent", " joy", " beautiful", " perfect"]
DEFAULT_NEG_WORDS = [" bad", " terrible", " sad", " hate", " awful", " worst", " horrible",
                     " miserable", " disgusting", " misery", " ugly", " wrong"]


@dataclass
class ConceptResult:
    name: str
    decodability: float                 # k-fold held-out probe accuracy (chance = 0.5)
    alphas: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)   # P(pos)/(pos+neg) per steering alpha
    verify: dict = field(default_factory=dict)          # ops.verify_causal at the headline alpha
    monotonic: bool = False

    @property
    def causal(self) -> bool:
        return bool(self.verify.get("causal")) and self.monotonic


def _layer_mean(state, component="att_num"):
    return state[component][0].mean(axis=1)             # (hidden,) — robust low-dim probe feature


def kfold_accuracy(feats, labels, k=6, ridge=10.0):
    """Standardized k-fold sign accuracy (train-fold stats only — no leakage).

    The data arrives class-sorted ([+1]*n, [-1]*n), so we shuffle (fixed seed) before splitting
    — otherwise each fold would be single-class, a noisy and unrepresentative CV."""
    feats, labels = list(feats), list(labels)
    idx = np.random.default_rng(0).permutation(len(feats))
    correct = 0
    for fold in np.array_split(idx, k):
        te = set(fold.tolist())
        tr = [i for i in idx if i not in te]
        X = np.stack([feats[i] for i in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        probe = LinearProbe("c", "m").fit([{"m": (feats[i] - mu) / sd} for i in tr],
                                          [labels[i] for i in tr], ridge=ridge)
        for i in fold:
            pred = 1.0 if probe.read({"m": (feats[i] - mu) / sd}).value >= 0 else -1.0
            correct += int(pred == labels[i])
    return correct / len(feats)


def _single_ids(tok, words):
    return np.array([tok.encode(w)[0] for w in words if len(tok.encode(w)) == 1])


def probe_and_verify(source, *, name="sentiment",
                     pos=DEFAULT_POS, neg=DEFAULT_NEG,
                     pos_words=DEFAULT_POS_WORDS, neg_words=DEFAULT_NEG_WORDS,
                     component="att_num", prime="I think it was", suffix=" really",
                     alphas=(-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3), headline_alpha=2.0):
    """Decode a concept from `source`'s state and causally verify it. Returns ConceptResult.

    (A) decodability — standardized k-fold accuracy of a LinearProbe on the layer-mean of
        `component`. (B) causality — a diff-in-means steering vector added to `component`,
        re-measuring the model's own P(pos)/(pos+neg) on the token after `prime`+`suffix`,
        swept over `alphas` (monotone ⇒ causal), with one ops.verify_causal at headline_alpha.
    """
    full = []                                          # per-text component (hidden, layers)
    for t in pos + neg:
        source.reset(); source.feed(t)
        full.append(source.get_state()[component][0])
    feats = [a.mean(axis=1) for a in full]
    labels = [1.0] * len(pos) + [-1.0] * len(neg)
    acc = kfold_accuracy(feats, labels)

    P = np.stack(full[:len(pos)]).mean(0)
    N = np.stack(full[len(pos):]).mean(0)
    steer = (P - N); steer /= (np.linalg.norm(steer) + 1e-9)
    typ = float(np.mean([np.linalg.norm(a) for a in full]))
    steer = steer * typ * 0.1                          # 1 alpha-unit = 10% of state magnitude

    pos_ids = _single_ids(source.tok, pos_words)
    neg_ids = _single_ids(source.tok, neg_words)
    suffix_ids = source.encode(suffix)

    def behavior(src):
        snap = src.get_state()
        for tid in suffix_ids:
            src.step(tid)
        p = src._last_logits.softmax(-1)[0].detach().cpu().numpy()
        ps, ns = float(p[pos_ids].sum()), float(p[neg_ids].sum())
        src.set_state(snap)
        return ps / (ps + ns) if (ps + ns) > 1e-9 else 0.5

    def intervene(alpha):
        def f(state):
            s = {k: v.copy() for k, v in state.items()}
            s[component][0] = s[component][0] + alpha * steer
            return s
        return f

    scores = []
    for a in alphas:
        source.reset(); source.feed(prime)
        source.set_state(intervene(a)(source.get_state()))
        scores.append(behavior(source))

    source.reset(); source.feed(prime)
    verify = verify_causal(source, intervene(headline_alpha), behavior)
    mono = all(scores[i] <= scores[i + 1] + 1e-6 for i in range(len(scores) - 1))
    return ConceptResult(name, acc, list(alphas), scores, verify, mono)
