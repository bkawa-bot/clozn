"""anchored.py -- ANCHORED MEMORY (X7 productized, notes/X7_PRODUCT_DESIGN.md): a card's memory as a
k-sparse bag of NAMED word-directions, so "what do you remember?" is a LOOKUP (the alpha table), never a
generation, and deleting a word from memory is a REAL edit (remove the direction, refit the rest).

    bag      = k-sparse {token: alpha} fit by OMP against the centroid of the card's content-word
               dir(token)s (clozn.analysis.microscope.decompose -- the SAME algorithm
               notes/x7_legible_memory/alpha_learning.py:fit_topk validated live)
    bag_vec  = normalize( sum_i alpha_i * dir(token_i) )                    (stored at fit time)
    compose  = v = normalize( sum_i g_i * normalize(bag_i) ), injected ONCE per generation at
               coef = s_total * BASE_NORM, s_total = SCALE * max_i(g_i)     (X7_PRODUCT_DESIGN.md section 4:
               the injection budget is ~one residual's worth TOTAL, never per-card)

THE VALIDATED ENVELOPE (hard -- notes/x7_legible_memory/LIVE_RESULTS.md, live tax curve 2026-07-11):
  * CONTENT traits anchor: parity with the trained free prefix at k >= 4, s = 0.5 (space: tax ~0 at
    k=4/8/32; baking: modest tax +0.33-0.67). Coherent (no loops) from k >= 4.
  * RULE/STYLE traits FAIL at every k (concise: tax +0.80-0.95 -- injecting the token "concise" can NAME
    brevity, it cannot BE brief). This module therefore REFUSES to anchor style/rule cards, with the
    measured reason (REFUSAL_STYLE_RULE) -- those route to the tone dials instead.
  * Operating point: L21, s=0.5, k in [4, 16] (K_DEFAULT=4 is the measured parity point; K_MAX=8 here,
    the conservative middle of the sweet spot). s=0.25 is coherent but too weak for content expression.

HONESTY (binding, X7_PRODUCT_DESIGN.md section 9): never "the model remembers X" -- X is injected as
weighted concept directions. The alpha table IS what is injected, not what is "known"; whatlearned() is a
pure read of the stored decomposition (WHATLEARNED_NOTE ships in every response). A refused style card is
a MEASURED boundary, not a policy choice.

Seams (nothing here talks to an engine directly):
  * directions come through clozn.analysis.microscope.DirProvider (dir_of_token(token) -> vec | None).
    The production provider is ConceptSteerDirProvider below, wrapping the SAME ConceptSteer path
    /steer/concept/* already drives (clozn/behavior/steering/concept_dir.py: /score token resolution +
    /jlens/unembed_row + dir(c) = normalize(J_l^T @ W_U[c])). Tests use a fake provider -- no engine.
  * the compiled payload rides the EXISTING steer wire (clozn/behavior/steering/engine_adapter.py +
    concept_dir.ConceptSteer conventions): `vector` is a UNIT direction the engine multiplies by `coef`
    server-side (engine_client.intervene(prompt, vector=, coef=, layer=) / score(steer_vec=..,
    steer={"coef","layer"})), and `steer_vec` is the SAME direction pre-scaled (coef folded in) for
    EngineSubstrate.chat's kw["steer_vec"] + kw["steer"]={"coef": 1.0, "layer": LAYER} convention.
  * persistence mirrors clozn/memory/cards.py exactly: one flat JSON file (BAGS_PATH, module-level so
    tests repoint it), atomic writes, IO that NEVER raises -- a corrupt/missing store is an empty store.

Runtime guard (X7_PRODUCT_DESIGN.md section 5): detect_loop() is the productized degeneracy flag;
halve_steer() is the retry payload. The substrate wiring lives in clozn/server/app.py
(_anchored_loop_guard, called from EngineSubstrate.chat()/chat_stream()): when detect_loop() fires on a
FULL-STRENGTH anchored injection, retry once at s_total/2 (halve_steer); if that still loops, do a final
pass with the anchored steer ZEROED entirely. Every fired guard is recorded honestly in the run's
mem_out["anchored_loop_guard"] and surfaces as a "memory-retried" / "memory-loop-guard" run flag --
over-injection becomes a visible, self-healing event, never a mystery bad reply, and never a claim that
the memory "worked". The streaming twin (chat_stream) can only detect-and-flag after the fact -- the
engine sets the steer at generation-start and the client has already received the pieces by the time a
loop would be detected, so there is no seamless mid-stream retry there.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

import numpy as np

from clozn._io import atomic_write_json
from clozn.analysis import microscope

# ================================================================== the validated envelope (constants)

# The fitted/validated injection layer: dir(c) causally raises token c's logprob at L21 at all tested
# scales, content-specifically (notes/keystone_dirc/README.md GREENLIT 2026-07-11); the whole X7 tax
# curve ran at L21. L2 fails outright; L25 needs s<=0.5 and was not used for the memory run.
LAYER = 21

# The content-trait operating scale: s=0.5 is where content reaches parity (LIVE_RESULTS.md -- s=0.25 is
# coherent but too weak: baking d collapses to ~0). The composition budget is SCALE * max(gate), TOTAL.
SCALE = 0.5

# Median L21 residual-row L2 norm, Qwen2.5-7B-Instruct Q4_K_M -- the realistic injection magnitude a
# UNIT direction is multiplied by on the wire (coef = s * BASE_NORM). Measured over cached hf_hidden
# activations by ../clozn-jlens-work/scripts/run_j5a_swap.py (MEDIAN_NORM; norm_calibration in
# dirc_selfconsistency_results.json); same value as clozn/behavior/steering/concept_dir.py's
# VALIDATED_MEDIAN_RESID_NORM[21]. Model/layer-specific: only valid for the fitted J-lens model.
BASE_NORM = 146.68

K_DEFAULT = 4     # the measured parity point for content traits (LIVE_RESULTS.md: space PARITY at k=4)
K_MAX = 8         # conservative middle of the validated k in [4, 16] sweet spot

# X7_PRODUCT_DESIGN.md section 3 step 1: >= 3 content candidates must resolve to directions for a card
# to be anchorable at all; below that it stays prompt-mode (deterministic, explainable, no model call).
MIN_CONTENT_WORDS = 3

ENVELOPE = "L21 / content tokens / s=0.5 / k in [4,8] -- X7 LIVE_RESULTS 2026-07-11"

# The measured refusal (LIVE_RESULTS.md: concise tax +0.80-0.95 at EVERY k; the boundary runs along
# trait CLASS, not k). Verbatim product copy -- a style/rule card cannot be anchored, only dial-routed.
REFUSAL_STYLE_RULE = ("anchored memory carries CONTENT, not rules/style -- measured in X7: rule cards "
                      "failed at every k. Route style to dials instead.")

# Ships with every whatlearned() response -- the structural-honesty claim, stated.
WHATLEARNED_NOTE = "this is a lookup of the stored decomposition, not a model self-report."

# One flat JSON store (dict: card_id -> bag), mirroring cards.py's CARDS_PATH pattern -- module-level so
# tests can repoint it at a temp file.
BAGS_PATH = os.path.join(os.path.expanduser("~/.clozn"), "anchored_bags.json")


# ============================================================================ content-word extraction

# Small inline stopword list: function words + the generic preference verbs every card leads with
# ("likes/enjoys/prefers X") -- without these, every bag would anchor "enjoys" instead of X. Content
# nouns are deliberately NOT filtered; this is a floor, not a lexicon.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "was", "were", "been", "being", "has", "had", "have",
    "this", "that", "these", "those", "with", "they", "them", "their", "then", "than", "from", "into",
    "onto", "your", "yours", "you", "our", "ours", "out", "one", "all", "any", "can", "could", "should",
    "would", "will", "just", "like", "likes", "liked", "love", "loves", "loved", "enjoy", "enjoys",
    "enjoyed", "prefer", "prefers", "preferred", "want", "wants", "wanted", "really", "very", "often",
    "some", "such", "only", "more", "most", "other", "over", "under", "about", "when", "what", "where",
    "which", "who", "whom", "why", "how", "its", "his", "her", "hers", "him", "she", "also", "because",
    "while", "does", "did", "doing", "done", "there", "here", "each", "own", "same", "too", "now",
    "get", "gets", "got", "use", "uses", "using", "used", "thing", "things", "stuff", "person",
    "people", "user", "every", "always", "never", "much", "many", "lot", "lots", "new", "way", "ways",
})

# Cheap style/rule heuristic (task-pinned patterns + the obvious tone words). This backs up the card's
# own `kind` field -- either signal refuses. Deliberately coarse: a false "rule" costs a prompt-mode
# card; a false "content" costs a measured-to-fail injection.
_STYLE_RULE_RE = re.compile(
    r"\b(always|never|respond|responds|answer|answers|reply|replies)\b"
    r"|\bbe\s+(more|less)\b"
    r"|\bkeep\s+(it|your|answers?|responses?|replies)\b"
    r"|\b(concise|concisely|briefly|brevity|terse|verbose|formal|informal|casual|polite|tone)\b",
    re.IGNORECASE)

_STYLE_RULE_KINDS = frozenset({"style", "rule", "instruction", "behavior"})


def content_words(text: str) -> list[str]:
    """The card's own candidate bank: lowercase word pieces, len > 2, stopword-filtered, deduped in
    first-appearance order. This is the whole dictionary a bag is fit from -- the card text itself, per
    the X7 recipe (no external vocabulary, so every anchor is a word the user can see on the card)."""
    if not text or not isinstance(text, str):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for w in re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower()):
        w = w.strip("'-")
        if len(w) <= 2 or w in _STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def is_style_or_rule(card: dict) -> bool:
    """True when this card is a style/behavioral-rule card -- the class LIVE_RESULTS.md measured as
    failing to anchor at every k. Checks the card's own `kind` first, then the cheap imperative/tone
    text heuristic (always/never/respond/answer/be more/keep it .../concise/tone...)."""
    if not isinstance(card, dict):
        return False
    if str(card.get("kind", "")).strip().lower() in _STYLE_RULE_KINDS:
        return True
    return bool(_STYLE_RULE_RE.search(str(card.get("text") or "")))


# ==================================================================================== the live provider

class ConceptSteerDirProvider:
    """microscope.DirProvider over the LIVE engine path -- the SAME seam /steer/concept/* already drives:
    a clozn/behavior/steering/concept_dir.py ConceptSteer, whose .compute(word) resolves the word to one
    vocab token via the engine's /score, fetches W_U[token] via /jlens/unembed_row, and returns the UNIT
    dir(c) = normalize(J_l^T @ W_U[c]) at the given layer. compute() never raises (labeled blocked dicts)
    and caches per (concept, layer), so a bag re-fit is cheap. Any failure -- multi-token word, engine
    down, no J-lens sidecar -- becomes None here, which decompose/fit simply skips (drop-the-word is the
    honest default for multi-token words, per X7_PRODUCT_DESIGN.md section 3 step 2)."""

    def __init__(self, concept_steer, layer: int = LAYER):
        self.cs = concept_steer
        self.layer = int(layer)

    def dir_of_token(self, token: str):
        try:
            built = self.cs.compute(token, layer=self.layer)
        except Exception:
            return None
        if not (isinstance(built, dict) and built.get("ok")):
            return None
        vec = built.get("vector")
        if not vec:
            return None
        return np.asarray(vec, dtype=np.float32)


# ========================================================================================== the store
# Mirrors clozn/memory/cards.py: flat JSON, atomic writes, IO never raises. Store shape:
# {card_id: bag}; bag keys documented in fit_bag().

def _load() -> dict:
    """The whole store; {} if missing/corrupt/wrong-shaped (never raises)."""
    try:
        if not os.path.isfile(BAGS_PATH):
            return {}
        import json
        with open(BAGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(bags: dict) -> bool:
    """Persist the whole store; False on any failure (never raises). Atomic via clozn._io -- a failure
    can never truncate the prior file."""
    try:
        atomic_write_json(BAGS_PATH, bags)
        return True
    except Exception:
        return False


def load_bags() -> dict:
    """{card_id: bag} -- the whole store, freshly read."""
    return _load()


def get_bag(card_id: str) -> dict | None:
    return _load().get(card_id)


def put_bag(bag: dict) -> dict | None:
    """Insert/replace a bag under its card_id; the bag back on success, None on IO failure."""
    try:
        card_id = bag["card_id"]
    except Exception:
        return None
    bags = _load()
    bags[card_id] = bag
    return bag if _save(bags) else None


def remove_bag(card_id: str) -> bool:
    bags = _load()
    if card_id not in bags:
        return False
    del bags[card_id]
    return _save(bags)


def set_on(card_id: str, on: bool) -> dict | None:
    """Toggle a bag's participation in compile_steer(); the updated bag, or None if absent/IO failure."""
    bags = _load()
    bag = bags.get(card_id)
    if bag is None:
        return None
    bag["on"] = bool(on)
    return bag if _save(bags) else None


def active_bags() -> list[dict]:
    """Every stored bag with on=True (the compile/whatlearned working set)."""
    return [b for b in _load().values() if isinstance(b, dict) and b.get("on", True)]


def public_bag(bag: dict) -> dict:
    """The HTTP/UI shape of a bag: everything except the raw `vector` floats (d_model of them --
    meaningless to a reader, heavy on the wire; the alpha table IS the display)."""
    return {k: v for k, v in bag.items() if k != "vector"}


# ============================================================================================= the fit

def _resolve_dirs(words, provider) -> dict:
    """{word: unit direction} for every word the provider can place; a raising/None/degenerate lookup
    skips that word (mirrors microscope.decompose_with_provider's tolerance)."""
    out: dict[str, np.ndarray] = {}
    for w in words:
        try:
            vec = provider.dir_of_token(w)
        except Exception:
            continue
        if vec is None:
            continue
        v = np.asarray(vec, dtype=np.float64)
        if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
            continue
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        out[w] = v / n
    return out


def _bag_vector(terms, dirs: dict) -> np.ndarray | None:
    """normalize( sum_i alpha_i * dir(token_i) ) -- the composed unit bag direction, stored so compile
    and whatlearned never need a provider again."""
    vec = None
    for t in terms:
        d = dirs.get(t.token if hasattr(t, "token") else t["token"])
        if d is None:
            continue
        a = float(t.alpha if hasattr(t, "alpha") else t["alpha"])
        vec = a * d if vec is None else vec + a * d
    if vec is None:
        return None
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return None
    return vec / n


def fit_bag(card: dict, provider, k: int = K_DEFAULT, *, lens_manifest_hash: str | None = None) -> dict:
    """Fit an anchored bag for a card. PURE (no store write -- callers put_bag() the result).

    The X7 recipe (X7_PRODUCT_DESIGN.md section 3): dictionary = the card's own content words resolved to
    unit dir(token)s through `provider`; target = the centroid of those directions (the validated target;
    the raw-residual target was tried and REJECTED in the lab -- sign-arbitrary alphas); fit = OMP top-k
    via microscope.decompose (the same algorithm alpha_learning.fit_topk validated live, ms on CPU).

    Returns ONE of:
      {"refused": True, "reason": ..., "card_id": ...}
          -- a style/rule card (the measured X7 failure class, REFUSAL_STYLE_RULE), or a card with fewer
             than MIN_CONTENT_WORDS resolvable content words (stays prompt-mode), or a degenerate fit.
      {"refused": False, "bag": {...}}
          -- bag keys: card_id, card_text, terms [{token, alpha} sorted by |alpha| desc], k (terms
             actually used), k_requested, reconstruction_cos, residual_norm, vector (UNIT list[float] --
             the composed direction), candidate_bank, layer, scale, on (True), envelope, fitted_at,
             lens_manifest_hash (None when unknown; a changed lens should trigger a re-fit -- ms).
    """
    card_id = card.get("id") if isinstance(card, dict) else None
    if is_style_or_rule(card):
        return {"refused": True, "reason": REFUSAL_STYLE_RULE, "card_id": card_id}

    words = content_words(str(card.get("text") or ""))
    dirs = _resolve_dirs(words, provider)
    if len(dirs) < MIN_CONTENT_WORDS:
        return {"refused": True, "card_id": card_id,
                "reason": (f"only {len(dirs)} content word(s) resolved to single-token directions "
                           f"(need >= {MIN_CONTENT_WORDS}) -- this card stays prompt-mode "
                           "(X7_PRODUCT_DESIGN.md section 3 step 1)")}

    try:
        k_eff = max(1, min(int(k), K_MAX))
    except (TypeError, ValueError):
        k_eff = K_DEFAULT

    # The validated target: normalize(mean_i dir(token_i)) over the card's bank (decompose
    # unit-normalizes internally, so passing the raw mean is exact).
    bank = list(dirs)                                   # resolved words, first-appearance order
    target = np.mean([dirs[w] for w in bank], axis=0)
    decomp = microscope.decompose(target, dirs, k=k_eff)
    if not decomp.terms:
        return {"refused": True, "card_id": card_id,
                "reason": "degenerate fit -- the card's word directions could not reconstruct a target"}

    vec = _bag_vector(decomp.terms, dirs)
    if vec is None:
        return {"refused": True, "card_id": card_id,
                "reason": "degenerate fit -- the composed bag direction has ~zero norm"}

    bag = {
        "card_id": card_id,
        "card_text": str(card.get("text") or ""),
        "terms": [{"token": t.token, "alpha": float(t.alpha)} for t in decomp.terms],
        "k": decomp.k_used,
        "k_requested": k_eff,
        "reconstruction_cos": float(decomp.reconstruction_cos),
        "residual_norm": float(decomp.residual_norm),
        "vector": [float(x) for x in vec],
        "candidate_bank": bank,
        "layer": LAYER,
        "scale": SCALE,
        "on": True,
        "envelope": ENVELOPE,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "lens_manifest_hash": lens_manifest_hash,
    }
    return {"refused": False, "bag": bag}


def delete_term(card_id: str, token: str, provider) -> dict:
    """THE edit: remove ONE word-direction from a stored bag and refit the remaining alphas jointly
    (microscope.decompose over exactly the remaining terms' directions -- OMP with dictionary == support
    degenerates to the joint least-squares refit). The word is removed from BOTH the terms AND the
    candidate bank, so the refit target (the remaining bank's centroid) no longer contains the deleted
    direction at all -- the word is gone from the memory, not just hidden from the display. Deleting the
    last term deletes the whole bag (an empty memory is no memory). The MIN_CONTENT_WORDS floor is
    deliberately NOT re-applied here: an explicit user deletion wins over the anchorability heuristic.

    Returns {"ok": True, "bag": updated-bag} / {"ok": True, "bag": None, "deleted_bag": True} /
    {"ok": False, "reason": ...}. Persists on success.
    """
    bag = get_bag(card_id)
    if bag is None:
        return {"ok": False, "reason": f"no anchored bag for card {card_id!r}"}
    token = str(token or "").strip().lower()
    terms = [t.get("token") for t in bag.get("terms", []) if isinstance(t, dict)]
    if token not in terms:
        return {"ok": False, "reason": f"bag for {card_id!r} has no term {token!r}"}

    remaining_terms = [t for t in terms if t != token]
    remaining_bank = [w for w in bag.get("candidate_bank", terms) if w != token]
    if not remaining_terms:
        remove_bag(card_id)
        return {"ok": True, "bag": None, "deleted_bag": True,
                "note": "last term removed -- the whole bag was deleted"}

    dirs = _resolve_dirs(set(remaining_bank) | set(remaining_terms), provider)
    term_dirs = {w: dirs[w] for w in remaining_terms if w in dirs}
    bank_vecs = [dirs[w] for w in remaining_bank if w in dirs] or list(term_dirs.values())
    if not term_dirs or not bank_vecs:
        return {"ok": False, "reason": "could not re-resolve the remaining term directions to refit"}

    target = np.mean(bank_vecs, axis=0)
    decomp = microscope.decompose(target, term_dirs, k=len(term_dirs))
    if not decomp.terms:
        return {"ok": False, "reason": "refit degenerated -- remaining terms explain none of the target"}
    vec = _bag_vector(decomp.terms, term_dirs)
    if vec is None:
        return {"ok": False, "reason": "refit degenerated -- composed direction has ~zero norm"}

    bag["terms"] = [{"token": t.token, "alpha": float(t.alpha)} for t in decomp.terms]
    bag["k"] = decomp.k_used
    bag["reconstruction_cos"] = float(decomp.reconstruction_cos)
    bag["residual_norm"] = float(decomp.residual_norm)
    bag["vector"] = [float(x) for x in vec]
    bag["candidate_bank"] = remaining_bank
    bag["fitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    if put_bag(bag) is None:
        return {"ok": False, "reason": "could not persist the edited bag"}
    return {"ok": True, "bag": bag}


# ======================================================================================== the compose

def compile_steer(bags: list[dict] | None = None, gates: dict | None = None,
                  scale: float = SCALE) -> dict | None:
    """Compose every active bag into ONE steer payload under the fixed X7 budget
    (X7_PRODUCT_DESIGN.md section 4 -- total injection is ~one residual's worth, NEVER per-card):

        v       = normalize( sum_i g_i * normalize(bag_i.vector) )
        s_total = scale * max_i(g_i)          (the budget scales with the most-relevant memory)
        coef    = s_total * BASE_NORM

    `bags`: explicit bag dicts, or None -> active_bags() from the store. `gates`: {card_id: g in [0,1]}
    (e.g. from clozn.memory.topic_gate); a bag missing from a given gates map defaults to 1.0 (fail-open,
    matching topic_gate's degrade-to-baseline contract); gates are clamped to [0,1] and g<=0 drops the
    bag. Pure read -- stored bag vectors only, NO provider/engine call.

    Returns None when nothing composes (no active bags / all gates zero), else:
      {"ok": True, "layer": LAYER,
       "vector": UNIT list[float],   # the intervene/score wire: engine multiplies coef * vector
       "coef": float,                # pair with `vector` (concept_dir.ConceptSteer convention)
       "steer_vec": list[float],     # SAME direction pre-scaled (coef folded in) -- EngineSubstrate.chat's
                                     #   kw["steer_vec"] + kw["steer"]={"coef": 1.0, "layer": LAYER}
       "s_total": float,
       "bags": [{"card_id", "gate", "alpha_top3"}]}   # the run-record field (design section 4 step 4)
    """
    if bags is None:
        bags = active_bags()
    entries = []
    for bag in bags:
        if not isinstance(bag, dict) or not bag.get("on", True):
            continue
        raw = bag.get("vector")
        if not raw:
            continue
        g = 1.0 if gates is None else float(gates.get(bag.get("card_id"), 1.0))
        g = min(1.0, max(0.0, g))
        if g <= 0.0:
            continue
        v = np.asarray(raw, dtype=np.float64)
        if v.ndim != 1 or not np.all(np.isfinite(v)):
            continue
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        entries.append((bag, g, v / n))
    if not entries:
        return None

    # Dimension guard: bags fit under a different lens/model (a different d_model) would make np.sum
    # raise, and the caller's blanket except would then silently drop ALL anchored memory. Keep only the
    # bags matching the first entry's width, so a stale bag skips itself instead of killing everything.
    dim = len(entries[0][2])
    entries = [e for e in entries if len(e[2]) == dim]

    total = np.sum([g * v for _, g, v in entries], axis=0)
    n = float(np.linalg.norm(total))
    if n < 1e-12:
        return None                      # opposed bags cancelled exactly -- nothing honest to inject
    v = total / n
    s_total = float(scale) * max(g for _, g, _ in entries)
    coef = s_total * BASE_NORM
    return {
        "ok": True,
        "layer": LAYER,
        "vector": [float(x) for x in v],
        "coef": float(coef),
        "steer_vec": [float(x) for x in (v * coef)],
        "s_total": float(s_total),
        "bags": [{"card_id": bag.get("card_id"), "gate": float(g),
                  "alpha_top3": [{"token": t.get("token"), "alpha": t.get("alpha")}
                                 for t in bag.get("terms", [])[:3]]}
                 for bag, g, _ in entries],
    }


# ======================================================================================== the receipt

def _decomposition_of(bag: dict) -> microscope.Decomposition:
    """Rebuild a microscope.Decomposition from a STORED bag, so render_receipt stays the single renderer
    (a pure function of stored data -- it cannot state a word that isn't in the bag)."""
    terms = [microscope.Term(token=str(t.get("token")), alpha=float(t.get("alpha", 0.0)), order=i)
             for i, t in enumerate(bag.get("terms", [])) if isinstance(t, dict)]
    return microscope.Decomposition(terms=terms,
                                    reconstruction_cos=float(bag.get("reconstruction_cos") or 0.0),
                                    residual_norm=float(bag.get("residual_norm") or 0.0),
                                    k_used=len(terms))


def alpha_table(bag: dict) -> str:
    """The human table for one bag ('+0.475  bakery' per line, |alpha| desc) -- a lookup, rendered by
    microscope.render_receipt over the stored terms. No model call exists in this code path."""
    return microscope.render_receipt(_decomposition_of(bag))


def whatlearned(bags: list[dict] | None = None) -> dict:
    """"What do you remember?" as a PURE LOOKUP over every active bag's stored decomposition -- no
    provider, no engine, no generation, so memory-content confabulation is structurally impossible
    (the X7 claim, verbatim). Each term carries its bag's reconstruction_cos (the honesty dial: low cos
    means the card's own words barely span the target -- read the words skeptically)."""
    if bags is None:
        bags = active_bags()
    out = []
    for bag in bags:
        if not isinstance(bag, dict) or not bag.get("on", True):
            continue
        cos = float(bag.get("reconstruction_cos") or 0.0)
        out.append({
            "card_id": bag.get("card_id"),
            "card_text": bag.get("card_text", ""),
            "reconstruction_cos": cos,
            "k": bag.get("k"),
            "terms": [{"token": t.get("token"), "alpha": t.get("alpha"), "reconstruction_cos": cos}
                      for t in bag.get("terms", []) if isinstance(t, dict)],
            "table": alpha_table(bag),
        })
    return {"note": WHATLEARNED_NOTE, "envelope": ENVELOPE, "bags": out}


# ==================================================================================== the loop guard

def detect_loop(pieces, window: int = 8) -> bool:
    """Runtime degeneracy guard (X7_PRODUCT_DESIGN.md section 5, productizing the lab's max-word-share
    loop flag): True when the last `window` generated pieces are a VERBATIM repeated cycle -- i.e. they
    are periodic with some period p <= window//2 (so at least two full cycles are present). Fires on
    'the cake the cake the cake the cake' and on a single stuttered token; never on ordinary prose,
    which has no exact short period.

    Contract for the substrate (wired by the lead, NOT here): call this per-generation over the pieces
    emitted so far; when it fires, ZERO the anchored steer for the REST of that reply (and flag the run,
    e.g. flags: ["memory-loop-guard"]) -- over-injection becomes a visible, self-healing event. Fewer
    than `window` pieces, or window < 2, is always False (not enough evidence to call a loop)."""
    toks = [p for p in (str(x) for x in (pieces or [])) if p.strip()]
    try:
        w = int(window)
    except (TypeError, ValueError):
        return False
    if w < 2 or len(toks) < w:
        return False
    tail = toks[-w:]
    for period in range(1, w // 2 + 1):
        if all(tail[i] == tail[i - period] for i in range(period, w)):
            return True
    return False


def halve_steer(comp: dict) -> dict:
    """The loop guard's retry payload: `comp` (a compile_steer() result) with the injected MAGNITUDE
    halved -- s_total/2, coef/2, steer_vec scaled by 0.5 -- while `vector` (the raw unit direction),
    `layer`, and `bags` are unchanged. This is the substrate's auto-retry-once-at-half-strength policy
    (X7_PRODUCT_DESIGN.md section 5): when detect_loop() fires on a full-strength injection, retry with
    THIS payload before giving up and zeroing the steer entirely. A pure dict transform -- no store or
    provider access, never raises on a well-formed compile_steer() result."""
    if not comp:
        return comp
    out = dict(comp)
    out["s_total"] = float(comp.get("s_total") or 0.0) / 2.0
    out["coef"] = float(comp.get("coef") or 0.0) / 2.0
    out["steer_vec"] = [float(x) * 0.5 for x in (comp.get("steer_vec") or [])]
    return out


# ======================================================================================== lens hash

def lens_manifest_hash(jlens_dir: str | None = None) -> str | None:
    """sha256[:12] of the J-lens manifest bytes (~/.clozn/jlens/manifest.json, honoring CLOZN_JLENS_DIR
    -- same resolution as concept_dir._default_jlens_dir). Stored on each bag so a refit/model change
    makes stale bags detectable (X7_PRODUCT_DESIGN.md section 2: self-invalidating, re-fit is ms).
    Never raises; None when the manifest is unavailable."""
    try:
        d = jlens_dir or os.environ.get("CLOZN_JLENS_DIR") or os.path.join(
            os.path.expanduser("~"), ".clozn", "jlens")
        with open(os.path.join(d, "manifest.json"), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return None
