"""concept_dir.py -- the any-concept dial: dir(c) = normalize(J_l^T @ W_U[c]).

Wraps the VALIDATED research primitive (../clozn-jlens-work/scripts/dirc.py; live-checked at L21,
scale 0.25-0.5: raises logprob(c) +6..9 nat over baseline, content-specific vs a random equal-norm
write; scale>=1 over-injects and loses coherence -- see
../clozn-jlens-work/artifacts/dirc_selfconsistency_result.txt and j5a_swap_result.txt) into a
product module: type ANY concept word, get a steer direction, with ZERO contrastive pos/neg
calibration prompts (unlike axes.py's tone dials, which need a harvested diff-of-means over a
whole pole pair). This module builds and caches the vector; engine_adapter.py's EngineSteer
remains the SEPARATE diff-of-means (tone-dial) mechanism -- the two compose (both ultimately ride
the same steer_vec/coef/layer wire contract: cloze_engine.EngineClient.intervene / .score, and
EngineSubstrate.chat's kw["steer_vec"]), they are not the same math.

Math (identical convention to dirc.py / oracle.py -- both are lab-only and NOT imported here; the
~15 lines below are the product's own copy of the same formulas, since the formulas themselves
are just numpy, not a restricted asset):
  * J_l is a [d_model, d_model] fp32 matrix such that `transport(h, J) = h @ J_l.T` reproduces
    `J_l @ h` for h a column vector (matches jlens.lens.JacobianLens.transport / oracle.py).
  * W_U[c] is row `c` of the model's unembed/lm_head matrix ([vocab, d_model]); `unembed(x) =
    rmsnorm(x) @ W_U.T`.
  * dir_c(token_id, layer) = normalize(J_l.T @ W_U[token_id]), optionally scaled to a realistic
    residual magnitude (`scale * typical_norm`).

*** THE ONE REAL BLOCKER (read before wiring this into anything) ***
J_l ships in the product today: ~/.clozn/jlens/J_layer{L}.f16 (the SAME raw fp16 sidecar the C++
engine's JlensServe::load reads for /jlens -- see engine/core/serve/server_shared.hpp). This
module reads it directly with plain numpy (load_jlens_jacobians below) -- no engine round trip
needed for J_l.

W_U (the model's unembed/lm_head matrix) is NOT exposed to product Python code ANYWHERE today:
  * The engine reads it straight out of the loaded GGUF's own (quantized) tensors, inside its own
    C++ process (JlensServe::load's out_head/out_norm) -- there is no HTTP route that returns the
    raw matrix, or even one row of it, or logits for an arbitrary candidate vector. /jlens computes
    a full read+transport+unembed server-side and only ever returns TOP-K TOKENS for a real
    residual it harvested from real text.
  * The only place a full-precision copy exists is the research lab
    (../clozn-jlens-work/artifacts/lm_head_weight.npy, HF-exported fp32) -- explicitly lab-only,
    never to be imported by product/server code (dirc.py's own docstring restriction).
So: THIS MODULE CANNOT BUILD A REAL dir(c) OUT OF THE BOX. `load_unembed()` raises
`UnembedUnavailable` (see BLOCKER_NOTE) unless the caller points it at a directory holding the
same 3-file shape the lab already produces (norm_weight.npy [d_model] fp32, lm_head_weight.npy
[vocab, d_model] fp32, unembed_meta.json {"rms_norm_eps": eps}) -- via the `unembed_dir=` argument
or the CLOZN_DIRC_UNEMBED_DIR env var. There is deliberately NO default path baked in (that would
be silently depending on a lab artifact from product code); every caller sees a labeled
"blocked": "unembed_unavailable" instead of a crash or a silently-wrong vector.
Shipping this for real needs ONE of:
  (a) a new engine route that returns W_U[token_id] (or, better, computes dir(c) server-side in
      one round trip, since the engine process already holds both J_l and the head) -- the
      lowest-latency, no-precision-loss option, and avoids ever handing a 152064 x 3584 fp32
      matrix (~2GB) to a Python client; or
  (b) a product-local unembed export shipped alongside the J-lens sidecar (the same 3-file shape
      above), generated once at `clozn jlens fit` time.
Neither exists yet. This is the flagged blocker -- see notes/FABLE_HANDOFF.md Build 1.

House honesty style (mirrors clozn/receipts/quant_receipts.py): never raise out of the
PRODUCT-FACING calls (ConceptSteer.compute/.steer_toward/.steer_vector) -- they return a
labeled `{"ok": False, "blocked": "...", "note": "..."}` dict instead. The lower-level math/loader
functions (dir_c, load_unembed, load_jlens_jacobians) DO raise (ValueError / FileNotFoundError /
UnembedUnavailable) -- they are the seam ConceptSteer.compute() catches, kept raising for anyone
building custom orchestration on top who wants a normal exception instead of a checked dict.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_LAYER = 21                                  # the validated tap (J5a swap-receipts, dirc self-consistency)
DEFAULT_STRENGTH = 0.35                              # midpoint of the validated 0.25-0.5 operating range
VALIDATED_SCALE_RANGE = (0.25, 0.5)

# Per-layer median residual-row L2 norm at the validated tap(s), Qwen2.5-7B-Instruct Q4_K_M -- the
# realistic injection magnitude a UNIT dir(c) should be multiplied by (`coef` on the engine's
# steer wire, which itself computes `coef * vector` server-side -- see
# engine/core/serve/routes_state.cpp / routes_whitebox.cpp's `coef * raw_vec[i]`). Sourced from
# ../clozn-jlens-work/scripts/run_j5a_swap.py's MEDIAN_NORM (measured over cached hf_hidden
# activations, ../clozn-jlens-work/artifacts/dirc_selfconsistency_results.json's norm_calibration).
# Model/layer-specific: only valid for the fitted J-lens model (today: Qwen2.5-7B-Instruct).
VALIDATED_MEDIAN_RESID_NORM = {21: 146.68, 25: 343.14}


# ============================================================================== the blocker + loaders

class UnembedUnavailable(RuntimeError):
    """No W_U (unembed/lm_head) source is configured. See the module docstring's BLOCKER section --
    J_l ships in the product J-lens sidecar; W_U does not, anywhere, today."""


BLOCKER_NOTE = (
    "dir(c) = normalize(J_l^T @ W_U[c]) needs W_U (the model's unembed/lm_head matrix, "
    "[vocab, d_model]), which is NOT exposed to product Python code anywhere today -- the engine "
    "reads it straight out of the loaded (quantized) GGUF inside its own C++ process "
    "(engine/core/serve/server_shared.hpp JlensServe::load), and no route hands back the raw "
    "matrix or a row of it. The only full-precision copy lives in the research lab "
    "(../clozn-jlens-work/artifacts/lm_head_weight.npy), which is lab-only and must not be "
    "imported by product code. Set unembed_dir= (or CLOZN_DIRC_UNEMBED_DIR) to point at a "
    "directory holding norm_weight.npy [d_model] fp32 + lm_head_weight.npy [vocab, d_model] fp32 "
    "+ unembed_meta.json {rms_norm_eps} -- there is no default; this is the flagged real blocker "
    "(notes/FABLE_HANDOFF.md Build 1), not a bug in this loader."
)


@dataclass
class UnembedWeights:
    """norm_weight [d_model] fp32, lm_head_weight [vocab, d_model] fp32, rms eps."""
    norm_weight: np.ndarray
    lm_head_weight: np.ndarray
    eps: float = 1e-6


def _default_jlens_dir() -> str:
    """~/.clozn/jlens, honoring CLOZN_JLENS_DIR -- the SAME env var name the C++ engine already
    checks (see engine/core/serve/routes_jlens.cpp's 400 body: "start with --jlens <dir> or set
    CLOZN_JLENS_DIR"), so one env var configures both sides consistently."""
    env = os.environ.get("CLOZN_JLENS_DIR")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".clozn", "jlens")


def _default_unembed_dir() -> Optional[str]:
    """No baked-in default (see BLOCKER_NOTE) -- only an explicit env var."""
    return os.environ.get("CLOZN_DIRC_UNEMBED_DIR") or None


def load_jlens_manifest(jlens_dir: Optional[str] = None) -> dict:
    d = jlens_dir or _default_jlens_dir()
    path = os.path.join(d, "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jlens_jacobians(jlens_dir: Optional[str] = None, layers=None) -> dict:
    """J_l for every requested fitted layer, read straight from the product J-lens sidecar
    (~/.clozn/jlens/J_layer{L}.f16 by default) -- the SAME raw fp16 [d_model, d_model] file
    engine/core/serve/server_shared.hpp's JlensServe::load reads for /jlens (see its "ggml layout
    Jl(k,m)=J[m,k]" comment: a flat row-major buffer with J[m,k] at offset m*d_model+k -- exactly
    what np.fromfile(...).reshape(d_model, d_model) recovers, matching dirc.py/oracle.py's
    `h @ J.T` transport convention). Upcast to fp32 for the CPU matmul.

    Raises FileNotFoundError / ValueError (never silently returns a wrong-shaped matrix) on a
    missing manifest/sidecar, an unfitted layer, or a byte-count mismatch. Returns {layer: ndarray}.
    """
    d = jlens_dir or _default_jlens_dir()
    manifest = load_jlens_manifest(d)
    d_model = int(manifest["d_model"])
    available = [int(x) for x in manifest.get("layers", [])]
    want = [int(x) for x in layers] if layers is not None else available
    out = {}
    for layer in want:
        if layer not in available:
            raise ValueError(f"layer {layer} is not a fitted J-lens layer (available: {available})")
        path = os.path.join(d, f"J_layer{layer}.f16")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing J-lens sidecar {path!r} for layer {layer}")
        flat = np.fromfile(path, dtype="<f2")
        if flat.size != d_model * d_model:
            raise ValueError(
                f"{path!r}: expected {d_model * d_model} fp16 values ({d_model}x{d_model}), got {flat.size}")
        out[layer] = flat.reshape(d_model, d_model).astype(np.float32)
    return out


def load_unembed(unembed_dir: Optional[str] = None) -> UnembedWeights:
    """Load W_U from a directory holding norm_weight.npy/lm_head_weight.npy/unembed_meta.json (the
    SAME 3-file shape ../clozn-jlens-work/scripts/dirc.py's load_unembed already produces).
    Raises UnembedUnavailable (see BLOCKER_NOTE) when no directory is configured (neither
    `unembed_dir` nor CLOZN_DIRC_UNEMBED_DIR) or the directory is missing the required files --
    this is the ONE call in this module that names the real blocker instead of degrading quietly.
    """
    d = unembed_dir or _default_unembed_dir()
    if not d:
        raise UnembedUnavailable(BLOCKER_NOTE)
    norm_path = os.path.join(d, "norm_weight.npy")
    head_path = os.path.join(d, "lm_head_weight.npy")
    meta_path = os.path.join(d, "unembed_meta.json")
    if not (os.path.isfile(norm_path) and os.path.isfile(head_path)):
        raise UnembedUnavailable(
            f"unembed_dir {d!r} is missing norm_weight.npy / lm_head_weight.npy -- {BLOCKER_NOTE}")
    norm_weight = np.load(norm_path).astype(np.float32)
    lm_head_weight = np.load(head_path).astype(np.float32)
    eps = 1e-6
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                eps = float(json.load(f).get("rms_norm_eps", eps))
        except Exception:
            pass
    if lm_head_weight.ndim != 2 or norm_weight.shape != (lm_head_weight.shape[1],):
        raise ValueError(
            f"unembed shape mismatch: norm_weight {norm_weight.shape} vs lm_head_weight {lm_head_weight.shape}")
    return UnembedWeights(norm_weight=norm_weight, lm_head_weight=lm_head_weight, eps=eps)


# ============================================================================== the math (dir(c) itself)

def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 / np.sqrt(variance + eps)) * weight.astype(np.float32)


def _transport(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """h @ J.T -- J_l @ h for each row of h (h: [n, d_model], J: [d_model, d_model])."""
    return h.astype(np.float32) @ J.astype(np.float32).T


def _unembed(x: np.ndarray, unembed_weights: UnembedWeights) -> np.ndarray:
    normed = _rmsnorm(x, unembed_weights.norm_weight, unembed_weights.eps)
    return normed @ unembed_weights.lm_head_weight.astype(np.float32).T


def dir_c(token_id: int, layer: int, J_by_layer: dict, unembed_weights: UnembedWeights, *,
          scale: float = 1.0, typical_norm: Optional[float] = None) -> np.ndarray:
    """The injection direction: normalize(J_l^T @ W_U[token_id]), magnitude-scaled.

    `typical_norm`: when given, the returned vector has norm `scale * typical_norm` (a realistic
    injection magnitude at `layer`). When None (the default), the returned vector has norm
    `scale` directly -- a unit direction at scale=1.0, which is what the self-consistency check
    uses (RMSNorm-then-argmax is invariant to any positive rescaling of its input) and what
    ConceptSteer sends over the wire (paired with a separate `coef` the engine multiplies in).

    Raises ValueError for an unfitted layer, an out-of-range token_id, or a degenerate (~zero)
    raw direction -- never silently returns a wrong/garbage vector.
    """
    if layer not in J_by_layer:
        raise ValueError(f"layer {layer} has no loaded J_l (loaded: {sorted(J_by_layer)})")
    J = J_by_layer[layer]
    lm_head = unembed_weights.lm_head_weight
    if not (0 <= int(token_id) < lm_head.shape[0]):
        raise ValueError(f"token_id {token_id} is out of range for vocab size {lm_head.shape[0]}")
    if lm_head.shape[1] != J.shape[0]:
        raise ValueError(f"d_model mismatch: J_l is {J.shape}, W_U is {lm_head.shape}")
    w_c = lm_head[int(token_id)].astype(np.float32)
    raw = J.T @ w_c
    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise ValueError(f"degenerate dir(c) for token {token_id} at layer {layer} (norm~0)")
    unit = raw / norm
    magnitude = scale if typical_norm is None else scale * typical_norm
    return (unit * magnitude).astype(np.float32)


def read_through_lens(vec: np.ndarray, layer: int, J_by_layer: dict,
                      unembed_weights: UnembedWeights) -> np.ndarray:
    """unembed(J_l @ vec) for a single [d_model] vector -- the self-consistency readout: logits
    [vocab] from pushing `vec` through the SAME J_l used to build it, then the model's own
    final-norm + unembed."""
    h = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    transported = _transport(h, J_by_layer[layer])
    logits = _unembed(transported, unembed_weights)
    return logits[0]


def rank_of(logits: np.ndarray, token_id: int) -> int:
    """0-based rank of `token_id` in `logits` (0 == top-1)."""
    return int((logits > logits[int(token_id)]).sum())


class ConceptDirSource:
    """Lazily loads + caches J_l (product sidecar) and W_U (see BLOCKER_NOTE) for one process."""

    def __init__(self, jlens_dir: Optional[str] = None, unembed_dir: Optional[str] = None):
        self.jlens_dir = jlens_dir
        self.unembed_dir = unembed_dir
        self._manifest: Optional[dict] = None
        self._J: dict = {}
        self._unembed: Optional[UnembedWeights] = None

    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = load_jlens_manifest(self.jlens_dir)
        return self._manifest

    def available_layers(self) -> list:
        try:
            return [int(x) for x in self.manifest().get("layers", [])]
        except Exception:
            return []

    def jacobians(self, layers=None) -> dict:
        want = [int(x) for x in layers] if layers is not None else self.available_layers()
        missing = [layer for layer in want if layer not in self._J]
        if missing:
            self._J.update(load_jlens_jacobians(self.jlens_dir, layers=missing))
        return self._J

    def unembed(self) -> UnembedWeights:
        """Raises UnembedUnavailable (never degrades silently) -- see BLOCKER_NOTE."""
        if self._unembed is None:
            self._unembed = load_unembed(self.unembed_dir)
        return self._unembed

    def unembed_available(self) -> bool:
        try:
            self.unembed()
            return True
        except UnembedUnavailable:
            return False


# ============================================================================== the product entry point

def _text_of(resp) -> str:
    """Extract generated text from an EngineClient .complete()/.intervene() response (OpenAI-ish
    {choices:[{text|message}]}) -- mirrors engine_adapter.EngineSteer._text."""
    ch = resp.get("choices") if isinstance(resp, dict) else None
    if ch:
        return ch[0].get("text") or (ch[0].get("message") or {}).get("content") or ""
    return (resp.get("text") or "") if isinstance(resp, dict) else str(resp)


class ConceptSteer:
    """Any-concept dial on a live engine client. Mirrors EngineSteer's persistent-dial surface
    (.set/.clear/.active/.steer_vector) so it drops into the same "dial UI" shape Fable already
    drives -- but each direction comes from dir(c), not diff-of-means harvesting, so there is no
    pos/neg calibration step: name a concept, get a direction.

    `engine_client` is a cloze_engine.EngineClient (or anything duck-typed against its
    `.score(prompt=, continuation=, topk=)` method, used only to resolve a concept WORD to its
    single vocab token id -- see resolve_token_id). Every product-facing method here NEVER raises;
    on any failure (an unresolvable/multi-token concept, an unfitted layer, or the unembed
    BLOCKER) it returns a labeled `{"ok": False, "blocked": "...", "note": "..."}` dict instead.
    """

    def __init__(self, engine_client, source: Optional[ConceptDirSource] = None,
                 layer: int = DEFAULT_LAYER, median_norm: Optional[float] = None):
        self.ec = engine_client
        self.source = source or ConceptDirSource()
        self.layer = int(layer)
        self.median_norm = float(median_norm) if median_norm is not None else VALIDATED_MEDIAN_RESID_NORM.get(self.layer)
        self.strength: dict = {}       # {concept: signed strength} -- the persistent dial state
        self._vecs: dict = {}          # {(concept, layer): unit dir(c) ndarray}
        self._token_ids: dict = {}     # {(concept, layer): token_id}

    # -- word -> token id --------------------------------------------------------------------

    def resolve_token_id(self, concept: str) -> dict:
        """Resolve a concept WORD to its single leading-space vocab token id via the engine's own
        tokenizer, reusing the EXISTING /score route (no new engine route needed): /score
        retokenizes its `continuation` text server-side and returns one entry per token (see
        cloze_engine.EngineClient.score's docstring). dir(c) needs exactly ONE vocab row, so a
        multi-token word is reported as unresolvable, not silently truncated to its first piece.

        Returns {"ok": True, "token_id": int, "piece": str} or {"ok": False, "note": "..."} --
        never raises (an engine round-trip can fail in many ways; all of them degrade here).
        """
        concept = (concept or "").strip()
        if not concept:
            return {"ok": False, "note": "empty concept"}
        try:
            r = self.ec.score(prompt="Consider the word:", continuation=" " + concept, topk=0)
        except Exception as e:
            return {"ok": False, "note": f"tokenization round-trip via /score failed: {e}"}
        toks = (r or {}).get("tokens") or []
        if len(toks) != 1:
            return {"ok": False,
                    "note": f"{concept!r} is not a single token ({len(toks)} pieces) -- dir(c) needs "
                            "exactly one vocab row; try a shorter/simpler word"}
        entry = toks[0]
        tid = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(tid, int):
            return {"ok": False, "note": "engine did not return an integer token id"}
        return {"ok": True, "token_id": tid, "piece": entry.get("piece")}

    # -- vector construction ------------------------------------------------------------------

    def compute(self, concept: str, *, layer: Optional[int] = None) -> dict:
        """Build (and cache) the UNIT dir(c) for `concept` at `layer` (default: self.layer).
        Never raises. Returns:
          {"ok": True, "concept", "layer", "token_id", "vector"} on success (`vector` is a plain
              list[float], unit norm -- pair it with a `coef` before sending to the engine; see
              steer_toward), or
          {"ok": False, "blocked": "token_resolution"|"unembed_unavailable"|"jlens_unavailable"|
              "bad_layer"|"error", "concept", "note"} on any failure.
        """
        layer = int(layer) if layer is not None else self.layer
        cache_key = (concept, layer)
        if cache_key in self._vecs:
            return {"ok": True, "concept": concept, "layer": layer,
                    "token_id": self._token_ids[cache_key], "vector": self._vecs[cache_key].tolist()}
        resolved = self.resolve_token_id(concept)
        if not resolved.get("ok"):
            return {"ok": False, "blocked": "token_resolution", "concept": concept, "layer": layer,
                    "note": resolved.get("note")}
        token_id = resolved["token_id"]
        try:
            unembed_weights = self.source.unembed()
        except UnembedUnavailable as e:
            return {"ok": False, "blocked": "unembed_unavailable", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": str(e)}
        if layer not in self.source.available_layers():
            return {"ok": False, "blocked": "bad_layer", "concept": concept, "layer": layer,
                    "token_id": token_id,
                    "note": f"no fitted J-lens sidecar for layer {layer} (available: "
                            f"{self.source.available_layers()})"}
        try:
            J_by_layer = self.source.jacobians(layers=[layer])
        except Exception as e:
            return {"ok": False, "blocked": "jlens_unavailable", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": str(e)}
        if layer not in J_by_layer:
            return {"ok": False, "blocked": "bad_layer", "concept": concept, "layer": layer,
                    "token_id": token_id,
                    "note": f"no fitted J-lens sidecar for layer {layer} (available: "
                            f"{self.source.available_layers()})"}
        try:
            vec = dir_c(token_id, layer, J_by_layer, unembed_weights, scale=1.0, typical_norm=None)
        except Exception as e:
            return {"ok": False, "blocked": "error", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": str(e)}
        self._vecs[cache_key] = vec
        self._token_ids[cache_key] = token_id
        return {"ok": True, "concept": concept, "layer": layer, "token_id": token_id, "vector": vec.tolist()}

    def steer_toward(self, concept: str, strength: float = DEFAULT_STRENGTH, *,
                     layer: Optional[int] = None) -> dict:
        """THE product entry point (mirrors the tone-dial shape: a name + a strength). Persists
        `concept`'s strength (so it composes with steer_vector() like a normal dial) AND returns
        an /intervene-ready payload in one call:
          {"ok": True, "concept", "token_id", "layer", "strength", "vector" (UNIT, list[float]),
           "coef" (float), "note"?} -- feed straight into
           engine_client.intervene(prompt, vector=res["vector"], coef=res["coef"],
                                    layer=res["layer"], max_tokens=...),
           or engine_client.score(..., steer_vec=res["vector"],
                                   steer={"coef": res["coef"], "layer": res["layer"]}).
        `coef = strength * this layer's validated median residual norm` -- the realistic
        injection magnitude (see VALIDATED_MEDIAN_RESID_NORM); the engine multiplies
        `coef * vector` itself, so `vector` stays a unit direction on the wire.
        On any failure, returns compute()'s `{"ok": False, "blocked", "note"}` shape (never
        raises), with `strength` folded in for the caller's bookkeeping.
        """
        built = self.compute(concept, layer=layer)
        if not built.get("ok"):
            built["strength"] = strength
            return built
        resolved_layer = built["layer"]
        median_norm = (self.median_norm if resolved_layer == self.layer
                       else VALIDATED_MEDIAN_RESID_NORM.get(resolved_layer))
        if not median_norm:
            return {"ok": False, "blocked": "no_norm_calibration", "concept": concept,
                    "layer": resolved_layer, "strength": strength,
                    "note": f"no validated median-residual-norm calibration for layer {resolved_layer}; "
                            "construct ConceptSteer(..., median_norm=...) explicitly for this layer"}
        self.strength[concept] = float(strength)
        coef = float(strength) * float(median_norm)
        note = None
        if abs(strength) >= 1.0:
            note = ("validated operating point is scale in [0.25, 0.5] at L21 (+6..9 nat over "
                    "baseline logprob, content-specific vs an equal-norm random write); "
                    "|strength|>=1.0 over-injects and degrades coherence (see j5a_swap_result.txt)")
        return {"ok": True, "concept": concept, "token_id": built["token_id"], "layer": resolved_layer,
                "strength": float(strength), "vector": built["vector"], "coef": coef, "note": note}

    # -- persistent-dial surface (mirrors EngineSteer) ----------------------------------------

    def set(self, concept: str, value: float):
        """Mirror EngineSteer.set: persist a strength, lazily (no vector build here)."""
        self.strength[concept] = max(-1.5, min(1.5, float(value)))

    def clear(self):
        self.strength = {}

    def active(self) -> dict:
        return {k: v for k, v in self.strength.items() if v}

    def steer_vector(self, strength: Optional[dict] = None) -> Optional[list]:
        """Sum every ACTIVE concept's coef-scaled unit vector into ONE raw steer vector at
        self.layer -- the SAME return contract as EngineSteer.steer_vector (a plain list[float],
        pre-scaled, or None when nothing is active) -- so a caller can add this into
        EngineSubstrate.chat's kw["steer_vec"] alongside tone dials. A concept whose vector can't
        be built (see compute()) is SKIPPED here, not fatal -- call steer_toward() directly to see
        why any one concept failed."""
        active = {k: v for k, v in (strength if strength is not None else self.strength).items() if v}
        if not active:
            return None
        total = None
        for concept, value in active.items():
            built = self.compute(concept, layer=self.layer)
            if not built.get("ok"):
                continue
            median_norm = self.median_norm or VALIDATED_MEDIAN_RESID_NORM.get(built["layer"])
            if not median_norm:
                continue
            contribution = np.asarray(built["vector"], dtype=np.float32) * (float(value) * float(median_norm))
            total = contribution if total is None else total + contribution
        return total.tolist() if total is not None else None


# ============================================================================== CLI: --selftest / --demo
#
# Mirrors engine/client/cloze_engine.py's own --selftest/--demo split: --selftest is offline (no
# engine, no GPU, safe as this module's default with no flags -- exercised in spirit by
# tests/test_concept_dir.py's fixture-based self-consistency checks, just runnable standalone
# here too). --demo is the LIVE smoke this session's guardrails explicitly DEFER (a GPU experiment
# is running elsewhere; do not boot the engine) -- it is real, runnable code, just never invoked
# by this session or by the pytest suite. It needs TWO things neither exists by default:
#   1. a running cloze-server with a J-lens sidecar loaded (`--jlens <dir>` / CLOZN_JLENS_DIR);
#   2. an unembed export (see BLOCKER_NOTE) -- e.g. for local dev, point --unembed-dir at
#      ../clozn-jlens-work/artifacts (which already has norm_weight.npy/lm_head_weight.npy/
#      unembed_meta.json in the lab's shape this loader expects).
# Once both exist: `python -m clozn.behavior.steering.concept_dir --demo --port 8095
#   --unembed-dir ../clozn-jlens-work/artifacts --concept ocean --strength 0.35`

def _selftest() -> int:
    """Offline self-consistency proof on a synthetic fixture (orthogonal J + orthonormal W_U) --
    no engine, no GPU, no real J-lens/unembed files. See tests/test_concept_dir.py for the same
    check plus the full loader/ConceptSteer suite."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        jdir = os.path.join(tmp, "jlens")
        os.makedirs(jdir)
        d_model = 32
        rng = np.random.default_rng(0)
        J, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
        J = J.astype(np.float32)
        J.astype("<f2").tofile(os.path.join(jdir, "J_layer21.f16"))
        with open(os.path.join(jdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "selftest", "d_model": d_model, "vocab": d_model, "layers": [21],
                      "engine_default_tap_layer": 21}, f)
        udir = os.path.join(tmp, "unembed")
        os.makedirs(udir)
        W, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
        W = W.astype(np.float32)
        np.save(os.path.join(udir, "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
        np.save(os.path.join(udir, "lm_head_weight.npy"), W)
        with open(os.path.join(udir, "unembed_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"rms_norm_eps": 1e-6}, f)

        source = ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
        J_by_layer = source.jacobians()
        unembed_weights = source.unembed()
        ranks = []
        for c in range(d_model):
            vec = dir_c(c, 21, J_by_layer, unembed_weights, scale=1.0)
            logits = read_through_lens(vec, 21, J_by_layer, unembed_weights)
            ranks.append(rank_of(logits, c))
        ok = all(r == 0 for r in ranks)
        print(f"selftest {'OK' if ok else 'FAIL'}: {sum(r == 0 for r in ranks)}/{d_model} tokens "
              f"recovered dir(c) at exact top-1 through their own lens")
        return 0 if ok else 1


def _demo(args) -> int:
    """LIVE smoke -- DEFERRED this session (see the section docstring above). Needs a running
    cloze-server (with --jlens) and a configured unembed export; never invoked automatically."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    import sys as _sys
    _sys.path.insert(0, os.path.join(repo_root, "engine", "client"))
    from cloze_engine import EngineClient  # local import: only the --demo path needs the engine SDK

    ec = EngineClient(host=args.host, port=args.port)
    print(f"server: {ec.health().get('model')}")
    source = ConceptDirSource(jlens_dir=args.jlens_dir, unembed_dir=args.unembed_dir)
    steer = ConceptSteer(ec, source=source, layer=args.layer)
    built = steer.steer_toward(args.concept, args.strength)
    if not built.get("ok"):
        print(f"blocked: {built.get('blocked')} -- {built.get('note')}")
        return 1
    print(f"dir({args.concept!r}) at L{built['layer']}: token_id={built['token_id']} "
         f"coef={built['coef']:.2f}")
    base = ec.complete(args.prompt, max_tokens=args.max_tokens)
    swapped = ec.intervene(args.prompt, vector=built["vector"], coef=built["coef"],
                           layer=built["layer"], max_tokens=args.max_tokens)
    print("baseline:", _text_of(base))
    print("swapped :", _text_of(swapped))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="concept_dir.py -- any-concept dial (dir(c))")
    ap.add_argument("--selftest", action="store_true", help="offline self-consistency proof (default, safe)")
    ap.add_argument("--demo", action="store_true", help="LIVE smoke against a running cloze-server (DEFERRED)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--jlens-dir", default=None, help="default: ~/.clozn/jlens or CLOZN_JLENS_DIR")
    ap.add_argument("--unembed-dir", default=None, help="see BLOCKER_NOTE; default: CLOZN_DIRC_UNEMBED_DIR")
    ap.add_argument("--concept", default="ocean")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args(argv)
    if args.demo:
        return _demo(args)
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
