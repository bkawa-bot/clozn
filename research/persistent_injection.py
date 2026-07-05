"""persistent_injection.py -- Wild Experiment #2 (Wave 1): the persistence PHASE DIAGRAM.

Pre-registration: research/WILD_WAVE1_PREREG.md, "Exp 2 -- the minimal persistent injection". The claim
under test: kv_timetravel.py's <1-turn half-life of a one-shot KV/activation edit (FINDINGS.md Law #3 --
"state is not storage") is UNIVERSAL PHYSICS, not a Qwen quirk, and there is a measurable persistence
phase diagram: what is the SMALLEST intervention that survives past one conversational turn?

ANTECEDENTS REUSED (not reimplemented):
  * kv_timetravel.py -- KVChat (the checkpointable multi-turn chat harness over one DynamicCache),
    resolve_model_path, warmth_score / WARM_MARKERS (the transparent warm-marker count), ntok, and the
    EXACT scripted "discouraging week at work" conversation from its own Phase 3 half-life measurement
    (same setup/followup turns -- so this experiment's curves are directly comparable to the one that
    established Law #3 in the first place, not a fresh unrelated probe).
  * phantom_kv.py -- PhantomKV (k trainable warm-started ghost K/V slots per layer) and train_phantom
    (frozen-backbone KL distillation) reused UNMODIFIED for the "phantom-KV vs raw edit" arm.
  * steering.py -- AXES / SEED_PROMPTS, the exact contrastive-pole recipe (mean(+pole) - mean(-pole) over
    neutral seeds) that both kv_timetravel's KVWarmDirection and this file's ValueSpaceDirection derive
    their injection direction from.
  * counterfactual._coherence -- the mandatory coherence axis (FINDINGS.md Law #6): every generated
    reply is checked for degeneration; an effect that "persists" as word-salad is flagged, not reported
    as a win.

ONE DELIBERATE DEVIATION from kv_timetravel.KVWarmDirection: that class captures its contrastive pairs
with a SYSTEM-role instruction plus a separate user seed. Gemma-2's chat template REJECTS a system role
outright, so ValueSpaceDirection (below) folds the pole instruction into the USER turn instead (same
text, same seeds, same layer -- mirror_bench.py's own convention for this exact constraint). This is why
KVWarmDirection is not imported directly: it would crash on Gemma.

A SECOND, more important fix relative to the antecedent: kv_timetravel.py always loads bf16 (never nf4),
so reading `v_proj.weight.dtype` is a safe way to learn the KV cache's dtype there. This experiment MUST
run nf4 on both models (single 16GB card). For an nf4-quantized Linear4bit layer, `.weight.dtype` is the
PACKED 4-bit STORAGE dtype, not the bf16 COMPUTE dtype the actual K/V cache tensors hold at runtime --
reading it would silently inject a garbage-scaled (or wrong-dtype) push. Fixed here by casting the
injection tensor to the LIVE cache tensor's own `.dtype`/`.device` at the point of injection
(_edit_kv_span), never guessed from a projection weight.

KV GEOMETRY, READ FROM CONFIG, NEVER HARDCODED (kv_geometry()). The pre-reg's shape facts: Qwen2.5-7B =
28 layers, 4 KV heads, head_dim 128 (= hidden 3584 / 28 attn heads -- derivable); Gemma-2-9B = 42 layers,
8 KV heads, head_dim 256 (hidden 3584 / 16 attn heads = 224 != 256 -- head_dim must be READ from
cfg.head_dim, never derived, or Gemma silently gets a wrong-shaped injection tensor). kv_geometry()
prefers cfg.head_dim and only falls back to the derived value if the config doesn't expose it.

THE SWEEP -- 10 cells, see ALL_CELLS below (a persistence phase diagram; one cell = one point in it):
  (a) positions: '1' (the single last token of the target turn) vs 'N' (the whole turn's span) -- dim 1.
  (b) cache: 'K' vs 'V' vs both ('kv' -- the combined cell, and phantom's fair comparison partner) -- dim 2.
  (c) cadence: 'once' (inject after setup, never touch the cache again) vs 'every_turn' (re-apply the
      SAME direction to each new turn's own span, right after that turn completes) -- dim 3. NOTE the
      timing simplification, stated up front: re-injection lands AFTER a turn's reply is generated
      (refreshing the cache for SUBSEQUENT turns only), not before that turn's own generation -- this
      reuses KVChat.generate_turn() unmodified rather than forking its internals for a mid-turn hook.
      Consequence: turn 0's reply is IDENTICAL between 'once' and 'every_turn' for the same (pos, side)
      cell by construction (both are generated from the identical post-setup edited cache); the cadences
      can only diverge from follow-up turn 1 onward.
  (d) mechanism: raw hand-injected direction (8 grid cells + the KV-combined cell) vs PHANTOM -- a
      phantom_kv.PhantomKV ghost slot trained (frozen backbone, KL distillation) to reproduce the warm
      instruction's own real KV cache, then carried through the identical scripted conversation. Compared
      against the 'Npos_KV_once' raw cell (the only raw cell that also touches both caches at N
      positions, inject-once) -- the closest apples-to-apples partner a hand-edit has to a trained slot.

NULLS (both required to clear at turn 0 or the decay curve is meaningless -- the pre-reg's own gate,
enforced in turns_to_noise()): (1) SHUFFLED-DIRECTION, matched norm -- a random permutation of the true
direction's own components (shuffled_like: provably norm-preserving, destroys the direction). (2)
NO-INJECTION baseline -- the identical scripted conversation, untouched. For the phantom arm, the null is
phantom_kv's own RANDOM-init phantom (matched shape, untrained) -- "shuffle the direction" has no
single-vector meaning for a k-slot trained cache, so this is the natural analogue, and it is already what
phantom_kv.py's own E1 null is.

METRIC. warmth_rate = kv_timetravel.warmth_score(reply) / tokens(reply) -- a RATE, not a raw count, so a
longer reply can't win on marker-count alone. Exactly what is counted is kv_timetravel.WARM_MARKERS
substring hits plus '!' count (see that file -- crude on purpose, documented, and eyeballed via the raw
replies kept in every result). turns_to_noise() computes, per cell, the first follow-up turn at which the
TRUE warm-vs-baseline delta falls to/below the noise floor set by the shuffled-null's OWN delta (mean
|d_null| over the measured turns) -- the operationalized "half-life". `gate_passed=False` means turn 0
itself never cleared the null; that cell's curve is still reported, but flagged, never silently trusted.
Every reply is ALSO run through counterfactual._coherence; a cell whose "persisting" warm branch is
flagged degenerate gets an explicit coherence_caution string in its result -- word-salad is not
persistence.

CAVEATS (stated loud, per house ethos): one seed, greedy decoding, two model families -- no claim beyond
"holds/breaks on these two"; mid layer = n_layers // 2 uniformly (steering.py's own default) -- NOT
checked against Gemma-2's alternating local/global attention pattern (sliding_window / sliding_window_
pattern are surfaced raw in kv_geometry()'s output if the config exposes them, but never interpreted or
branched on here -- a real open question flagged, not answered); the phantom arm uses a FULL-RECOMPUTE
multi-turn runner (re-feeds the whole transcript every turn from a fresh phantom.build_cache()) rather
than KVChat's incremental prefill, because a phantom's k invisible slots break KVChat's
n_tok-counts-real-tokens bookkeeping -- correct, but O(turns^2) not O(turns); PhantomKV's own dtype
inference (next(model.parameters()).dtype) is reused as-is, trusted from phantom_kv.py's own separately
validated runs, not re-derived here.

GPU: nf4 on a single 16GB card, one model per process (mirror_bench.py's convention). Gemma-2's chat
template rejects a system role -- every prompt in this file (ValueSpaceDirection's pole instructions, the
scripted conversation, the phantom's teacher-cache instruction) rides in the USER turn only, never system.

Run (repo root, CUDA venv), one model per process, then compare:
    PY=C:/Users/brigi/src/cloze/.venv/Scripts/python.exe
    $PY research/persistent_injection.py --model Qwen/Qwen2.5-7B-Instruct \
        --out research/runs/persistent_injection_qwen7b.json
    $PY research/persistent_injection.py --model google/gemma-2-9b-it \
        --out research/runs/persistent_injection_gemma9b.json
    $PY research/persistent_injection.py --compare research/runs/persistent_injection_qwen7b.json \
        research/runs/persistent_injection_gemma9b.json
Smoke first (2 cells -- one raw, one phantom, so BOTH mechanisms' wiring is proven; 2 follow-up turns):
    $PY research/persistent_injection.py --model Qwen/Qwen2.5-7B-Instruct --smoke \
        --out research/runs/persistent_injection_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")   # WinError 1314 workaround on this PC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the pre-reg's named antecedents -- imported, never re-derived (see module docstring for the two
# deliberate deviations: Gemma's no-system-role template, and the nf4 dtype fix).
from kv_timetravel import KVChat, resolve_model_path, warmth_score, ntok as kv_ntok, PIN as KV_PIN
from phantom_kv import (PhantomKV, train_phantom, TRAIN_PROMPTS as PHANTOM_TRAIN_PROMPTS,
                        _teacher_greedy, _teacher_target_logits)
from steering import AXES, SEED_PROMPTS
from counterfactual import _coherence

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- experiment-wide defaults (all overridable via CLI) ----------------------------------------------
DEFAULT_MAX_NEW = 96           # matches kv_timetravel.py phase3's own generation budget
DEFAULT_TURNS = 6              # follow-up turns per branch (== len(FOLLOWUPS), the antecedent's own count)
DEFAULT_DOSE = 4.0             # matches kv_timetravel.py phase3's default warm dose
DEFAULT_PHANTOM_K = 8
DEFAULT_PHANTOM_STEPS = 100
DEFAULT_PHANTOM_LR = 0.02
SHUFFLE_SEED = 1234            # deterministic shuffled-direction null (reproducible across runs)
_PHANTOM_TRAIN_SPAN = 40       # teacher tokens per training prompt (matches phantom_kv.py's CFG['answer_span'])

# the scripted conversation -- VERBATIM from kv_timetravel.py's own Phase 3 half-life measurement, so this
# experiment's curves are directly comparable to the one that established Law #3 in the first place.
SETUP_USERS = [
    "I've had a really discouraging week at work and I'm doubting myself.",
    "Do you have any advice for me?",
]
FOLLOWUPS = [
    "What should I focus on tomorrow morning?",
    "How do I explain my week to my manager?",
    "Is it worth asking a colleague for help?",
    "What's a small win I could aim for this week?",
    "How do I stop replaying my mistakes at night?",
    "Any final thought before I sign off?",
]


def _scripted_conversation(n_turns: int) -> tuple[list[str], list[str]]:
    """SETUP_USERS unchanged; FOLLOWUPS clamped to the first `n_turns` (>=1, <= len(FOLLOWUPS))."""
    n = max(1, min(int(n_turns), len(FOLLOWUPS)))
    return SETUP_USERS, FOLLOWUPS[:n]


# ---- nf4 auto-detect -----------------------------------------------------------------------------------
# mirror_bench.py's exact convention, MIRRORED rather than imported so this file's only research-module
# dependencies are the ones the pre-reg names as Exp 2's antecedents (kv_timetravel, phantom_kv, steering,
# counterfactual) -- not coupled to a sibling file that may be mid-edit elsewhere in this session.
_SMALL = ("0.5b", "1.5b", "-1b", "1b-", "2b", "3b", "-1.7b")


def wants_four_bit(name: str, override: str) -> bool:
    if override == "yes":
        return True
    if override == "no":
        return False
    return not any(s in name.lower() for s in _SMALL)


# ---- KV geometry, read from config, never hardcoded ----------------------------------------------------
def kv_geometry(cfg) -> dict:
    """KV cache geometry read LIVE from the model config -- never hardcoded (the pre-reg's exact warning:
    Qwen2.5-7B = 28L/4kv/128hd, Gemma-2-9B = 42L/8kv/256hd, and hidden_size MATCHES (3584) while KV
    geometry does not). head_dim is NOT assumed to be hidden_size // num_attention_heads -- that is WRONG
    for Gemma-2 (3584/16=224 != the real 256) -- cfg.head_dim is read directly when the config exposes it
    (both Qwen2Config and Gemma2Config do) and only derived as a fallback."""
    n_layers = int(cfg.num_hidden_layers)
    n_kv_heads = int(cfg.num_key_value_heads)
    n_attn_heads = int(cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None)
    head_dim = int(head_dim) if head_dim else int(cfg.hidden_size // n_attn_heads)
    geo = {"n_layers": n_layers, "n_kv_heads": n_kv_heads, "n_attn_heads": n_attn_heads,
           "head_dim": head_dim, "hidden_size": int(cfg.hidden_size)}
    # Gemma-2's alternating local/global attention -- surfaced RAW if the config exposes it, never
    # interpreted/branched on (which specific layers are local vs global is an HF-internal convention this
    # rig does not assert without being able to check it against the pinned transformers build -- a named,
    # open caveat, not a guess).
    for k in ("sliding_window", "sliding_window_pattern"):
        v = getattr(cfg, k, None)
        if v is not None:
            geo[k] = v
    return geo


# ---- warmth metric --------------------------------------------------------------------------------------
def warmth_rate(score: int, n_tokens: int) -> float:
    """kv_timetravel.warmth_score(reply), normalized by reply length -- a RATE, not a raw count, so a long
    reply can't win on marker-count alone. `score` is warmth_score's own output (WARM_MARKERS substring
    hits + '!' count); `n_tokens` is kv_timetravel.ntok's own count. Documented exactly, no hidden math."""
    return round(score / max(1, int(n_tokens)), 4)


# ---- the shuffled-direction null -------------------------------------------------------------------------
def shuffled_like(vec: torch.Tensor, seed: int) -> torch.Tensor:
    """The 'same-size push, wrong direction' null: a random permutation of vec's OWN components. A
    permutation preserves the L2 norm EXACTLY (same multiset of squared components, reordered) and the
    per-dimension marginal distribution, but destroys the direction -- if the warmth effect at turn 0
    survives THIS control (in addition to the no-injection baseline), it is not just 'any big vector in
    the value cache does something warm-ish'. Deterministic given `seed` (reproducible across runs)."""
    g = torch.Generator(device=vec.device).manual_seed(int(seed))
    perm = torch.randperm(vec.numel(), generator=g, device=vec.device)
    return vec[perm]


# ---- injection tensor + the in-place cache edit ------------------------------------------------------
def _kv_add(unit_vec: torch.Tensor, val_norm: float, dose: float, n_kv_heads: int, head_dim: int) -> torch.Tensor:
    """The additive tensor for one cache position: dose * val_norm * unit_dir, reshaped to
    [n_kv_heads, head_dim]. Dtype/device conversion happens at the INJECTION site (_edit_kv_span), read
    off the LIVE cache tensor itself -- not guessed from a projection weight's dtype, which for an
    nf4-quantized layer is the PACKED 4-bit storage dtype, not the bf16 compute dtype the cache actually
    holds (see the module docstring's nf4-dtype-bug note)."""
    return (dose * val_norm) * unit_vec.reshape(n_kv_heads, head_dim)


def _edit_kv_span(cache, layer: int, side: str, span: tuple[int, int], add_bh: torch.Tensor):
    """Add `add_bh` ([n_kv_heads, head_dim]) into cache K (side='k') or V (side='v') at `layer`, positions
    [span[0]:span[1]). In-place; broadcasts over the span. Generalizes kv_timetravel._edit_value_span to
    either cache, and casts `add_bh` to the LIVE tensor's own dtype/device (nf4-safe -- see docstring)."""
    s0, s1 = span
    t = cache.layers[layer].keys if side == "k" else cache.layers[layer].values
    add = add_bh.to(device=t.device, dtype=t.dtype)
    t[:, :, s0:s1, :] = t[:, :, s0:s1, :] + add[None, :, None, :]


def _span_from_lengths(before_len: int, incl_len: int, mode: str) -> tuple[int, int]:
    """Pure position-index math (no tokenizer): 'N' = the whole final-user-turn span [before_len,
    incl_len); '1' = just its LAST position [incl_len-1, incl_len) -- the smallest possible edit, one past
    position, per the pre-reg's dimension (1)."""
    if mode == "1":
        return (incl_len - 1, incl_len)
    return (before_len, incl_len)


def _final_user_span(chat, mode: str) -> tuple[int, int]:
    """Token-position span of `chat`'s CURRENT final user turn, in `mode` ('1' or 'N') -- same rendering
    idiom as kv_timetravel.py's phase2/phase3 (render up to but excluding the final user turn, then
    including it). Must be called right after a turn completes (chat.messages ends [..., user, assistant])."""
    before = chat._render(chat.messages[:-2], add_gen=False) if len(chat.messages) >= 2 else []
    incl = chat._render(chat.messages[:-1], add_gen=False)
    return _span_from_lengths(len(before), len(incl), mode)


# ---- turns-to-noise: the operationalized half-life -----------------------------------------------------
def turns_to_noise(d_true: list[float], d_null: list[float]) -> dict:
    """The half-life scalar: the first follow-up turn (0-indexed) at which the TRUE warm-vs-baseline delta
    falls to/below the noise floor set by the shuffled-direction null's OWN delta (same turns, same
    branch structure, only the direction destroyed). noise_floor = mean(|d_null|) over ALL measured turns
    (a single scalar band, not per-turn, so one noisy null turn can't manufacture an early "decayed"
    verdict). gate_passed = |d_true[0]| > noise_floor -- per the pre-reg ("the warmth effect must clear
    both nulls at turn 0 for the decay curve to mean anything"); when False the curve is still returned,
    but flagged, not silently trusted. turns_to_noise=None means "PERSISTS through the full measured
    window" -- a real, reportable outcome, not a missing value."""
    if not d_true:
        return {"turns_to_noise": None, "gate_passed": False, "noise_floor": 0.0, "turn0_effect": None,
                "decay_note": "no data"}
    noise_floor = round((sum(abs(x) for x in d_null) / len(d_null)) if d_null else 0.0, 4)
    turn0 = d_true[0]
    gate_passed = abs(turn0) > noise_floor
    first = None
    for i, v in enumerate(d_true):
        if abs(v) <= noise_floor:
            first = i
            break
    if not gate_passed:
        decay_note = "turn 0 never cleared the shuffled-direction null -- GATED, curve not meaningful"
    elif first is None:
        decay_note = "PERSISTS through the full measured window (no decay to noise)"
    else:
        decay_note = f"decays to noise by follow-up turn {first}"
    return {"turns_to_noise": first, "gate_passed": gate_passed, "noise_floor": noise_floor,
            "turn0_effect": round(turn0, 4), "decay_note": decay_note}


# ---- the sweep cells: a 2x2x2 raw-edit grid + the KV-combined cell + phantom ----------------------------
def _make_grid_cells() -> list[dict]:
    cells = []
    for pos in ("1", "N"):
        for side in ("k", "v"):
            for cadence in ("once", "every_turn"):
                cid = f"{pos}pos_{side.upper()}_{cadence}"
                cells.append({"id": cid, "pos": pos, "side": side, "cadence": cadence, "mechanism": "raw"})
    return cells


GRID_CELLS = _make_grid_cells()                        # 8: the 2x2x2 factorial (positions x cache x cadence)
EXTRA_CELLS = [
    {"id": "Npos_KV_once", "pos": "N", "side": "kv", "cadence": "once", "mechanism": "raw"},
    {"id": "phantom", "pos": "N", "side": "kv", "cadence": "once", "mechanism": "phantom"},
]
ALL_CELLS = GRID_CELLS + EXTRA_CELLS                    # 10 cells total -- the persistence phase diagram
SMOKE_CELL_IDS = ["Npos_V_once", "phantom"]             # one raw cell (closest to kv_timetravel's own
                                                         # proven phase-3 setup) + phantom -- both
                                                         # mechanisms' wiring proven cheaply


def _resolve_cells(cells_arg: str | None, smoke: bool) -> list[dict]:
    by_id = {c["id"]: c for c in ALL_CELLS}
    if smoke:
        ids = SMOKE_CELL_IDS
    elif not cells_arg or cells_arg == "all":
        ids = list(by_id)
    else:
        ids = [c.strip() for c in cells_arg.split(",") if c.strip()]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown cell id(s): {missing}. valid ids: {sorted(by_id)}")
    return [by_id[i] for i in ids]


# ---- the value-space contrastive direction (K or V, Gemma-safe) -----------------------------------------
class ValueSpaceDirection:
    """Warm direction in a chosen KV projection's OUTPUT space (k_proj or v_proj) -- the same contrastive
    recipe as steering.py's residual axis and kv_timetravel.KVWarmDirection's value-space analogue: mean
    (+pole) - mean(-pole) over steering.SEED_PROMPTS, captured via a forward hook at the target layer's
    k_proj / v_proj output. UNLIKE KVWarmDirection, the pole instruction is folded into the USER turn
    (never a system role) -- Gemma-2's chat template rejects system, and this makes the SAME code path
    work for both families rather than branching per model."""

    def __init__(self, model, tok, layer: int, which: str = "v"):
        assert which in ("k", "v"), which
        self.model, self.tok, self.layer, self.which = model, tok, layer, which
        self.vec: torch.Tensor | None = None      # unit direction in this projection's output space
        self.val_norm = 0.0                        # typical per-position projection-output norm (for dosing)

    @torch.no_grad()
    def _proj_last(self, user_text: str) -> torch.Tensor:
        """The chosen projection's output at the LAST prompt token, flattened to [n_kv_heads*head_dim]."""
        ids = self.tok.apply_chat_template([{"role": "user", "content": user_text}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        proj = (self.model.model.layers[self.layer].self_attn.k_proj if self.which == "k"
                else self.model.model.layers[self.layer].self_attn.v_proj)
        grab = {}
        h = proj.register_forward_hook(lambda m, i, o: grab.__setitem__("x", o.detach()))
        try:
            self.model(ids, use_cache=False)
        finally:
            h.remove()
        return grab["x"][0, -1].float()

    @torch.no_grad()
    def compute(self, seeds=SEED_PROMPTS, axis: str = "warm") -> dict:
        ax = AXES[axis]
        pv = [self._proj_last(f"{ax['pos']}\n\n{s}") for s in seeds]
        nv = [self._proj_last(f"{ax['neg']}\n\n{s}") for s in seeds]
        pos, neg = torch.stack(pv).mean(0), torch.stack(nv).mean(0)
        d = pos - neg
        self.vec = d / (d.norm() + 1e-8)
        self.val_norm = float(torch.stack(pv + nv).norm(dim=-1).mean())
        return {"which": self.which, "axis": axis, "raw_norm": round(float(d.norm()), 3),
                "val_norm": round(self.val_norm, 2), "dim": int(self.vec.numel())}


# ---- per-reply scoring: warmth rate + the mandatory coherence axis ---------------------------------------
def _score_reply(tok, text: str) -> dict:
    n = kv_ntok(tok, text)
    s = warmth_score(text)
    coh = _coherence(text)
    return {"reply": text, "tokens": n, "warmth": s, "warmth_rate": warmth_rate(s, n),
            "degenerate": coh["degenerate"], "degenerate_reason": coh["reason"]}


def _assemble_cell_result(cell: dict, warm_rows: list[dict], null_rows: list[dict],
                          base_rows: list[dict]) -> dict:
    """Combine three matched per-turn row lists (warm-injected / null-injected / no-injection baseline --
    same scripted turns, same model, only the cache edit differs) into the cell's decay curve + the
    turns-to-noise verdict. d_true/d_null are WARMTH-RATE deltas vs the shared baseline at each follow-up
    turn -- the causal effect of the injection, turn by turn."""
    n = min(len(warm_rows), len(null_rows), len(base_rows))
    d_true = [round(warm_rows[i]["warmth_rate"] - base_rows[i]["warmth_rate"], 4) for i in range(n)]
    d_null = [round(null_rows[i]["warmth_rate"] - base_rows[i]["warmth_rate"], 4) for i in range(n)]
    ttn = turns_to_noise(d_true, d_null)
    degenerate_turns = [i for i in range(n) if warm_rows[i]["degenerate"]]
    caution = (f"turns {degenerate_turns} degenerate in the WARM branch -- an effect 'persisting' there "
               f"is word-salad, not persistence" if degenerate_turns else "")
    return {"cell": cell, "d_true": d_true, "d_null": d_null, **ttn,
            "degenerate_turns_warm": degenerate_turns, "coherence_caution": caution,
            "warm": warm_rows, "null": null_rows, "baseline": base_rows}


# ---- raw-edit branch runner (shared by every grid + KV-combined cell) ------------------------------------
@torch.no_grad()
def _raw_branch(model, tok, layer: int, sides: tuple, pos_mode: str, cadence: str, dose: float, kind: str,
                directions: dict, geo: dict, n_turns: int) -> list[dict]:
    """One full scripted conversation (setup + n_turns follow-ups) on a fresh KVChat, optionally with a
    value-space edit applied to `sides` ('k'/'v', possibly both) over `pos_mode` ('1'/'N') positions,
    injected ONCE right after setup (cadence='once') or refreshed after every follow-up turn
    (cadence='every_turn' -- see the module docstring for the exact once/every-turn timing this
    simplification implies). kind in {'warm','shuffled','none'}; 'none' ignores sides/pos_mode/cadence/
    dose entirely -- the shared no-injection baseline. Deterministic greedy decode, so callers should
    compute the 'none' branch ONCE per model and reuse it across every raw cell rather than re-running
    the identical conversation per cell."""
    setup_users, followups = _scripted_conversation(n_turns)
    chat = KVChat(model, tok)
    for u in setup_users:
        chat.generate_turn(u, max_new=DEFAULT_MAX_NEW)

    add_by_side = {}
    if kind != "none":
        n_kv, hd = geo["n_kv_heads"], geo["head_dim"]
        for s in sides:
            base_vec = directions[s].vec
            vec = base_vec if kind == "warm" else shuffled_like(base_vec, seed=SHUFFLE_SEED)
            add_by_side[s] = _kv_add(vec, directions[s].val_norm, dose, n_kv, hd)
        span = _final_user_span(chat, pos_mode)
        for s in sides:
            _edit_kv_span(chat.cache, layer, s, span, add_by_side[s])

    rows = []
    for u in followups:
        chat.generate_turn(u, max_new=DEFAULT_MAX_NEW)
        if kind != "none" and cadence == "every_turn":
            span = _final_user_span(chat, pos_mode)
            for s in sides:
                _edit_kv_span(chat.cache, layer, s, span, add_by_side[s])
        reply = chat.checkpoints[-1]["reply"]
        rows.append(_score_reply(tok, reply))
    return rows


def run_raw_cell(model, tok, layer, geo, directions, cell, dose, n_turns, base_rows) -> dict:
    sides = ("k", "v") if cell["side"] == "kv" else (cell["side"],)
    warm_rows = _raw_branch(model, tok, layer, sides, cell["pos"], cell["cadence"], dose, "warm",
                            directions, geo, n_turns)
    shuf_rows = _raw_branch(model, tok, layer, sides, cell["pos"], cell["cadence"], dose, "shuffled",
                            directions, geo, n_turns)
    return _assemble_cell_result(cell, warm_rows, shuf_rows, base_rows)


# ---- phantom-KV branch runner ---------------------------------------------------------------------------
@torch.no_grad()
def _warm_teacher_cache(model, tok, pole_text: str):
    """The pole instruction's OWN real KV cache -- the phantom's warm-start + distillation teacher, the
    same recipe as phantom_kv.teacher_block_cache but for OUR text (a steering.AXES pole instruction
    folded into a user turn -- Gemma has no system role) instead of phantom_kv's own CARDS block."""
    ids = tok.apply_chat_template([{"role": "user", "content": pole_text}],
                                  add_generation_prompt=True, tokenize=True)
    cache = DynamicCache()
    emb = model.get_input_embeddings()
    e = emb(torch.tensor([ids], device=DEV))
    att = torch.ones(1, len(ids), device=DEV, dtype=torch.long)
    model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
          cache_position=torch.arange(len(ids), device=DEV), use_cache=True)
    return cache, len(ids)


@torch.no_grad()
def _build_phantom_train_examples(model, tok, teacher_cache, block_len, prompts, span):
    """(prompt_ids, teacher_greedy_target_ids, teacher_target_logits) triples -- phantom_kv.train_phantom's
    own input shape, built with phantom_kv's own teacher helpers against OUR teacher cache."""
    examples = []
    for pr in prompts:
        pid = tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True, tokenize=True)
        tgt = _teacher_greedy(model, tok, pid, teacher_cache, block_len, span)
        if not tgt:
            continue
        tl = _teacher_target_logits(model, tok, pid, tgt, teacher_cache, block_len).detach()
        examples.append((pid, tgt, tl))
    return examples


@torch.no_grad()
def _phantom_conversation(model, tok, phantom, setup_users, followups, max_new=DEFAULT_MAX_NEW) -> list[dict]:
    """Multi-turn greedy chat with `phantom` (or None) prepended to the KV cache -- a FULL-RECOMPUTE runner
    (re-feeds the whole transcript every turn from a fresh phantom.build_cache()) rather than KVChat's
    incremental delta-prefill, because KVChat's n_tok bookkeeping assumes every cached position
    corresponds to a real rendered token; a phantom prefix breaks that assumption (k invisible slots
    occupy cache positions with no token of their own). Slower (O(turns^2), not O(turns)) but exactly
    correct, and this rig only runs a handful of short turns. Mirrors phantom_kv.gen_with_phantom's
    single-turn decode loop, extended to a running multi-turn transcript. Only follow-up turns are scored
    (setup turns still run, to build the real transcript/cache state, matching _raw_branch's convention)."""
    k = phantom.k if phantom is not None else 0
    messages: list[dict] = []
    rows = []
    all_turns = list(setup_users) + list(followups)
    n_setup = len(setup_users)
    emb = model.get_input_embeddings()
    eos = tok.eos_token_id
    for i, u in enumerate(all_turns):
        messages.append({"role": "user", "content": u})
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        cache = phantom.build_cache() if phantom is not None else DynamicCache()
        e = emb(torch.tensor([ids], device=DEV))
        att = torch.ones(1, k + len(ids), device=DEV, dtype=torch.long)
        cpos = torch.arange(k, k + len(ids), device=DEV)
        out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache, cache_position=cpos,
                   use_cache=True)
        gen = []
        pos = k + len(ids)
        for _ in range(max_new):
            nxt = int(out.logits[0, -1].argmax())
            if nxt == eos:
                break
            gen.append(nxt)
            e = emb(torch.tensor([[nxt]], device=DEV))
            att = torch.ones(1, pos + 1, device=DEV, dtype=torch.long)
            out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
                       cache_position=torch.tensor([pos], device=DEV), use_cache=True)
            pos += 1
        reply = tok.decode(gen, skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": reply})
        if i >= n_setup:
            rows.append(_score_reply(tok, reply))
    return rows


def run_phantom_cell(model, tok, layer, cell, n_turns, axis, phantom_k, phantom_steps, phantom_lr,
                     smoke) -> dict:
    """The pre-reg's 4th arm: does a TRAINED ghost-slot KV entry persist longer than a hand-injected
    direction? Trains a phantom_kv.PhantomKV (k warm-started + distilled slots per layer, BOTH K and V --
    why this cell's raw comparison partner is 'Npos_KV_once') to reproduce the effect of the chosen axis's
    positive-pole instruction's own real KV cache, then runs the identical scripted conversation with that
    trained phantom prepended, vs phantom_kv's own RANDOM-init null (matched shape, untrained), vs a true
    no-phantom baseline."""
    pole_text = AXES[axis]["pos"]
    teacher_cache, block_len = _warm_teacher_cache(model, tok, pole_text)
    prompts = PHANTOM_TRAIN_PROMPTS[:2] if smoke else PHANTOM_TRAIN_PROMPTS
    span = 10 if smoke else _PHANTOM_TRAIN_SPAN
    train_examples = _build_phantom_train_examples(model, tok, teacher_cache, block_len, prompts, span)
    print(f"  [phantom] teacher block_len={block_len}, {len(train_examples)} training examples "
          f"(k={phantom_k})", flush=True)

    ph = PhantomKV(model, phantom_k, teacher_cache=teacher_cache, random_init=False)
    tr = train_phantom(model, tok, ph, train_examples, steps=phantom_steps, lr=phantom_lr,
                       max_norm=60.0, log_prefix="  [phantom-train] ")
    print(f"  [phantom] trained: {tr}", flush=True)
    rnd = PhantomKV(model, phantom_k, teacher_cache=teacher_cache, random_init=True)

    setup_users, followups = _scripted_conversation(n_turns)
    warm_rows = _phantom_conversation(model, tok, ph, setup_users, followups)
    null_rows = _phantom_conversation(model, tok, rnd, setup_users, followups)
    base_rows = _phantom_conversation(model, tok, None, setup_users, followups)

    result = _assemble_cell_result(cell, warm_rows, null_rows, base_rows)
    result["phantom_k"] = phantom_k
    result["train"] = tr
    result["teacher_block_len"] = block_len
    result["note"] = ("null = phantom_kv's RANDOM-init phantom (matched shape, untrained) -- the "
                      "shuffled-direction analogue for a trained multi-vector slot; baseline = no "
                      "phantom at all (k=0, plain chat).")
    return result


# ---- model load -----------------------------------------------------------------------------------------
def load_model(model_name: str, four_bit: bool):
    path = resolve_model_path(model_name)
    print(f"[load] {model_name} ({'nf4' if four_bit else 'bf16'}, {DEV}) from {path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if four_bit and DEV == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---- orchestration ---------------------------------------------------------------------------------------
def run(model_name, out_path, four_bit_override="auto", smoke=False, cells_arg="all", turns=DEFAULT_TURNS,
        dose=DEFAULT_DOSE, layer_override=None, axis="warm", phantom_k=DEFAULT_PHANTOM_K,
        phantom_steps=DEFAULT_PHANTOM_STEPS, phantom_lr=DEFAULT_PHANTOM_LR):
    four_bit = wants_four_bit(model_name, four_bit_override)
    n_turns = 2 if smoke else max(1, turns)
    cells = _resolve_cells(cells_arg, smoke)

    model, tok = load_model(model_name, four_bit)
    geo = kv_geometry(model.config)
    layer = layer_override if layer_override is not None else geo["n_layers"] // 2
    print(f"[geometry] {geo} | layer={layer} axis={axis}{' [SMOKE]' if smoke else ''}", flush=True)

    res = {"model": model_name, "four_bit": four_bit, "smoke": smoke, "geometry": geo, "layer": layer,
           "axis": axis, "dose": dose, "n_turns": n_turns, "cells_run": [c["id"] for c in cells],
           "pin": KV_PIN, "direction_compute": {}, "cells": {}}
    _save(res, out_path)

    sides_needed = set()
    for c in cells:
        if c["mechanism"] == "raw":
            sides_needed |= ({"k", "v"} if c["side"] == "kv" else {c["side"]})

    seeds = SEED_PROMPTS[:4] if smoke else SEED_PROMPTS
    directions = {}
    for side in sorted(sides_needed):
        d = ValueSpaceDirection(model, tok, layer, which=side)
        info = d.compute(seeds=seeds, axis=axis)
        directions[side] = d
        res["direction_compute"][side] = info
        print(f"[direction/{side}] {info}", flush=True)
    if sides_needed:
        _save(res, out_path)

    base_rows = None
    if any(c["mechanism"] == "raw" for c in cells):
        print("[baseline] shared no-injection conversation (raw mechanism, reused across every raw cell "
              "-- deterministic greedy decode, so computing it once is exact, not an approximation) ...",
              flush=True)
        base_rows = _raw_branch(model, tok, layer, (), "N", "once", 0.0, "none", directions, geo, n_turns)
        res["raw_baseline"] = base_rows
        _save(res, out_path)

    for cell in cells:
        print(f"\n=== CELL {cell['id']} ===", flush=True)
        t0 = time.time()
        if cell["mechanism"] == "phantom":
            row = run_phantom_cell(model, tok, layer, cell, n_turns, axis,
                                   phantom_k=(4 if smoke else phantom_k),
                                   phantom_steps=(8 if smoke else phantom_steps),
                                   phantom_lr=phantom_lr, smoke=smoke)
        else:
            row = run_raw_cell(model, tok, layer, geo, directions, cell, dose, n_turns, base_rows)
        row["seconds"] = round(time.time() - t0, 1)
        res["cells"][cell["id"]] = row
        _print_cell_line(cell, row)
        _save(res, out_path)

    _summary(res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _save(res: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def _print_cell_line(cell: dict, row: dict):
    gate = "OK" if row.get("gate_passed") else "GATE-FAIL"
    ttn = row.get("turns_to_noise")
    ttn_s = "persists" if ttn is None else f"turn {ttn}"
    cc = " *COHERENCE-FLAG*" if row.get("coherence_caution") else ""
    print(f"  [{cell['id']}] gate={gate} turn0_effect={row.get('turn0_effect')} "
          f"noise_floor={row.get('noise_floor')} decay={ttn_s}{cc} ({row.get('seconds', '?')}s)", flush=True)


def _summary(res: dict):
    print("\n" + "=" * 88, flush=True)
    print(f"PERSISTENT INJECTION -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'}) "
          f"layer={res['layer']}", flush=True)
    print(f"{'cell':16} {'gate':9} {'turn0_d':9} {'noise_flr':10} {'decay':14} coherence", flush=True)
    for cid, row in res["cells"].items():
        ttn = row.get("turns_to_noise")
        ttn_s = "persists" if ttn is None else str(ttn)
        gate = "PASS" if row.get("gate_passed") else "FAIL"
        cc = row.get("coherence_caution") or "-"
        print(f"{cid:16} {gate:9} {str(row.get('turn0_effect')):9} {str(row.get('noise_floor')):10} "
              f"{ttn_s:14} {cc[:40]}", flush=True)
    print("\nLaw #3 (state is not storage) predicts EVERY 'once'-cadence raw cell decays to noise within a "
          "couple of turns, and only 'every_turn' re-injection (or a trained phantom slot) holds longer -- "
          "read the table above against that prediction.", flush=True)


def compare(paths: list[str]):
    """Cross-family phase-diagram table from >=2 per-model JSONs. Pure read + print, no torch call made."""
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    names = [r["model"].split("/")[-1] for r in runs]
    all_cells = sorted({cid for r in runs for cid in r.get("cells", {})})
    print("\n" + "=" * 88)
    print("CROSS-FAMILY PERSISTENCE PHASE DIAGRAM -- turns-to-noise per cell ('persists' = never decayed)")
    print(f"{'cell':16} " + " ".join(f"{n[:24]:26}" for n in names))
    for cid in all_cells:
        row_strs = []
        for r in runs:
            row = r.get("cells", {}).get(cid)
            if row is None:
                row_strs.append("-")
                continue
            ttn = row.get("turns_to_noise")
            ttn_s = "persists" if ttn is None else str(ttn)
            gate = "" if row.get("gate_passed") else "*GATEFAIL*"
            cc = "*DEGEN*" if row.get("coherence_caution") else ""
            row_strs.append(f"{ttn_s}{gate}{cc}")
        print(f"{cid:16} " + " ".join(f"{c:26}" for c in row_strs))
    print("\ngeometry (read from config, never hardcoded):")
    for r, n in zip(runs, names):
        print(f"  {n}: {r.get('geometry')}")
    print("\nUniversal (Law #3 holds cross-family) iff every 'once'-cadence raw cell decays fast on BOTH "
          "families and only 'every_turn'/phantom cells persist -- Qwen-shaped if that split breaks on "
          "Gemma instead.")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Exp 2 -- persistence phase diagram (research/WILD_WAVE1_PREREG.md)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="research/runs/persistent_injection.json")
    ap.add_argument("--four-bit", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="2 cells (one raw + phantom), 2 follow-up turns -- prove the wiring cheap")
    ap.add_argument("--cells", default="all", help="comma-separated cell ids, or 'all' (default)")
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="follow-up turns per branch")
    ap.add_argument("--dose", type=float, default=DEFAULT_DOSE, help="injection dose (raw-edit cells only)")
    ap.add_argument("--layer", type=int, default=None, help="override the mid layer (default: n_layers // 2)")
    ap.add_argument("--axis", default="warm", help="steering.AXES key to inject (default: warm)")
    ap.add_argument("--phantom-k", type=int, default=DEFAULT_PHANTOM_K)
    ap.add_argument("--phantom-steps", type=int, default=DEFAULT_PHANTOM_STEPS)
    ap.add_argument("--phantom-lr", type=float, default=DEFAULT_PHANTOM_LR)
    ap.add_argument("--compare", nargs="+", metavar="RUN.json",
                    help="print the cross-family phase-diagram table from >=2 run JSONs")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run(args.model, args.out, four_bit_override=args.four_bit, smoke=args.smoke, cells_arg=args.cells,
            turns=args.turns, dose=args.dose, layer_override=args.layer, axis=args.axis,
            phantom_k=args.phantom_k, phantom_steps=args.phantom_steps, phantom_lr=args.phantom_lr)
