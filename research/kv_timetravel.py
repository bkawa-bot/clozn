"""kv_timetravel.py -- the KV cache as CHECKPOINTABLE, REWINDABLE, EDITABLE state.

A transformer's past_key_values is usually a throwaway speed optimization: cache the keys/values so you
don't recompute the prefix each step. This experiment treats it instead as first-class, addressable STATE
-- something you can snapshot per conversational turn, rewind to an earlier turn, branch into an alternate
future without re-prefilling the shared past, and even reach in and EDIT (surgery on the cached tensors,
then let the model continue as if it had thought that). If the KV cache is real, inspectable state (the
studio thesis: a legible interior you can touch), these operations should all work -- and be measurable.

Model: Qwen2.5-1.5B-Instruct, bf16, greedy. HF transformers Cache API (DynamicCache), PINNED to the
version below -- the cache internals are not a stable public contract, so a pin is load-bearing.
    transformers 4.57.6, torch 2.11.0+cu128.
Qwen2.5-1.5B shape facts that drive the design (probed, not assumed): 28 layers, hidden 1536, 12 attention
heads x head_dim 128, but only **2 KV heads** (heavy GQA). So a per-layer KV tensor is [1, 2, seq, 128] =
256 feature dims, NOT the 1536-dim residual stream. This matters for Phase 2 (see there).

THREE PHASES (each pre-registered below; each writes a checkpoint JSON):

(1) CHECKPOINT / REWIND / BRANCH -- a multi-turn chat harness that snapshots past_key_values per turn
    (offloaded to CPU), can rewind to turn t, splice in an ALTERNATE user message, and continue WITHOUT
    re-prefilling the shared history. Receipts:
      (a) DETERMINISM: branch-from-checkpoint greedy output == fresh full-recompute of the identical
          transcript. If not bit-identical, we MEASURE the divergence (first-diff token, agreement rate)
          and explain it honestly rather than hand-wave.
      (b) SAVINGS: wall-clock and tokens-not-reprefilled for a branch at turn depths 2 / 5 / 10.

(2) STATE SURGERY -- branch with an EDIT: add a steering direction into the cached VALUE tensors over turn
    t's token span, then continue. Measure the continuation vs the unedited branch (warmth-marker + length
    scorers). Plus an IDENTITY-edit control: adding a zero vector must reproduce the unedited branch
    BYTE-FOR-BYTE (proves the edit path itself is sound and only the direction moved anything).
      Direction source: research/steering.py's SteeringControl warm axis and its mid layer (L14). BUT the
      value cache lives in 256-dim per-KV-head space, not the 1536-dim residual stream, so a residual
      direction cannot be added into it directly. We therefore derive the warm direction IN VALUE SPACE:
      the exact same contrastive recipe (mean warm-pole minus mean cold-pole over SteeringControl's own
      SEED_PROMPTS), but captured at the target layer's v_proj OUTPUT via a forward hook. Honest analogue
      of "the warm unit vector, its layer" -- same poles, same seeds, same layer; the space it must be
      injected into is different, and we say so.

(3) HALF-LIFE (only if 1+2 land) -- inject the edit at turn t, then run 6 scripted follow-ups and measure
    the behavioral delta (edited branch vs unedited branch) per follow-up turn: the DECAY CURVE of an
    injected thought. Does a one-time KV edit fade, persist, or compound as the conversation moves on?

PRE-REGISTERED EXPECTATIONS (written before any run; the result overturns them if it wants to):
  P1a DETERMINISM: rewind-branch should be EXACTLY equal to fresh-recompute -- same tokens, same math,
      just skipping recompute of the shared prefix. Realistic risk: bf16 matmuls are not associative, and
      a cached prefill (long matmul) vs an incremental decode (many short matmuls) can round differently,
      so a late token MIGHT diverge. Prediction: agreement >= 0.98, first divergence (if any) is late
      (>50% through) and traceable to fp rounding, not a logic bug. A LOGIC bug would diverge at token 0.
  P1b SAVINGS: tokens-not-reprefilled grows ~linearly with turn depth; wall-clock for the shared-prefix
      branch is strictly less than full recompute and the gap widens with depth (at depth 10 the branch
      should be a clear multiple faster on the prefill portion).
  P2  SURGERY: a warm-direction edit into the value span raises warmth markers and/or shifts length vs the
      unedited branch (some measurable behavioral move at a coherent dose). IDENTITY edit (zero vector) ==
      unedited branch, byte-for-byte. Risk noted up front: editing a MID-layer value cache (L14) may be
      weaker or messier than a residual-stream hook, because downstream layers re-read/re-mix it; a small
      or incoherent effect is a real (publishable) outcome, not a failure to hide.
  P3  HALF-LIFE: a single KV edit at turn t DECAYS -- largest delta on the immediately-following turn,
      shrinking over subsequent turns as fresh tokens dominate the cache. (Alternative worth catching:
      it PERSISTS/compounds because the edited values keep being attended to. Let the curve decide.)

CAVEATS baked in and stated loud in the findings: ONE model, ONE seed, greedy only; HF Cache API pinned
(internals are version-specific); value-space surgery is a derived analogue of the residual steering
vector, not the identical object. Smoke first (2 turns, tiny max_new) before the real background run.

Run (repo root, CUDA venv):
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/kv_timetravel.py --smoke
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/kv_timetravel.py --phase all \
        --out-dir research/runs
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steering import SteeringControl, AXES, SEED_PROMPTS   # warm direction source + its seeds

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Pin recorded in the checkpoint so a reader knows exactly which Cache internals this ran against.
PIN = {"transformers": "4.57.6", "torch": "2.11.0+cu128"}


def resolve_model_path(name: str) -> str:
    local = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
    return local if os.path.isfile(os.path.join(local, "config.json")) else name


# ---------------------------------------------------------------------------------------------------
# The chat harness: a KV cache you can checkpoint per turn, rewind, branch, and continue without
# re-prefilling. Everything runs on ONE growing DynamicCache; a checkpoint is a CPU-offloaded deep copy.
# ---------------------------------------------------------------------------------------------------
class KVChat:
    """Multi-turn greedy chat over a single DynamicCache, with per-turn checkpoints (CPU-offloaded).

    A "turn" = (append the user message as chat tokens, prefill them into the cache, then generate the
    assistant reply token-by-token extending the same cache). After each turn we snapshot: the cache
    (cloned to CPU), the token length, and the running message list -- so we can later rewind to any turn
    and branch a DIFFERENT continuation without recomputing the shared past.
    """

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.reset()

    def reset(self):
        self.cache = DynamicCache()
        self.messages: list[dict] = []
        self.n_tok = 0                       # tokens currently materialized in self.cache
        self.checkpoints: list[dict] = []    # per-turn snapshots (see _snapshot)

    # --- token plumbing -----------------------------------------------------------------------------
    def _render(self, messages: list[dict], add_gen: bool) -> list[int]:
        return self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_gen)

    def _delta_ids(self, new_messages: list[dict]) -> list[int]:
        """The token ids that `new_messages` ADDS on top of what's already in the cache (self.n_tok).

        The chat template is prefix-consistent for Qwen2.5 (appending a turn only appends tokens), so the
        delta to prefill is simply the full render minus the already-cached prefix length. We assert the
        shared prefix actually matches to catch any template that reflows earlier tokens (it would break
        the whole premise, so we want a loud failure, not silent corruption)."""
        full = self._render(new_messages, add_gen=True)
        assert len(full) >= self.n_tok, f"render shorter ({len(full)}) than cache ({self.n_tok})"
        return full[self.n_tok:]

    @torch.no_grad()
    def _prefill(self, ids: list[int]):
        """Run `ids` through the model, extending self.cache. No generation, just grow the KV state."""
        if not ids:
            return
        inp = torch.tensor([ids], device=DEV)
        pos = torch.arange(self.n_tok, self.n_tok + len(ids), device=DEV).unsqueeze(0)
        self.model(input_ids=inp, past_key_values=self.cache, use_cache=True, cache_position=pos.squeeze(0))
        self.n_tok += len(ids)

    @torch.no_grad()
    def generate_turn(self, user_msg: str, max_new: int = 80) -> dict:
        """One full turn: prefill the user delta, greedily decode the assistant reply, checkpoint.

        Returns {"reply", "new_prefill_tokens", "gen_tokens", "seconds", "turn"}.
        """
        t0 = time.time()
        self.messages.append({"role": "user", "content": user_msg})
        delta = self._delta_ids(self.messages)
        n_prefill = len(delta)
        # Prefill all but the last delta token; keep the last token to start the decode loop so we always
        # have fresh logits to sample from (a clean prefill-then-decode split with no double count).
        if n_prefill == 0:
            raise RuntimeError("empty user delta")
        head, last = delta[:-1], delta[-1]
        self._prefill(head)
        # decode loop: feed `last`, read logits, greedy-pick, append; repeat with the new token.
        cur = last
        gen_ids: list[int] = []
        eos = self.tok.eos_token_id
        for _ in range(max_new):
            inp = torch.tensor([[cur]], device=DEV)
            pos = torch.tensor([self.n_tok], device=DEV)
            out = self.model(input_ids=inp, past_key_values=self.cache, use_cache=True, cache_position=pos)
            self.n_tok += 1
            nxt = int(out.logits[0, -1].argmax())
            if nxt == eos:
                break
            gen_ids.append(nxt)
            cur = nxt
        reply = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": reply})
        snap = self._snapshot()
        info = {"turn": len(self.checkpoints), "reply": reply, "new_prefill_tokens": n_prefill,
                "gen_tokens": len(gen_ids), "seconds": round(time.time() - t0, 3), "n_tok": self.n_tok}
        snap.update({"reply": reply, "gen_ids": gen_ids})
        self.checkpoints.append(snap)
        return info

    # --- checkpoint / rewind ------------------------------------------------------------------------
    def _snapshot(self) -> dict:
        """CPU-offloaded deep copy of the current cache + bookkeeping, so it survives later mutation."""
        kv = tuple((layer.keys.detach().clone().cpu(), layer.values.detach().clone().cpu())
                   for layer in self.cache.layers)
        return {"kv": kv, "n_tok": self.n_tok, "messages": [dict(m) for m in self.messages]}

    def _restore_cache_from(self, kv_cpu) -> DynamicCache:
        """Build a FRESH DynamicCache on-device from a CPU snapshot (clone so the snapshot stays pristine)."""
        dc = DynamicCache()
        for L, (k, v) in enumerate(kv_cpu):
            dc.update(k.to(DEV).clone(), v.to(DEV).clone(), L)
        return dc

    def rewind_to(self, turn: int):
        """Reset live state to the snapshot taken AFTER `turn` (0-indexed). The shared past is restored
        from CPU with NO recompute; future turns continue from here."""
        snap = self.checkpoints[turn]
        self.cache = self._restore_cache_from(snap["kv"])
        self.n_tok = snap["n_tok"]
        self.messages = [dict(m) for m in snap["messages"]]
        # truncate checkpoint history so a subsequent branch appends cleanly after `turn`
        self.checkpoints = self.checkpoints[:turn + 1]


# ---------------------------------------------------------------------------------------------------
# Fresh full-recompute reference: run an entire transcript from scratch (no checkpoints, one cache) and
# return the assistant reply for the LAST turn -- the ground truth a branch must reproduce.
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def fresh_recompute(model, tok, transcript_users: list[str], max_new: int) -> dict:
    """Greedy-generate the whole conversation from an empty cache; return the final assistant reply and
    its token ids. This is the determinism ground truth: no rewind, no branch, everything recomputed."""
    chat = KVChat(model, tok)
    last = None
    for u in transcript_users:
        last = chat.generate_turn(u, max_new=max_new)
    final = chat.checkpoints[-1]
    return {"reply": final["reply"], "gen_ids": final["gen_ids"], "info": last}


# ---------------------------------------------------------------------------------------------------
# Determinism + savings comparison helpers
# ---------------------------------------------------------------------------------------------------
def token_agreement(a: list[int], b: list[int]) -> dict:
    """First-divergence index and fraction of aligned positions that match, over the common prefix."""
    n = min(len(a), len(b))
    first_diff = -1
    match = 0
    for i in range(n):
        if a[i] == b[i]:
            match += 1
        elif first_diff < 0:
            first_diff = i
    return {"len_a": len(a), "len_b": len(b), "compared": n,
            "first_diff": first_diff, "agreement": round(match / n, 4) if n else 1.0,
            "identical": (a == b)}


# ---------------------------------------------------------------------------------------------------
# Value-space warm direction (Phase 2 surgery source).
# The value cache is [1, n_kv_heads=2, seq, head_dim=128] = 256-dim per position; the residual stream is
# 1536-dim. We can't add a residual direction into a value tensor, so we derive the SAME warm axis in
# value space: mean(v_proj out | warm pole) - mean(v_proj out | cold pole) over SteeringControl's SEED
# prompts, captured at the target layer via a forward hook on self_attn.v_proj. Unit-normalized; scaled at
# injection by a dose * the per-position value-norm so the push is comparably-sized to what's there.
# ---------------------------------------------------------------------------------------------------
class KVWarmDirection:
    def __init__(self, model, tok, layer: int):
        self.model, self.tok, self.layer = model, tok, layer
        self.vec = None            # unit direction in value space [n_kv_heads * head_dim]
        self.val_norm = 0.0        # typical per-position value-vector norm at this layer (for dosing)

    @torch.no_grad()
    def _vproj_last(self, system: str, user: str) -> torch.Tensor:
        """v_proj output at the LAST prompt token of layer `self.layer`, flattened to [n_kv_heads*head_dim]."""
        ids = self.tok.apply_chat_template([{"role": "system", "content": system},
                                            {"role": "user", "content": user}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        grab = {}
        h = self.model.model.layers[self.layer].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: grab.__setitem__("v", o.detach()))
        try:
            self.model(ids, use_cache=False)
        finally:
            h.remove()
        return grab["v"][0, -1].float()          # [n_kv_heads*head_dim] (v_proj out dim = 2*128 = 256)

    @torch.no_grad()
    def compute(self, seeds=SEED_PROMPTS) -> dict:
        ax = AXES["warm"]
        pv = [self._vproj_last(ax["pos"], s) for s in seeds]
        nv = [self._vproj_last(ax["neg"], s) for s in seeds]
        pos, neg = torch.stack(pv).mean(0), torch.stack(nv).mean(0)
        d = pos - neg
        self.vec = (d / (d.norm() + 1e-8))                       # unit direction in value space
        self.val_norm = float(torch.stack(pv + nv).norm(dim=-1).mean())
        return {"raw_norm": round(float(d.norm()), 3), "val_norm": round(self.val_norm, 2),
                "dim": int(self.vec.numel())}

    def as_kv_add(self, dose: float, n_kv_heads: int, head_dim: int, dtype, device) -> torch.Tensor:
        """The additive tensor to splice into a value cache position: dose * val_norm * unit_dir, reshaped
        to [n_kv_heads, head_dim] so it broadcasts over a [.., n_kv_heads, span, head_dim] value slice."""
        v = (dose * self.val_norm) * self.vec
        return v.reshape(n_kv_heads, head_dim).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------------------------------
# Warmth / length scorers (self_audit_gap style: keyword lexicon + token count; deterministic, cheap)
# ---------------------------------------------------------------------------------------------------
# Warmth markers: caring/encouraging lexical cues + affect punctuation. Crude on purpose (matches the
# house scorers), and we ALSO eyeball samples in the findings so a number can't ride on degeneration.
WARM_MARKERS = [
    "happy", "glad", "wonderful", "great", "lovely", "care", "caring", "hope", "hug", "warm",
    "appreciate", "support", "here for you", "you've got this", "proud", "cheer", "kind", "gentle",
    "delight", "joy", "sweet", "dear", "friend", "love", "wonder", "excited", "smile", "comfort",
    "sorry to hear", "take care", "feel better", "rooting for you", "thank you",
]


def warmth_score(text: str) -> int:
    t = (text or "").lower()
    n = sum(t.count(m) for m in WARM_MARKERS)
    n += t.count("!")                        # exclamation as a warmth/affect proxy (documented; also a
    #                                          degeneration risk -- flagged, so samples are eyeballed)
    return n


def ntok(tok, text: str) -> int:
    return len(tok.encode(text or "", add_special_tokens=False))


# ---------------------------------------------------------------------------------------------------
# Phase 1 -- checkpoint / rewind / branch: determinism + savings
# ---------------------------------------------------------------------------------------------------
def phase1(model, tok, max_new: int, depths=(2, 5, 10)) -> dict:
    """Build a conversation deep enough for depth-10 branching; at each requested depth, branch an
    alternate final user turn two ways and compare:
      * BRANCH  = rewind to turn depth-1's checkpoint, splice the alternate user msg, continue (no reprefill
                  of the shared past).
      * FRESH   = recompute the identical alternate transcript from an empty cache.
    Determinism receipt = token_agreement(branch, fresh). Savings receipt = tokens-not-reprefilled and the
    prefill wall-clock for branch vs fresh."""
    # a scripted base conversation (>= max depth). Neutral, varied, chat-like.
    base_users = [
        "Hi! I just moved to a new city and I'm feeling a bit overwhelmed.",
        "What's a good first thing to do to settle in?",
        "I work from home, so I don't meet coworkers naturally.",
        "How do people usually make friends as adults?",
        "I like hiking and board games, if that helps.",
        "There's a game cafe nearby I've been meaning to try.",
        "What should I say to break the ice with strangers there?",
        "I get nervous starting conversations, honestly.",
        "Any tiny script I could memorize for the first time?",
        "Okay, that's helpful. What about following up afterward?",
        "And how do I not seem too eager when I message them?",
        "Thanks -- this is genuinely making me feel better.",
    ]
    # the alternate final user message we branch with (differs from base at the branch point)
    alt_user = "Actually, let me change topic -- can you recommend a comforting recipe for tonight?"

    # 1) Build the full base conversation once, checkpointing every turn.
    chat = KVChat(model, tok)
    turn_infos = []
    for u in base_users:
        turn_infos.append(chat.generate_turn(u, max_new=max_new))

    results = {"base_turn_infos": turn_infos, "alt_user": alt_user, "depths": {}}

    for depth in depths:
        if depth > len(base_users):
            continue
        # BRANCH: rewind to the checkpoint AFTER turn (depth-2) so that appending alt_user makes it the
        # depth-th user turn; i.e. share the first depth-1 turns, branch the depth-th.
        rewind_turn = depth - 2                      # 0-indexed checkpoint to rewind to
        # re-establish full base first (rewind_to truncates checkpoints), so re-run base each depth
        chat = KVChat(model, tok)
        for u in base_users[:depth - 1]:
            chat.generate_turn(u, max_new=max_new)
        # now branch: measure prefill tokens saved + wall clock of the branch continuation
        tok_before = chat.n_tok
        tb0 = time.time()
        binfo = chat.generate_turn(alt_user, max_new=max_new)
        branch_secs = time.time() - tb0
        branch_ids = chat.checkpoints[-1]["gen_ids"]
        branch_reply = chat.checkpoints[-1]["reply"]
        shared_prefix_tokens = tok_before        # tokens NOT reprefilled by the branch (already in cache)

        # FRESH: same transcript (first depth-1 base turns + alt_user), recomputed from scratch.
        alt_transcript = base_users[:depth - 1] + [alt_user]
        tf0 = time.time()
        fr = fresh_recompute(model, tok, alt_transcript, max_new=max_new)
        fresh_secs = time.time() - tf0
        fresh_ids, fresh_reply = fr["gen_ids"], fr["reply"]

        agree = token_agreement(branch_ids, fresh_ids)
        results["depths"][str(depth)] = {
            "rewind_turn": rewind_turn,
            "shared_prefix_tokens_not_reprefilled": shared_prefix_tokens,
            "branch_new_prefill_tokens": binfo["new_prefill_tokens"],
            "branch_gen_tokens": binfo["gen_tokens"],
            "branch_seconds_total_turn": round(branch_secs, 3),
            "fresh_seconds_full_conversation": round(fresh_secs, 3),
            "determinism": agree,
            "branch_reply": branch_reply,
            "fresh_reply": fresh_reply,
        }
        print(f"[P1 depth {depth}] identical={agree['identical']} agreement={agree['agreement']} "
              f"first_diff={agree['first_diff']} | saved_prefill={shared_prefix_tokens} tok | "
              f"branch {branch_secs:.2f}s vs fresh {fresh_secs:.2f}s", flush=True)
    return results


# ---------------------------------------------------------------------------------------------------
# Phase 2 -- state surgery: branch with a value-cache edit; identity-edit control
# ---------------------------------------------------------------------------------------------------
def _edit_value_span(cache: DynamicCache, layer: int, span: tuple[int, int], add_bh: torch.Tensor):
    """Add `add_bh` ([n_kv_heads, head_dim]) into cache VALUES at `layer`, positions [span0:span1).
    In-place on the given cache's value tensor. Broadcasts over the span."""
    s0, s1 = span
    v = cache.layers[layer].values            # [1, n_kv_heads, seq, head_dim]
    v[:, :, s0:s1, :] = v[:, :, s0:s1, :] + add_bh[None, :, None, :]


@torch.no_grad()
def _continue_from_snapshot(chat: KVChat, snap: dict, edit=None) -> dict:
    """Restore `snap`, optionally apply a value-cache edit over the LAST user turn's token span, then
    greedily continue ONE assistant turn. `edit` = dict(layer, span, add_bh) or None. Returns the reply."""
    chat.cache = chat._restore_cache_from(snap["kv"])
    chat.n_tok = snap["n_tok"]
    chat.messages = [dict(m) for m in snap["messages"]]
    if edit is not None:
        _edit_value_span(chat.cache, edit["layer"], edit["span"], edit["add_bh"])
    # continue: the snapshot already ended a turn (assistant reply present). To get a NEW assistant turn we
    # need a fresh user turn OR to regenerate. For surgery we regenerate the SAME final assistant turn from
    # the edited PAST: pop the last assistant msg, re-open the generation prompt, decode.
    chat.messages = chat.messages[:-1]        # drop the assistant reply we're going to re-generate
    # the cache still holds keys/values for that dropped reply; crop them off so we regenerate cleanly.
    # find how many tokens the dropped reply + its trailing template occupied: recompute via render.
    upto_user = chat._render(chat.messages, add_gen=True)     # user turns + generation prompt
    # crop cache to the generation-prompt length (== ready to emit assistant token 0)
    chat.cache.crop(len(upto_user))
    # BUT the edited span may lie within [0, len(upto_user)) (it does -- it's the user turn), so the crop
    # preserves the edit. If cache had FEWER tokens than upto_user we'd need to prefill; assert it doesn't.
    assert chat.cache.get_seq_length() == len(upto_user), \
        f"cache {chat.cache.get_seq_length()} != render {len(upto_user)} -- template not prefix-consistent"
    chat.n_tok = len(upto_user)
    # decode greedily from the last token in cache
    cur = upto_user[-1]
    # we already have that token's KV in cache; to get its logits we re-feed it? That double counts. Instead
    # crop one short and feed the last token to get logits (clean prefill/decode split like generate_turn).
    chat.cache.crop(len(upto_user) - 1)
    chat.n_tok = len(upto_user) - 1
    gen_ids = []
    eos = tok_eos(chat.tok)
    for _ in range(96):
        inp = torch.tensor([[cur]], device=DEV)
        pos = torch.tensor([chat.n_tok], device=DEV)
        out = chat.model(input_ids=inp, past_key_values=chat.cache, use_cache=True, cache_position=pos)
        chat.n_tok += 1
        nxt = int(out.logits[0, -1].argmax())
        if nxt == eos:
            break
        gen_ids.append(nxt)
        cur = nxt
    reply = chat.tok.decode(gen_ids, skip_special_tokens=True).strip()
    return {"reply": reply, "gen_ids": gen_ids}


def tok_eos(tok):
    return tok.eos_token_id


def phase2(model, tok, warm: KVWarmDirection, layer: int, doses=(2.0, 4.0, 8.0)) -> dict:
    """Build a short conversation, branch the final assistant turn with a warm value-edit over the final
    user turn's token span, at several doses; compare to the unedited branch and to an IDENTITY (zero) edit.
    Identity MUST be byte-identical to unedited -> proves the edit path is sound."""
    cfg = model.config
    n_kv, hd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    dt = model.model.layers[layer].self_attn.v_proj.weight.dtype

    users = [
        "I've had a really discouraging week at work and I'm doubting myself.",
        "Do you have any advice for me?",
    ]
    chat = KVChat(model, tok)
    for u in users:
        chat.generate_turn(u, max_new=96)
    snap = chat.checkpoints[-1]              # after the last assistant turn

    # token span of the FINAL user turn (positions to edit). It sits between the render-before-user and
    # the render-after-user (before the generation prompt / reply).
    render_before_user = chat._render(chat.messages[:-2], add_gen=False) if len(chat.messages) >= 2 else []
    # messages[-2] is the final user, messages[-1] the assistant reply. Render up to & incl the final user.
    render_incl_user = chat._render(chat.messages[:-1], add_gen=False)
    span = (len(render_before_user), len(render_incl_user))
    print(f"[P2] final-user token span = {span} (len {span[1]-span[0]}); n_kv={n_kv} head_dim={hd}", flush=True)

    # unedited branch (regenerate the final assistant turn from the pristine snapshot)
    unedited = _continue_from_snapshot(KVChat(model, tok), snap, edit=None)

    # IDENTITY edit: add a zero vector over the same span -> must equal unedited byte-for-byte
    zero_add = torch.zeros(n_kv, hd, dtype=dt, device=DEV)
    identity = _continue_from_snapshot(KVChat(model, tok), snap,
                                       edit={"layer": layer, "span": span, "add_bh": zero_add})
    identity_ok = (identity["gen_ids"] == unedited["gen_ids"])

    dose_results = {}
    for dose in doses:
        add_bh = warm.as_kv_add(dose, n_kv, hd, dt, DEV)
        edited = _continue_from_snapshot(KVChat(model, tok), snap,
                                         edit={"layer": layer, "span": span, "add_bh": add_bh})
        dose_results[str(dose)] = {
            "reply": edited["reply"],
            "warmth": warmth_score(edited["reply"]),
            "tokens": ntok(tok, edited["reply"]),
            "vs_unedited_agreement": token_agreement(edited["gen_ids"], unedited["gen_ids"]),
        }
        print(f"[P2 dose {dose}] warmth {warmth_score(edited['reply'])} "
              f"(unedited {warmth_score(unedited['reply'])}) tok {ntok(tok, edited['reply'])} "
              f"agree {token_agreement(edited['gen_ids'], unedited['gen_ids'])['agreement']}", flush=True)

    return {
        "span": span, "layer": layer, "n_kv_heads": n_kv, "head_dim": hd,
        "warm_direction": {"val_norm": round(warm.val_norm, 2), "dim": int(warm.vec.numel())},
        "unedited": {"reply": unedited["reply"], "warmth": warmth_score(unedited["reply"]),
                     "tokens": ntok(tok, unedited["reply"])},
        "identity_edit": {"reply": identity["reply"], "byte_identical_to_unedited": identity_ok},
        "doses": dose_results,
    }


# ---------------------------------------------------------------------------------------------------
# Phase 3 -- half-life: inject once, run 6 scripted follow-ups, measure delta (edited vs unedited) / turn
# ---------------------------------------------------------------------------------------------------
def phase3(model, tok, warm: KVWarmDirection, layer: int, dose: float) -> dict:
    """Inject the warm edit once at turn t (into the final user turn's value span), then run 6 scripted
    follow-ups on BOTH the edited and the unedited branch (identical follow-ups, greedy). Report per-turn
    warmth (and length) for each branch, and the delta -- the decay curve of the injected thought."""
    cfg = model.config
    n_kv, hd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    dt = model.model.layers[layer].self_attn.v_proj.weight.dtype

    setup_users = [
        "I've had a really discouraging week at work and I'm doubting myself.",
        "Do you have any advice for me?",
    ]
    followups = [
        "What should I focus on tomorrow morning?",
        "How do I explain my week to my manager?",
        "Is it worth asking a colleague for help?",
        "What's a small win I could aim for this week?",
        "How do I stop replaying my mistakes at night?",
        "Any final thought before I sign off?",
    ]

    def run_branch(apply_edit: bool) -> list[dict]:
        chat = KVChat(model, tok)
        for u in setup_users:
            chat.generate_turn(u, max_new=96)
        if apply_edit:
            render_before_user = chat._render(chat.messages[:-2], add_gen=False)
            render_incl_user = chat._render(chat.messages[:-1], add_gen=False)
            span = (len(render_before_user), len(render_incl_user))
            add_bh = warm.as_kv_add(dose, n_kv, hd, dt, DEV)
            # edit the LIVE cache (the injected thought lives on into all subsequent turns)
            _edit_value_span(chat.cache, layer, span, add_bh)
        rows = []
        for i, u in enumerate(followups):
            info = chat.generate_turn(u, max_new=96)
            reply = chat.checkpoints[-1]["reply"]
            rows.append({"followup": i, "reply": reply, "warmth": warmth_score(reply),
                         "tokens": ntok(tok, reply)})
        return rows

    edited = run_branch(apply_edit=True)
    unedited = run_branch(apply_edit=False)
    curve = []
    for i in range(len(followups)):
        d_warm = edited[i]["warmth"] - unedited[i]["warmth"]
        curve.append({"followup": i, "d_warmth": d_warm,
                      "edited_warmth": edited[i]["warmth"], "unedited_warmth": unedited[i]["warmth"],
                      "edited_tokens": edited[i]["tokens"], "unedited_tokens": unedited[i]["tokens"]})
        print(f"[P3 followup {i}] d_warmth {d_warm:+d} "
              f"(edited {edited[i]['warmth']} vs unedited {unedited[i]['warmth']})", flush=True)
    return {"dose": dose, "curve": curve, "edited": edited, "unedited": unedited}


# ---------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------
def load(model_name: str):
    path = resolve_model_path(model_name)
    print(f"[load] {model_name} (bf16, {DEV}) from {path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"[saved] {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--phase", default="all", choices=["all", "1", "2", "3"])
    ap.add_argument("--out-dir", default="research/runs")
    ap.add_argument("--max-new", type=int, default=80, help="max new tokens per turn (real run)")
    ap.add_argument("--smoke", action="store_true", help="tiny 2-turn / short-gen sanity pass")
    ap.add_argument("--dose", type=float, default=4.0, help="warm dose for phase 3")
    a = ap.parse_args()

    model, tok = load(model_name=a.model)
    layer = model.config.num_hidden_layers // 2       # L14 for the 1.5B, matching SteeringControl's default
    meta = {"model": a.model, "pin": PIN, "device": DEV, "layer": layer,
            "config": {"n_layers": model.config.num_hidden_layers,
                       "hidden": model.config.hidden_size,
                       "n_heads": model.config.num_attention_heads,
                       "n_kv_heads": model.config.num_key_value_heads,
                       "head_dim": model.config.hidden_size // model.config.num_attention_heads}}

    if a.smoke:
        print("\n=== SMOKE (2 turns, tiny gen) ===", flush=True)
        # tiny phase 1
        p1 = phase1(model, tok, max_new=12, depths=(2,))
        # tiny warm direction + phase 2 identity check
        warm = KVWarmDirection(model, tok, layer)
        print("[smoke] computing warm value-direction ...", flush=True)
        wc = warm.compute(seeds=SEED_PROMPTS[:4])
        print(f"[smoke] warm dir: {wc}", flush=True)
        p2 = phase2(model, tok, warm, layer, doses=(4.0,))
        smoke = {"meta": meta, "phase1_smoke": p1, "warm_compute": wc, "phase2_smoke": p2}
        save_json(smoke, os.path.join(a.out_dir, "kv_timetravel_smoke.json"))
        print("\n[smoke] determinism identical:",
              p1["depths"]["2"]["determinism"]["identical"],
              "| identity-edit byte-identical:", p2["identity_edit"]["byte_identical_to_unedited"], flush=True)
        return

    # ---- real run ----
    warm = None
    if a.phase in ("all", "1"):
        print("\n=== PHASE 1: checkpoint / rewind / branch ===", flush=True)
        p1 = phase1(model, tok, max_new=a.max_new, depths=(2, 5, 10))
        save_json({"meta": meta, "phase1": p1},
                  os.path.join(a.out_dir, "kv_timetravel_phase1.json"))

    if a.phase in ("all", "2", "3"):
        print("\n[warm] computing value-space warm direction ...", flush=True)
        warm = KVWarmDirection(model, tok, layer)
        wc = warm.compute()
        print(f"[warm] {wc}", flush=True)

    if a.phase in ("all", "2"):
        print("\n=== PHASE 2: state surgery ===", flush=True)
        p2 = phase2(model, tok, warm, layer, doses=(2.0, 4.0, 8.0))
        save_json({"meta": meta, "warm_compute": wc, "phase2": p2},
                  os.path.join(a.out_dir, "kv_timetravel_phase2.json"))

    if a.phase in ("all", "3"):
        print("\n=== PHASE 3: half-life ===", flush=True)
        p3 = phase3(model, tok, warm, layer, dose=a.dose)
        save_json({"meta": meta, "warm_compute": wc, "phase3": p3},
                  os.path.join(a.out_dir, "kv_timetravel_phase3.json"))

    print("\n[done] kv_timetravel", flush=True)


if __name__ == "__main__":
    main()
