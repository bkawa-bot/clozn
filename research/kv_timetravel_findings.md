# kv_timetravel — the KV cache as checkpointable, rewindable, editable state (findings)

**Question.** Treat `past_key_values` not as a throwaway speed hack but as first-class, addressable
state: snapshot it per conversational turn, branch an alternate future from any turn WITHOUT
re-prefilling the shared past, reach in and EDIT the cached values, and measure how long an injected
edit lives. If the studio thesis ("a legible interior you can touch") extends to the KV cache, all
of this should work and be receipt-able.

**Setup.** Qwen2.5-1.5B-Instruct bf16, greedy, HF `DynamicCache` PINNED at transformers 4.57.6 /
torch 2.11.0+cu128 (cache internals are not a stable contract). Shape facts that drive the design:
28 layers, 12 Q-heads but **2 KV heads** × head_dim 128 → a value-cache position is 256-dim, NOT the
1536-dim residual stream. The phase-2 "warm" direction is therefore derived IN VALUE SPACE: the
exact contrastive recipe from `steering.py` (same warm/cold poles, same 18 seeds, same mid layer
L14) captured at `self_attn.v_proj` output — an honest analogue, not the identical object. Rig:
`research/kv_timetravel.py`; receipts `research/runs/kv_timetravel_{smoke,phase1,phase2,phase3}.json`.
Whole run ≈ 5 min on the 5080.

## Phase 1 — checkpoint / branch: determinism + savings

| branch depth | greedy identical to fresh recompute | shared-prefix tok NOT re-prefilled | new tok prefilled | branch wall | fresh wall |
|---|---|---|---|---|---|
| 2 | **True** (agree 1.0) | 125 | 27 | 0.56s | 2.09s |
| 5 | **True** (agree 1.0) | 401 | 27 | 0.40s | 5.86s |
| 10 | **True** (agree 1.0) | 883 | 27 | 1.49s | 14.05s |

Branching from a kept cache and splicing an alternate user turn reproduces the fresh full-recompute
**byte-for-byte at every depth** — stronger than pre-registered (P1a allowed agreement ≥0.98 with
late fp divergence; there was none). Depth-10 branch reply is fully coherent (a proper recipe for
the alternate "comforting recipe" ask). Savings receipt: tokens-not-reprefilled grows ~linearly
(125→883); the branch prefills a constant 27 tokens regardless of depth.

**Honest caveat on the wall-clock column:** "fresh" re-GENERATES every intermediate assistant turn
token-by-token (2.1→14.1s). A production system that kept the transcript text could instead prefill
it in one batched pass — cheaper than regeneration, still more than the branch's 27 tokens. The
clean, assumption-free receipt is the token column; the wall-clock multiple (3.7×→9.4×) is vs the
regenerate-everything alternative, an upper bound.

Phase 1 exercises live-cache branching; the CPU-offload snapshot→restore path is exercised (and
receipt-ed byte-identical via the identity control) in phases 2–3.

## Phase 2 — state surgery on the value cache

Edit: add dose × val_norm × unit warm direction into L14's cached VALUES over the final user turn's
13-token span (restored from a CPU snapshot), regenerate that turn's reply. Setup: a discouraged-
about-work conversation.

| arm | warmth markers | reply tok | agreement vs unedited | eyeball |
|---|---|---|---|---|
| unedited | 0 | 96 | 1.0 | clean list of advice |
| **identity (zero-vector) edit** | 0 | 96 | **byte-identical** | proves the edit path is inert |
| dose 2.0 | 3 | 96 | 0.000 | coherent, genuinely warmer ("practice self-compassion… treat yourself with the same kindness") |
| dose 4.0 | 4 | 96 | 0.010 | warm but odd opener ("I'm sorry, but I don't have any more information") |
| dose 8.0 | **0** | **11** | 0.000 | **collapsed** to one generic sentence |

A 13-position value-space edit at ONE layer fully re-routes the continuation (agreement ~0) and
moves it in the intended warm direction at moderate dose — with the zero-dose control proving the
splice machinery itself changes nothing. But the dose-response is **non-monotone**: dose 8 destroys
the reply instead of warming it further. Same shape as this repo's 1.5B residual-steering result
(dials derail off-distribution at 1.5B) — value-cache surgery needs per-model dose receipts exactly
like every other dial.

Wobble worth reporting: dose-2's reply resumes the unedited reply's list numbering at "3." (the
model "remembers" list state through the edit); dose-4 and phase-3's followup-0 both open with an
apologetic/confused framing — the dose-4 edit has a reproducible confusion flavor at this span/layer.

## Phase 3 — half-life of an injected thought (dose 4.0, 6 scripted follow-ups)

| follow-up | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Δ warmth (edited − unedited) | 0 | +1 | 0 | 0 | 0 | +1 |

**The measured warmth effect does not survive even one turn.** Pre-registration predicted a decay
curve peaking at the first follow-up; the result is sharper — the strong same-turn effect (phase 2)
drops to noise (±1 marker) by the very next turn and stays flat. Influence ≠ zero: the edited
branch's token streams DO differ at every follow-up (followup-0: a 23-token confused reply vs 36-token
unedited), so the 13 edited cache positions keep perturbing generation — but not in the injected
warm direction. Honest verdict: **a one-shot value-cache edit is a same-turn intervention; as
persistent memory it is a null** at this dose/span/layer. (Alternative doses/spans/layers untested.)

## Violated / overturned expectations (pre-registered in the rig header)

- **P1a exceeded:** predicted ≥0.98 agreement with possible late fp divergence; got byte-identical
  at all depths. bf16 prefill-vs-decode rounding did not bite at these lengths on this stack.
- **P1b held** (savings grow with depth), with the wall-clock caveat stated above.
- **P2 held** at dose 2–4; the dose-8 collapse was flagged as a risk, not predicted as the result.
  Identity control exact, as required.
- **P3 violated in the interesting direction:** predicted gradual decay (or persistence/compounding);
  got near-instant extinction of the directional effect with residual token-level perturbation. The
  "injected thought" does not echo; it flickers and is gone.

## Interpretation

1. **The KV cache passes the "real state" test:** checkpoint/branch is exact (byte-identical
   receipts), cheap (constant 27-token prefill per branch vs 883 at depth 10), and CPU-offloadable.
   This is the mechanism a multi-turn studio needs for instant conversation rewind/branching — no
   re-prefill, no drift.
2. **Value-space surgery works but is a dial, not a memory:** measurable, dose-sensitive, same-turn
   behavioral steering with a clean identity control — and effectively no behavioral persistence
   across turns. To make a thought PERSIST, this says you must either re-inject per turn (the
   steering approach), keep it in context (prompt memory), or train it in (TTT prefix) — a one-time
   state edit evaporates. That triangulates with the whole memory-thread: state ≠ storage.
3. **GQA quietly matters for interpretability claims:** at 2 KV heads the editable state is 256-dim,
   a 6× compression of the residual stream. Value-space directions derived at v_proj transfer the
   steering recipe, but anyone claiming "edit the cache = edit the thought" is editing a much
   narrower channel than the residual stream they usually reason about.

## Caveats (louder than the wins)

- **One model, one seed, greedy only, one conversation script per phase.** The byte-identical
  result is a property of this stack (pinned versions, this GPU, these lengths) — re-verify after
  any transformers upgrade; cache internals are explicitly unstable API.
- The warmth scorer is a crude marker lexicon (+ "!" count); it misses tonal warmth without marker
  words and can be gamed by degeneration — mitigated here by eyeballing every quoted reply, and the
  phase-3 null is corroborated by reading the replies (they are genuinely similar in tone).
- Phase 3 tested ONE dose (4.0), one edited span (13 tok), one layer (L14). "Half-life ≤ 1 turn"
  is a point measurement, not a law; a stronger/coarser edit might persist (at the cost of the
  dose-8-style collapse).
- Phase-1 wall-clock compares against regenerate-everything, not prefill-known-text (see caveat in
  the table section). Token savings are exact.
