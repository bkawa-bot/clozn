# persistent_injection — DEFERRED (nf4 KV-edit wiring bug) (findings)

*Wild Experiment #2, Wave 1. Pre-registration: `WILD_WAVE1_PREREG.md` (exp 2). Status as of 2026-07-05:
**deferred — rig built + unit-tested, but the smoke exposed a wiring bug that makes the on-model result
untrustworthy, so no cross-family run was taken.** Rig: `research/persistent_injection.py` (+ 49
model-free tests, all passing); smoke: `research/runs/persistent_injection_smoke.json`.*

## What this experiment was for

Test whether `kv_timetravel`'s **<1-turn half-life of a one-shot KV edit** (Law #3, "state is not
storage") is universal physics, and map the persistence phase diagram — the smallest intervention that
survives past one turn (edit 1 vs N positions × K vs V cache × once vs re-inject × raw vs trained
phantom slot), measured by a transparent warm-marker rate decaying over turns, with shuffled-direction
and no-injection nulls. Law #3 itself is already **established at 1.5B bf16** by the antecedent; this was
the cross-family + phase-diagram extension, i.e. confirmatory, not decision-critical.

## Why it's deferred — the bug (diagnosed, not fixed)

The Qwen-7B smoke ran end to end and the honesty gate worked, but **every cell reported
`turn0_effect = 0.0` and GATE-FAIL**: the warm-injected turn-0 reply was **byte-identical** to the
no-injection baseline (and to the shuffled-null). This is not the weak-intervention / dose-calibration
pattern seen in parliament (2/5 stances live) and quine (26% shift): the injected magnitude is
`dose × val_norm = 4.0 × 13.91 ≈ 56` per position — **4× the natural V-cache norm** (val_norm 13.91 is
the measured norm of a real V position). A perturbation 4× the natural magnitude would grossly change
greedy generation *if it landed*. Byte-identical output means **the KV edit is not reaching the
generation path** — a wiring bug in the nf4 `DynamicCache` port, most likely one of: the edit mutating a
copy rather than the live cache; the edited span not being the one attended to during turn-0 decode; or
a re-prefill rebuilding the cache and discarding the edit. The geometry read is correct (Qwen 28 layers /
4 KV heads / head_dim 128; direction dim 512), and the phantom arm trained fine (KL 0.59→0.14), so the
break is specifically in `_edit_kv_span` reaching live generation, not in setup.

**The gate did its job:** it flagged GATE-FAIL and refused to report a decay curve off a non-effect,
rather than laundering byte-identical replies into a fake "0-turn half-life." That is the coherence/gate
discipline working — the rig is honest about its own failure.

## To resume (for a future session)

1. Confirm the live-vs-copy question first: log the cache tensor's norm at the edited span
   *immediately before turn-0 decode* — if the +56 push isn't visible there, the edit is on a copy /
   wrong object. If it IS visible but output is unchanged, the edited positions aren't in the decode's
   attention window (the span/positions math, or an nf4 cache that re-materializes on read).
2. A `--dose` flag exists; but do NOT reach for a bigger dose — 4× natural norm already proves it's not
   magnitude. Fix the wiring, re-smoke (expect a clear turn-0 effect that then decays), then run the
   cross-family pair.
3. Everything above the edit is trusted (geometry, direction, phantom training, the 10-cell sweep, the
   turns_to_noise gate) — the fix is localized to `_edit_kv_span` / how the raw branch hands the edited
   cache to `KVChat.generate_turn`.

Tracked as a to-do in `NEXT_STEPS.md`. Rig committed so the work + diagnosis survive.
