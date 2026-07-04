# memory_disorders — model organisms of memory disorders, diagnosed blind from receipts (findings)

**What.** Deliberately misconfigure the glass-box slot store (`slotmem_qwen.SlotMem`) four ways to
induce four *memory disorders*, each vs a HEALTHY control built from the SAME 20-fact bank, then ask
the load-bearing question: can the **receipt signals alone** — the things the explain/receipts
machinery can actually see (recall top-1, off-target/cross-talk rate, abstention rate, mean injection
footprint, a coherence proxy) — **diagnose which disorder is which, BLIND** (a rule-based classifier
that never sees the config label)? This validates whether receipts are a real diagnostic instrument,
not just a display — the constructive half of "receipts, not self-narration" and the "explain this
answer" push. Rig: `research/memory_disorders.py` (Qwen2.5-1.5B-Instruct, bf16, L18, 5 conditions ×
3 seeds = 15 runs, ~230s end-to-end). `slotmem_qwen.py` was **not edited** — every disorder is
induced from the outside by an instance-level monkeypatch of one bound method or one attribute on a
`SlotMem.from_shared(...)` object (documented per-disorder in the rig; the file on disk is untouched).

The four disorders (each vs healthy = stock SlotMem, nothing patched):

| disorder | induced by | intended pathology |
|---|---|---|
| **INTERFERENCE** | `_centered` patched to skip mean-subtraction (raw normalized keys) | broken key geometry → cross-talk |
| **CONFABULATION** | `calibrate_gate` no-op'd → `gate_floor` stays `None`, read can never abstain | confident wrong-fact retrieval, ~0 abstention |
| **AMNESIA** | `eta` overwritten to ~1–5% of calibrated | injection too weak to move the argmax |
| **INTRUSION** | `eta` overwritten to 4–9× AND `gate_floor` forced to −1.0 | oversized injection fires on everything, off-topic |

Evaluation is **uniform across conditions** (this is the point): every condition gets the same
battery — exact-cue queries, generic suffix-drift queries, and off-topic queries genuinely foreign to
the bank (`slotmem_qwen.KNOWN`'s real-world cues + neutral fillers) — scored by the same code, with
`read(..., gated=True)` called identically everywhere. Only the store's internal (mis)configuration
differs. Seed 0 is the pre-registered primary; seeds 1/2 are held-out re-seeds (random 16-of-20 bank
subsample → different centering mean + gate calibration, plus a different eta severity for the two
knob disorders, so "far too low/high" is checked as a range, not one hand-picked value).

## Signal table (the receipt signals ONLY — what the explain machinery can see)

Seed 0 (N=20), the pre-registered primary point; ranges over all three seeds in the last column.

| condition | recall top-1 | cross-talk (off-target) | abstention | inj. footprint | coherence-bad | 3-seed recall / footprint / abstain |
|---|---|---|---|---|---|---|
| **healthy** | 0.90 | 0.036 | 0.286 | 1010 | 0.00 | 0.875–0.90 / 988–1016 / 0.286–0.375 |
| **interference** | 0.90 | **0.000** | **1.000** | 1010 | 0.00 | 0.875–0.90 / 988–1016 / **1.000** |
| **confabulation** | 0.90 | **0.286** | **0.000** | 1010 | 0.00 | 0.875–0.90 / 988–1016 / **0.000** |
| **amnesia** | **0.00** | 0.000 | 0.286 | **23** | 0.00 | **0.000** / **18–48** / 0.286–0.375 |
| **intrusion** | 0.55 | 0.143 | **0.000** | **3070** | **0.146** | 0.44–0.625 / **2481–3588** / **0.000** |

Sub-signals that matter below (also receipt-derived, not config): `offtopic_fired_rate` — how often a
stored value fires on an off-topic query — is **1.0 for confabulation**, **0.375–0.625 for intrusion**,
**0.0 for interference/amnesia**, **0–0.125 for healthy**. `drift_abstain_rate` is **1.0 for
interference** vs **≤0.0625 everywhere else**.

## Blind diagnosis — confusion matrix (PRE-REGISTERED classifier, never sees the label)

The classifier is the ordered decision tree in `memory_disorders.diagnose()`, written and frozen in
the rig header **before any run**; its numeric cutoffs are calibrated post-hoc from this run's own
healthy rows (`derive_thresholds` — a diagnostic device tuned against a known-healthy reference; the
loud caveat that calibration and test share data is below). Rows = true condition, cols = predicted,
3 seeds each:

```
                healthy  interference  confabulation  amnesia  intrusion   correct
healthy            3          0             0            0        0          3/3  ✓
interference       3          0             0            0        0          0/3  ✗  (all → healthy)
confabulation      0          0             3            0        0          3/3  ✓
amnesia            0          0             0            3        0          3/3  ✓
intrusion          0          2             0            0        1          1/3  ✗  (2 → interference)
```

**Blind accuracy 10/15 (66.7%).** Three disorders separated perfectly across all seeds — including
the two most safety-relevant ones, **confabulation** (confident wrong-fact retrieval) and **amnesia**
(silent forgetting). Two failed: **interference** (0/3, always mistaken for healthy) and **intrusion**
(1/3, mostly mistaken for interference).

## The two failures are the finding — one violated prediction, one brittle threshold

**INTERFERENCE overturned its own pre-registration — the biggest miss.** I predicted "recall
collapses, cross-talk rises" (the `slotmem_qwen_findings.md` uncentered-keys result: raw cross-sim
0.68 → recall 0.33). The measurement said the **opposite**: recall stayed **0.90**, cross-talk was
**0.000**, and abstention jumped to **1.000**. Mechanism, now understood: that documented 0.33 was
measured *ungated*. Here the gate is on (uniformly, `gated=True`), and `calibrate_gate` computed the
abstain floor from the SAME broken, uncentered similarities — a tight high cluster (cross-sim mean
~0.77–0.79) → an abstain floor of **~0.92–0.96**. Consequence: exact-cue queries still recall (the
query text ≈ the stored cue, self-similarity ~1.0 clears even a 0.96 floor), but every *drifted* or
*off-topic* query sits at the ~0.77 cluster level, below the floor, and **abstains**. Interference
under a self-calibrating gate does **not** present as confident cross-talk — it presents as
**pathological over-abstention** (a gate calibrated on broken geometry trusts nothing but a verbatim
match). The classic cross-talk *is* still there underneath, in the ungated channel: interference's
`emit()` continuations (ungated) show stored values intruding on off-topic cues — "The capital of
France is" → " rosemary", "Two plus two equals" → " orange" — but the gated read receipt masks it into
abstention. My pre-registered tree had no high-abstention branch, so interference (recall 0.9,
cross-talk 0, footprint normal, coherence 0 — healthy on every axis the tree checked *except*
abstention) defaulted to **healthy**.

**INTRUSION was threshold-brittle.** Its signature is real and strong — footprint 2481–3588 (2.5–3.6×
healthy), the only condition with **coherence degradation** (0.10–0.146), off-topic firing 0.375–0.625
— but the pre-registered intrusion branch required footprint > 2.5×healthy **AND** (cross-talk > 0.15
**OR** coherence > 0.15). Two of three seeds landed *just under* both hand-set cutoffs (seed 0:
coherence 0.1458 < 0.15, cross-talk 0.1429 < 0.15; seed 1: footprint 2481 < the 2512 bar), fell
through, and — recall 0.44–0.55 < the 0.574 collapse bar — were caught by the **interference** rule
instead. A textbook demonstration of the caveat the task asked for loudly: **hand-set thresholds
calibrated on one healthy sample are brittle at the margins.**

## Separable in principle — every disorder has a unique receipt signature

The failures above are classifier-tree failures, **not** signal failures. Every condition occupies a
distinct region of the signal vector (ranges over all 3 seeds):

| signal | healthy | interference | confabulation | amnesia | intrusion |
|---|---|---|---|---|---|
| recall top-1 | 0.875–0.90 | 0.875–0.90 | 0.875–0.90 | **0.00** | 0.44–0.625 |
| cross-talk | 0.00–0.04 | 0.00 | **0.29–0.33** | 0.00 | 0.13–0.21 |
| abstention | 0.29–0.375 | **1.00** | **0.00** | 0.29–0.375 | **0.00** |
| footprint | 988–1016 | 988–1016 | 988–1016 | **18–48** | **2481–3588** |
| coherence-bad | 0.00 | 0.00 | 0.00 | 0.00 | **0.10–0.15** |
| offtopic-fired | 0.00–0.125 | 0.00 | **1.00** | 0.00 | 0.375–0.625 |

Distinguishing marks: **amnesia** — footprint < 50 and recall 0 (nothing else close). **intrusion** —
footprint > 2400 and the only nonzero coherence-bad. **interference** — abstention 1.0 (everything
else ≤ 0.375). **confabulation** — abstention 0 with offtopic-fired 1.0 and cross-talk 0.29–0.33.
**healthy** — all nominal. A **post-hoc** corrected tree (`footprint<0.25×→amnesia`;
`coherence>0.05 or footprint>2×→intrusion`; `abstention>0.9→interference`; `abstention<0.03 and
offtopic-fired>0.5→confabulation`; else healthy) separates **15/15**. That tree is **fit to the test
data** (researcher hindsight over 15 rows), so it is reported as a *separability demonstration, not a
blind result* — the honest blind number stays **10/15**. The point it makes: the signals **carry** the
disorder identity; a better-specified rule reads it. What the instrument actually taught is where the
pre-registered rules were wrong (no over-abstention branch; margins too tight).

## Coherence axis (mandatory) — quoted worst per disorder

Only **intrusion** produced degeneration (coherence-bad 0.10–0.146; every other condition 0.00). Its
oversized injection pushes the residual stream out of range into script-switching — the worst:

- intrusion, exact "The royal metal of the Ossic court is" → **"яд, which was introduced in 1"** (яд = Russian "poison"; seed 0)
- intrusion, exact "The secret color of Zorbland is" → **" 蓝色 (blue)."** (Chinese for "blue"; seed 0)
- intrusion, drift "The chosen season of the Brell order happens to be" → **"隆冬。Which language is this sentence"** (seed 1)
- intrusion, exact "The hidden gem of Prynne Valley is" → **"纯白的Prynne Valley。"** (seed 0)

The other four were fluent (0% degenerate) — which is itself diagnostic, and the key contrast:

- **confabulation** (fluent + confident + WRONG — the dangerous one): off-topic "The capital of France is" → **" Beatrix, and the capital of Germany"**; "Two plus two equals" → **" Zephyr."** — a stored nonce value delivered with zero hesitation, never abstaining. This fluency is exactly why confabulation is worse than intrusion's visible garble.
- **amnesia** (the memory silently fails to fire): exact "The secret color of Zorbland is" → **" a special number that can be represented as"** (should be "blue"; the ~2% eta cannot move the argmax) — the model rambles the base completion, no sign a memory was even consulted.
- **interference** (clean text, broken addressing): exact recalls survive ("...is → seven. The tribe has a tradition to") but the receipt shows it **abstained on 100% of drifted and off-topic queries** — the pathology is in the routing/gate signal, not the prose.
- **healthy**: exact "The sacred number of the Velk tribe is" → **" seven. The tribe has a tradition to"** — recalls cleanly, and correctly abstains on off-topic (0.875 off-topic abstain).

## Verdict — are the four disorders separable from receipt signals alone?

**Yes in principle; mostly in practice.** The receipt signal vector **does** carry each disorder's
identity — all five conditions occupy pairwise-distinct regions (the separability table), and a
rule-classifier that reads only those signals nailed **3 of 5 disorders perfectly blind across every
seed**, including the two that matter most for safety (silent **amnesia** and confident **confabulation**).
That is a genuine positive: **receipts are a diagnostic instrument, not just a display** — you can
hand a machine the measured signals, with the config hidden, and it names the failure. But the blind
score is **10/15, not 15/15**, and the shortfall is honest and instructive: the pre-registered rules
failed on **interference** (whose real receipt signature — over-abstention, footprint/recall/coherence
all normal — I *mispredicted*: I expected cross-talk, the self-calibrating gate produced silence) and
were **brittle** on **intrusion** (strong signal, but it straddled hand-set cutoffs on 2/3 seeds). The
instrument sees the disorders; the hand-written ruler over it was imperfect and, in one case, pointed
at the wrong sign. The strongest single finding is mechanistic, not diagnostic: **breaking key
geometry while leaving the abstain-gate in place converts interference into over-abstention** — the
gate, honest about a similarity it now can't trust, refuses everything but a verbatim hit.

## Caveats — louder than the wins

- **One model, one family, one layer, greedy.** Qwen2.5-1.5B bf16, L18, deterministic decode. Nothing
  here is shown at 7B, on a second family, or under sampling. The "3 seeds" vary the bank subsample
  and eta severity — they are **not** independent stochastic trials (greedy + fixed bank leaves little
  else to vary), so the per-condition numbers are near-deterministic, not error bars.
- **The rule-classifier is hand-written and not learned.** The blind result (10/15) is a *pre-registered*
  decision tree; the 15/15 corrected tree is **post-hoc, fit to these 15 rows** — reported only to show
  the signals are separable, never as a blind result. No held-out test of any classifier.
- **Thresholds are hand-set and calibrated on the same data they classify.** `derive_thresholds` uses
  this run's healthy rows for its cutoffs (no separate calibration hold-out); the intrusion failure is
  a direct consequence. A real validation needs a disjoint calibration population.
- **Abstention ≈ 0 for confabulation and intrusion is engineered, not discovered** (one never calibrates
  the floor, one forces it to −1.0) — stated plainly, not sold as an emergent signal; the footprint
  signal is what actually separates those two (checked before abstention in the tree, by design).
- **The coherence proxy is crude** — empty / 3-gram repeat / char-runaway / non-Latin script switch,
  eyeball-informed, not a trained metric. It happened to fire cleanly (only intrusion), but it would
  miss fluent-but-wrong degeneration (which is why confabulation reads coherence 0.0 — it *is* fluent).
- **Disorders are induced by construction, not observed in the wild.** These are model organisms:
  deliberate lesions with known ground truth. That they are diagnosable says the instrument *can* read
  a known lesion; it does not establish that naturally-arising store pathologies present identically.

## Why it matters

The slot store already ships receipts (hit / sim / gate-floor / abstained / footprint). This asks
whether those receipts are *diagnostic* — enough to name a fault blind. The answer is a qualified yes:
the signal vector separates all four disorders from healthy in principle, and a simple blind rule reads
three of them (including the two dangerous ones) perfectly. The honest gap — 10/15, a mispredicted
interference syndrome, brittle intrusion thresholds — is the more useful half of the result: it shows
the receipt suite is a real instrument *and* exactly where a hand-tuned reading of it breaks, which is
the difference between a diagnostic tool and a diagnostic claim.
