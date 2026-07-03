# vector_telepathy — do slot-memory keys port through a fitted linear bridge? (findings)

**Question.** A prior session asserted activation-space memories "cannot port because geometries
differ." This rig gives that claim the receipts treatment: build a 20-fact slot store on Qwen2.5-
1.5B-Instruct (model A, bf16, L18, H=1536), fit a linear bridge to Qwen2.5-7B-Instruct (model B,
nf4, L18, H=3584) on 512 diverse sentences, port ONLY the key vectors (never text) through the
bridge, and measure recall on the target model against a text-recompile ceiling and two nulls.
Reverse direction (B->A) runs the same table. Scope, stated honestly: same-family, same-vocab
(values need no bridge — shared token ids, taken from the target model's own unembedding); the
pure telepathy question is whether a cue's MEANING as a residual vector survives a fitted map,
isolated via SELECT (top-1 key retrieval) separate from EXPRESS (answer argmax after injection,
which adds value/dose mechanics on top). Cross-family (different tokenizer/vocab) is untested.

**Setup.** `research/vector_telepathy.py`. Bridges tried: ridge-256 (primary, pre-registered),
ridge-128/512 (opportunistic, same fit pool), Procrustes-on-PCA192 (rotation-only), random-matrix
null (norm-matched to ridge-256's W), shuffled-pair-fit null (same lambda as ridge-256, fit on
mismatched X-Y pairs). 20 exact fact cues + 10 paraphrases. 544 fit/held-out sentences (512 fit +
32 held, disjoint from all fact templates/subjects — leakage-asserted in the rig). eta = target
model's own calibration in every arm including ceiling. Repro:
`<venv>/python.exe research/vector_telepathy.py [--selftest|--smoke]`. Full run: 490.6s.

**Bug found in smoke, fixed before the full run (commit 4989c1f):** `run_port_arms()` is defined
with 9 params `(mem, bridges, keys_src, keys_tgt_native, bank, paras, cfg, tag, ceiling)` but both
call sites (stage_B, stage_C) passed only 8 positional args, omitting `cfg` — `tag`/`ceiling`
shifted into the wrong slots and the trailing `ceiling` arg had nowhere to bind, crashing every run
before any port arm executed. Confirmed via traceback + a static AST arity check (0 other
mismatches found in-file after the fix). Smoke passed end-to-end after the fix (390.8s); this is a
pipeline defect, not a science finding — the bridge-fitting code that ran BEFORE the crash point
had already produced sane numbers, foreshadowing the full-run result.

**GPU-gate workaround used (a run-time shim, not a file edit):** the committed rig's
`wait_for_gpu()` gates on total VRAM < 3 GB sustained — unsatisfiable on this box (WDDM floor holds
~4.7-4.8 GB even fully idle, confirmed via `nvidia-smi` at rest: 4792 MiB / 0% util). Ran via a
runtime shim that monkeypatches only `wait_for_gpu` to gate on `utilization.gpu < 15%` sustained
120s instead (the documented workaround). `vector_telepathy.py` itself is untouched by this.

## Result — full run (n=20 exact, n=10 paraphrase, 544 sentences, layer 18)

| direction | arm | exact select | exact express | para select | para express | heldout cos (cen) | cue cos (cen) |
|---|---|---|---|---|---|---|---|
| A->B | no_memory | 0.0 | 0.0 | 0.0 | 0.0 | -- | -- |
| A->B | **ceiling** | 1.0 | 0.8 | 0.9 | 0.9 | -- | -- |
| A->B | ridge (256, primary) | 0.65 | 0.5 | 0.5 | 0.5 | 0.7788 | 0.1819 |
| A->B | ridge512 | 0.8 | 0.6 | 0.7 | 0.7 | 0.8072 | 0.2308 |
| A->B | ridge128 | 0.6 | 0.45 | 0.5 | 0.5 | 0.7261 | 0.1134 |
| A->B | procrustes (d=192) | 0.85 | 0.7 | 0.6 | 0.6 | 0.7560 | 0.1673 |
| A->B | random (null) | 0.05 | 0.05 | 0.2 | 0.2 | 0.0056 | -0.0069 |
| A->B | shuffled_fit (null) | 0.0 | 0.0 | 0.0 | 0.0 | -0.0149 | -0.0218 |
| B->A | no_memory | 0.0 | 0.0 | 0.0 | 0.0 | -- | -- |
| B->A | **ceiling** | 1.0 | 0.9 | 0.9 | 0.9 | -- | -- |
| B->A | ridge (256, primary) | 0.75 | 0.7 | 0.5 | 0.5 | 0.8404 | 0.2348 |
| B->A | ridge512 | 0.8 | 0.75 | 0.5 | 0.5 | 0.8674 | 0.3176 |
| B->A | ridge128 | 0.65 | 0.6 | 0.4 | 0.4 | 0.7910 | 0.1445 |
| B->A | procrustes (d=192) | 0.8 | 0.7 | 0.5 | 0.5 | 0.8022 | 0.1909 |
| B->A | random (null) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0092 | -0.0100 |
| B->A | shuffled_fit (null) | 0.05 | 0.05 | 0.0 | 0.0 | 0.0030 | -0.0085 |

Full receipts: `research/runs/telepathy/telepathy_full.json` (per-arm per-item hit/select/top-token,
plus `research/runs/telepathy/arm_*.json` per-arm files and `stage_{a,b,c}*` caches).

## Eyeballing the actual items (house rule: eyeball before believing metrics)

The aggregate numbers are corroborated at the item level, not just averaged noise. On A->B ridge256,
when SELECT succeeds the top predicted token consistently IS the correct answer's first token —
"tea"->" tea", "king"->" king", "Nimbus"->" Nimbus", "Juniper"->" Jun" — i.e., the port isn't just
picking the right key by luck, the injected value then also drives the right logit. When SELECT
fails, the misses are not uniform noise: they cluster on a few specific wrong keys (e.g. "Zorbland",
"Maar Island", "Fenwick Mine", "Bryce war" cues all mis-hit onto the "Velk tribe" or "Quill Society"
keys) — a real, if imperfect, confusability structure in the bridged space, consistent with a
genuine (lossy) semantic map rather than the null hypothesis of noise.

## Grading against the pre-registration (E1-E10, header of `vector_telepathy.py`)

| # | prediction | measured | verdict |
|---|---|---|---|
| E1 | ridge-256 A->B centered cosine 0.40-0.75 | **0.7788** | just above range — bridge fit slightly stronger than predicted |
| E2 | cue-key diagnostic within ~0.10 of E1 | 0.1819 vs 0.7788 (diff 0.60) | **missed** — cue-specific keys sit far below the general held-out fit; the bridge generalizes to prose better than to this specific prefix-cue genre |
| E3 | A->B ridge-256 exact select 0.55-0.85; express ~= select x 0.85 | select **0.65** (in range); express 0.5 vs predicted ~0.55 | in range, express slightly under the ratio |
| E4 | A->B ridge-256 paraphrase select 0.35-0.65 | **0.5** | in range |
| E5 | Procrustes 0.05-0.25 BELOW ridge on select | procrustes **0.85** is ABOVE ridge256's 0.65 (+0.20) | **inverted** — answers the conditional literally: Procrustes matching/beating ridge means the A/B residual spaces are rotation-similar, not merely linearly-reachable through a fuller affine map |
| E6 | random select <=0.10; shuffled-fit select <=0.15 | random 0.05/0.0 (ab/ba); shuffled_fit 0.0/0.05 | **confirmed cleanly**, both directions |
| E7 | B->A within +-0.15 of A->B on select | ridge256: 0.65 vs 0.75 (diff 0.10); ridge512: 0.8 vs 0.8 (diff 0.0) | **confirmed** |
| E8 | ridge-512 beats ridge-256 by +0.02-0.10 on held-out cosine | A->B +0.0284; B->A +0.0270 | **confirmed**, both directions |
| E9 | ceiling B express ~0.75-0.85; A express ~0.90; paraphrase ceiling ~9/10 both | B=0.8, A=0.9, para 9/10 both | **confirmed, exactly** |
| E10 | PARTIAL-to-KILL of the impossibility claim | see verdict below | — |

## Verdict

**Applying the pre-set grading rule** (KILL if ridge express >= 70% of ceiling express AND both
nulls collapsed; CONFIRM if ridge select <= 2x random-null select OR centered bridge cosine < 0.35):

- Nulls collapsed cleanly in every arm, both directions (random <=0.05, shuffled-fit <=0.05 on
  select; centered held-out cosine <=0.01 in magnitude) — the CONFIRM branch's "collapsed nulls"
  precondition for the impossibility claim does NOT hold; the bridge is a real map, not noise.
- Centered bridge cosine is 0.72-0.87 across every non-null arm — nowhere close to <0.35, so the
  CONFIRM branch does not fire on that criterion either.
- KILL criterion on the PRIMARY pre-registered arm (ridge-256): A->B express 0.5/0.8 = **62.5%**
  (misses 70%); B->A express 0.7/0.9 = **77.8%** (clears 70%). Mixed on the exact letter of the rule.
- KILL criterion on the STRONGEST arm actually measured (ridge-512, run opportunistically since
  n_pool=512 >= 512): A->B 0.6/0.8 = **75%**; B->A 0.75/0.9 = **83.3%** — clears 70% both directions.
  Procrustes clears it too (A->B 0.7/0.8=87.5%, B->A 0.7/0.9=77.8%).

**Stated plainly: PARTIAL-to-KILL, leaning KILL, exactly as pre-registered in E10 — with an honest
asterisk that the primary pre-registered arm (ridge-256) undershoots the KILL bar on the harder
direction (A->B) while every richer bridge (ridge-512, Procrustes) clears it comfortably in both
directions.** The "vectors can't port" claim is FALSIFIED for the case tested: same-family,
same-tokenizer, key-only telepathy through a bridge fit on ~500 ordinary sentences recovers most of
the ceiling's recall (65-85% select, 50-75% express vs a ceiling of 80-90% express) while both
nulls are flat. Ported memory keys are not indistinguishable from noise, and they are not full
fidelity either — a real, measurable, majority-but-not-complete transfer.

**The most informative single number is E5's inversion.** Procrustes — a ROTATION-ONLY map through
a shared 192-dim PCA subspace, strictly less expressive than ridge's full affine map — matches or
BEATS ridge on select in both directions (A->B: 0.85 vs 0.65; B->A: 0.8 vs 0.75). The pre-
registration read this as diagnostic: "if Procrustes ~= ridge, the spaces are rotation-similar, not
just linearly reachable." That's what happened, more strongly than hypothesized. Two same-family
Qwen models trained independently at different scale, one in bf16 and one in nf4, converge on
residual geometries related by something close to a pure rotation+scale at this layer — a much
tighter geometric relationship than "there exists SOME linear map," and the more surprising result
of the whole rig.

## Caveats (loud, per house ethos)

- **One family (Qwen2.5), one layer (L18), one seed.** Cross-family (different tokenizer/vocab,
  where values would ALSO need to cross the bridge, not just keys) is untested and likely much
  harder — this result says nothing about whether telepathy works between architecturally distinct
  models.
- **n=20 exact / n=10 paraphrase is small.** Individual arm numbers move by 0.05-0.10 per single
  item; the qualitative pattern (nulls flat, ridge/Procrustes well above nulls but below ceiling,
  Procrustes >= ridge) is the trustworthy part, not the third decimal place.
- **Values never crossed the bridge** — shared vocab made that trivial (target model's own
  unembedding row for the shared token id). The telepathy question tested is purely "does a cue's
  MEANING as a direction survive a fitted map," which is the harder and more interesting half, but
  it is not the FULL memory-porting problem; a cross-vocab experiment would need to bridge values
  too and is a materially different (harder) rig.
- **The cue-key diagnostic (E2) is the one clean miss** — bridged cue-keys sit closer to the target
  model's own native key for a DIFFERENT cue than expected (cue_cos ~0.11-0.32 vs the general
  held-out fit's ~0.72-0.87). The bridge generalizes better to ordinary prose than to this specific
  short prefix-style cue genre, which is exactly the genre the store's real keys are drawn from —
  a caveat worth remembering if this ever became a product mechanism rather than a research probe.
- Smoke-scale (n=4) numbers were noisy in the expected direction (nulls occasionally spiked from
  small-n luck, e.g. smoke's `random para sel=0.5` on n=4) — resolved cleanly at full scale (n=20),
  underscoring why the house rule is smoke-then-full, not smoke-as-final-answer.
