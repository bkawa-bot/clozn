# Worked model-diff case: Qwen2.5-0.5B-Instruct → Reasoning-0.5b

This is the Phase-1 acceptance run for `clozn diff-model` and the experiment-to-CI path: a real public
fine-tune pair, not two renamed copies or synthetic fixtures. It asks two separate questions:

1. Can Clozn prove that the SFT is not a silent no-op and localize next-token behavior changes?
2. Can a target + guard experiment catch both a concrete gain and a concrete regression, then fail CI?

It does not make a general claim that the candidate is better.

## Artifacts and lineage

| Role | Artifact | Evidence |
|---|---|---|
| Reference | [bartowski/Qwen2.5-0.5B-Instruct-GGUF, Q4_K_M](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf) | 397,808,192 bytes; SHA-256 `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| Candidate | [bartowski/Reasoning-0.5b-GGUF, Q4_K_M](https://huggingface.co/bartowski/Reasoning-0.5b-GGUF/blob/3b502edf3245f8726a18389c57837ce38f704600/Reasoning-0.5b-Q4_K_M.gguf) | 397,806,080 bytes; SHA-256 `d64ca01789ee360bfb45f21b59db1e961de71d247034504d15e267ef8bb2b610` |
| Candidate source | [KingNish/Reasoning-0.5b](https://huggingface.co/KingNish/Reasoning-0.5b) | Model metadata identifies `Qwen/Qwen2.5-0.5B-Instruct` as the base and labels the training as SFT on a reasoning dataset; Apache-2.0 |

The two downloads total 795,614,272 bytes (about 759 MiB). The run used the CPU engine on a MacBook Air
with an Apple M5 and 16 GB unified memory on 2026-07-20. Both engines were resident concurrently.

## Reproduction

The files were downloaded from the revision-pinned links above and checked with `shasum -a 256`. From
the Clozn repository:

```bash
clozn diff-model \
  ~/.clozn/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  ~/.clozn/models/Reasoning-0.5b-Q4_K_M.gguf \
  --runs 8 --max-tokens 128 --both --cpu
```

The run was repeated with `--own-templates`. Clozn's engine rendered the canonical template probe
identically for these two GGUFs, so the deployed-template and reference-template runs produced identical
metrics. This matters: the observed difference is not being attributed to a template mismatch.

## Preconditions passed

- All four tokenizer probes matched in both token IDs and token pieces: English, arithmetic/digits,
  source code, and multilingual Unicode.
- The canonical two-message chat rendering matched byte-for-byte.
- All 16 ladders completed: eight reference-anchored and eight candidate-anchored, with no skipped runs
  and no unknown flip states.

## Results

| Direction | Verified runs | Tokens | Preserved | Argmax flips | Mean absolute Δ nats | Heuristic verdict |
|---|---:|---:|---:|---:|---:|---|
| Reference-anchored | 8/8 | 519 | 90.0% | 52 | 0.181779 | `CHANGED` |
| Candidate-anchored | 8/8 | 576 | 88.0% | 69 | 0.172663 | `CHANGED` |

The candidate-anchored flips concentrated in reasoning (21), code (18), refusal (17), arithmetic (11),
and list formatting (2); factual QA and JSON formatting had none on this small sample. In the reverse
direction, the largest counts were factual QA (17), reasoning (11), code (11), and arithmetic (10).

The command's `CHANGED` screen requires at least 100 verified tokens. A candidate is labeled
`NO_DETECTABLE_DIFF` only when it has zero argmax flips and mean absolute Δ nats below 0.02. Both
directions cleared the sample floor and exceeded both change signals by a wide margin. This is strong
evidence that the candidate weights were actually applied; it is not evidence that every change is good.

## Target + guard experiment

The checked-in [experiment manifest](../examples/qwen-reasoning-sft-experiment.v0.json) uses the same
files behind two local gateways. In separate terminals:

```bash
clozn serve ~/.clozn/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 18081 --ctx 2048 --cpu
clozn serve ~/.clozn/models/Reasoning-0.5b-Q4_K_M.gguf --port 18082 --ctx 2048 --cpu
clozn experiment run examples/qwen-reasoning-sft-experiment.v0.json --out result.json
```

The manifest deliberately uses externally checkable assertions rather than asking either model to grade
itself. With greedy decoding and seed 0, all six cells ran and retained stable model identity:

| Variant | Target | Guard | Observed change |
|---|---:|---:|---|
| Base | 0/1 | 2/2 | Baseline |
| Reasoning SFT | 1/1 | 1/2 | One target gain; one guard regression |

The gain was a simple syllogism. The base answered “Yes” and invented meanings for the nonce words; the
SFT answered “No.” The regression was structured output. The base returned a fenced JSON object with the
requested `answer` and `units` fields, while the SFT returned prose: “The answer is 120 minutes.” Both
models preserved the capital-of-France guard.

The strict CI policy failed exactly where intended:

```bash
clozn ci check --experiment result.json \
  --min-target-gains 1 --max-guard-regressions 0 --require-run-identity
# exit 1: structured-json regressed (pass -> fail)
```

Changing the explicit budget to `--max-guard-regressions 1` returned exit 0. This proves the artifact is
not just a report: a pipeline can reject the candidate, or consciously accept a known regression, from
the same raw case × variant × seed evidence.

## What the receipt means—and does not mean

Each flip is a one-step counterfactual under an identical teacher-forced prefix: the reference and
candidate had different top-1 next-token choices at that position. It does **not** claim the complete
free-running answers would diverge at exactly that point, and a non-flip probability shift is not an
answer change. The built-in eight-prompt ladder and the three-case experiment are smoke screens, not
benchmarks of reasoning quality. The source model also documents a two-stage inference flow using
`add_reasoning_prompt` followed by a `reasoning` role; Clozn's ordinary GGUF chat path does not reproduce
that special workflow. These measurements describe the candidate as it actually runs through Clozn's
standard compatibility path, not the full advertised two-pass recipe. Separate GGUF conversions are
another small confound even at the same quantization label.

Together, the tools keep their claims clean: `diff-model` answers “did the weights change?”; the
target/guard experiment answers “did our declared outcomes improve, and which declared guarantees
broke?” This case found both a real gain and a release-blocking regression without pretending three
prompts establish overall model quality.
