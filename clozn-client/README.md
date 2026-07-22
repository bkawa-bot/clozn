# clozn-client

Typed, synchronous Python clients for Clozn research workflows. The package deliberately exposes
two separate clients because the two URLs have different contracts:

- `CloznClient` targets the **public product gateway** and reads runs, timelines, diagnoses,
  receipt bundles, explanations, and causal evidence.
- `EngineClient` targets an explicitly started **private/native engine** and performs
  teacher-forced scoring, steering, and attention-knockout scoring.

A gateway URL is never silently treated as an engine URL. In a normal `clozn serve` process the
worker is private and supervised on a random loopback port; direct engine research calls require a
worker URL you intentionally started or otherwise explicitly obtained.

## Install from the repository

```bash
python -m pip install ./clozn-client
```

Gateway and scoring operations are standard-library-only. Activation arrays are optional:

```bash
python -m pip install './clozn-client[arrays]'
```

## Inspect a recorded run through the gateway

```python
from clozn_client import CloznClient

clozn = CloznClient("http://127.0.0.1:8080")
run = clozn.run("clozn_run_id")
print(run.response)

for event in clozn.timeline(run.id).events:
    print(event.get("kind"), event)

bundle = clozn.export_receipt(run.id)
print(bundle.raw["schema"])
```

Exact client/session association selectors can be supplied either per lookup or as constructor
headers:

```python
clozn = CloznClient(
    "http://127.0.0.1:8080",
    client_id="my-local-tool",
    session_id="experiment-17",
)
latest = clozn.latest_run()
```

## Score directly on a native engine

```python
from clozn_client import AttentionKnockout, EngineClient

engine = EngineClient("http://127.0.0.1:8091")
baseline = engine.score(
    prompt="Context: Paris is in France.\nAnswer:",
    continuation=" Paris",
    topk=3,
)

cut = engine.knockout_score(
    prompt="Context: Paris is in France.\nAnswer:",
    continuation_ids=[baseline.tokens[0].id],
    knockouts=[
        AttentionKnockout(
            layer=0,
            queries=(8,),
            keys=(2, 3, 4),
            renormalize=True,
        )
    ],
)
print(baseline.sum_logprob, cut.sum_logprob)
```

`renormalize=True` is the default because removing attention mass without renormalizing also
changes the amplitude of the attention output. The client transmits the intervention
specification; it does not label the result as causal proof by itself. Controls, model identity,
and artifact qualification remain the researcher's responsibility.

## Save and replay an intervention experiment

A manifest contains one immutable teacher-forced request and named intervention arms. Replay runs
an automatic no-intervention baseline first, then each arm in stable order.

```python
from clozn_client import (
    AttentionKnockout,
    EngineClient,
    InterventionArm,
    InterventionManifest,
    ScoreRequest,
)

manifest = InterventionManifest(
    name="capital-source-knockout",
    request=ScoreRequest(
        prompt="Context: Paris is in France.\nAnswer:",
        continuation=" Paris",
        topk=3,
    ),
    expected_health={"capabilities": {"attn_knockout": True}},
    arms=(
        InterventionArm(
            name="cut-context-span",
            attention_knockout=(
                AttentionKnockout(layer=0, queries=(8,), keys=(2, 3, 4)),
            ),
            metadata={"role": "candidate"},
        ),
    ),
)
manifest.save("capital-knockout.json")

result = EngineClient("http://127.0.0.1:8091").replay_manifest(manifest)
print(manifest.sha256)
print(result.arms[0].support_drop)
```

The manifest schema is `clozn.intervention_manifest.v1`. Its SHA-256 is computed from canonical
JSON, so key ordering and pretty-printing do not change its identity. `expected_health` is checked
before any scoring call; missing or mismatched fields fail closed. Positive `support_drop` means
the intervention lowered the recorded continuation's summed log-probability relative to baseline.
It is a measured effect, not an automatic provenance verdict.

See [`examples/`](examples/) for run inspection, saved-manifest replay, and a candidate-vs-control
knockout scan.

## Supported slice

### Gateway

- readiness and run listing
- exact/latest/cursor-based run lookup
- timeline, diagnosis, lineage, family, and confidence spans
- JSON or Markdown receipt export
- free explanation and explicit causal receipt calls

### Native engine

- health/capabilities
- greedy/native completion helper
- teacher-forced `/score`
- steering vector/config passthrough for `/score`
- teacher-forced `/score` with version-current `attn_knockout` specifications
- canonical versioned manifests with expected-health checks and stable baseline/arm replay

Activation research is available through the optional NumPy extra:

- exact float32 tensor decoding for `/harvest`
- position-major residual writes through `/state`
- named patch sweeps that reuse one harvest and preserve its actual layer

The client still does not standardize broader hook names until the engine exposes and tests their exact semantics.


## Harvest and patch residuals

```python
import numpy as np
from clozn_client import EngineClient, PatchArm

engine = EngineClient("http://127.0.0.1:8091")
h = engine.harvest("The capital of France is")
pos = h.n_tokens - 1

sweep = engine.patch_sweep(
    "The capital of France is",
    (
        PatchArm("identity", (pos,), h.activations[[pos]]),
        PatchArm("amplify", (pos,), h.activations[[pos]] * np.float32(1.25)),
    ),
    layer=h.layer,
)
for arm in sweep.arms:
    print(arm.name, arm.observation.moved_l2, arm.observation.shifted)
```

`patch_sweep()` harvests once and writes every arm at the layer actually returned by the engine.
It does not infer causal meaning from movement; identity and magnitude-matched controls remain explicit arms.

## Portable patch sweeps

Activation edits can be stored in `clozn.patch_sweep_manifest.v1` files with inline
position-major float32 rows, optional engine-health expectations, deterministic SHA-256
identities, and JSON result artifacts:

```python
from clozn_client import EngineClient, PatchManifestArm, PatchSweepManifest

manifest = PatchSweepManifest(
    name="identity-and-amplify",
    text="The capital is",
    layer=4,
    arms=(
        PatchManifestArm("identity", (0,), ((1.0, 2.0),)),
        PatchManifestArm("amplify", (0,), ((2.0, 4.0),)),
    ),
)
artifact = EngineClient().run_patch_manifest(manifest)
artifact.write("patch-result.json")
```

## Command line

Validate either supported manifest schema without contacting an engine:

```bash
python -m clozn_client validate experiment.manifest.json
```

Replay either scoring or activation manifests and write a portable result:

```bash
python -m clozn_client replay experiment.manifest.json \
  --engine-url http://127.0.0.1:8091 \
  --output result.json
```

## Batch replay

Mixed scoring and activation manifests can be replayed deterministically into a result
directory. Each result is named by the manifest SHA-256, and `index.json` records successes
and isolated failures:

```bash
python -m clozn_client batch experiments/ \
  --engine-url http://127.0.0.1:8091 \
  --output-dir results/
```

Use `--fail-fast` when a CI job should stop after the first engine or contract failure.

## Compare batch runs in CI

`clozn-client` can compare two `clozn.batch_run.v1` indexes without contacting an engine.
Numeric increases beyond the configured tolerance are regressions; a new `shifted=true`, missing
manifest, changed failure, arm, or result schema is also a regression.

```bash
python -m clozn_client compare baseline/index.json candidate/index.json \
  --max-metric-delta 0.05 --output comparison.json
```

The command exits `0` when the gate passes, `1` when regressions are found, and `2` for invalid
inputs or operational errors. The portable output schema is `clozn.batch_comparison.v1`.

### CI-native regression reports

The comparison command can emit CI reports alongside its canonical JSON artifact:

```bash
python -m clozn_client compare baseline/index.json candidate/index.json \
  --max-metric-delta 0.05 \
  --output comparison.json \
  --junit comparison.junit.xml \
  --github-summary "$GITHUB_STEP_SUMMARY" \
  --github-annotations comparison.annotations.txt
```

`--junit` writes one testcase per experiment. `--github-summary` writes a compact
Markdown scorecard and regression details. `--github-annotations` writes GitHub
Actions workflow commands that can be emitted with `cat` in a later step.

## Capture reproducibility provenance

Record the exact client, Python runtime, platform, native engine health payload, and
manifest identities used by an experiment run. The record is content-addressed; its
SHA-256 excludes only the capture timestamp, so equivalent environments retain the
same identity.

```bash
python -m clozn_client provenance experiments/*.manifest.json \
  --engine-url http://127.0.0.1:8091 \
  --metadata git_sha="$(git rev-parse HEAD)" \
  --output provenance.json
```

Loading a record verifies its embedded digest and rejects tampering.

## Minimal intervention contract

`clozn-client` publishes `clozn.intervention_contract.v1`, a deliberately small vocabulary for
operations consumed by Clozn receipts. It is not a general hook registry:

- `score.teacher_forced`
- `intervention.attention_knockout`
- `capture.residual.layer_output`
- `intervention.residual.replace_rows`

Each operation records its endpoint, exact semantics, tensor shape, replay class, qualification
requirement, and limits. Generic steering remains usable through legacy scoring manifests but is
not promoted into the stable contract until a receipt workflow and qualified native semantics
require it.

```python
from clozn_client import EngineClient, InterventionOperation

engine = EngineClient("http://127.0.0.1:8091")
report = engine.contract_report(
    InterventionOperation.SCORE_TEACHER_FORCED,
    InterventionOperation.ATTENTION_KNOCKOUT,
)
report.require_compatible()
```

Passing a manifest derives only the stable operations it actually requires:

```python
report = engine.contract_report(manifest)
```

A successful check establishes endpoint/capability compatibility. It does not by itself claim
that a replay is bit-identical or that a model/artifact has passed scientific qualification.

## Receipt-bound capture budgets

Residual captures use a conservative client-side `CaptureBudget` by default. Completed patch
artifacts embed `clozn.intervention_contract_binding.v1` plus capture shape/storage statistics, so
portable evidence names the contract and replay class that produced it. This is a receipt safety
boundary, not a general tensor-management API.

## Native protocol advertisement

Contract checks fail closed unless `GET /health` explicitly advertises
`clozn.engine_protocol.v1`, protocol version `1.0`, and the exact SHA-256 of the
client's `clozn.intervention_contract.v1`. Endpoint presence alone is reported as
`unadvertised`, not compatible.

```python
report = EngineClient().contract_report(manifest)
report.require_compatible()
```

This is deliberately stricter than feature probing: receipts should not claim
qualified semantics merely because an endpoint happened to respond.
