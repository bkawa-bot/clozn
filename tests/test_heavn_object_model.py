"""Node-backed contracts for Studio's dependency-free shared object model."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "studio" / "heavn" / "object_model.mjs"
REPLAY = ROOT / "studio" / "heavn" / "modules" / "replay.mjs"
EXPERIMENT = ROOT / "studio" / "heavn" / "modules" / "experiment.mjs"
NODE = shutil.which("node")


def _node(script: str):
    assert NODE, "Node.js is required for Studio object-model tests"
    source = (
        f'import {{ normalizeRun, normalizeEvidence, normalizeExperiment }} '
        f'from {json.dumps(MODULE.as_uri())};\n{script}'
    )
    proc = subprocess.run(
        [NODE, "--input-type=module", "--eval", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_module_is_pure_dependency_free_and_exports_the_three_views():
    source = MODULE.read_text(encoding="utf-8")
    assert "export function normalizeRun" in source
    assert "export function normalizeEvidence" in source
    assert "export function normalizeExperiment" in source
    assert "fetch(" not in source
    assert "import " not in source
    assert "%" not in source
    assert "percentage" not in source.lower()
    assert "circuit" not in source.lower()


def test_run_and_experiment_homes_consume_the_shared_objects():
    replay = REPLAY.read_text(encoding="utf-8")
    experiment = EXPERIMENT.read_text(encoding="utf-8")
    assert 'import { normalizeRun } from "../object_model.mjs"' in replay
    assert "const run = normalizeRun(rec);" in replay
    assert 'import { normalizeExperiment } from "../object_model.mjs"' in experiment
    assert "const experiment = normalizeExperiment(res);" in experiment


def test_normalize_run_prefers_captured_prompt_and_preserves_lineage_and_raw():
    out = _node(r"""
const run = {
  id: "run-2", model: "qwen-local",
  messages: [{ role: "user", content: "raw question" }],
  assembled_messages: [
    { role: "system", content: "stored memory" },
    { role: "user", content: "assembled question" },
  ],
  final_prompt: "<exact rendered prompt>", response: "recorded answer",
  trace: { steps: [{ piece: "recorded", conf: 0.8 }] }, finish_reason: "stop",
  identity: { model_sha256: "abc", template_fingerprint: "tpl" },
  parent_run_id: "run-1", changes_applied: { greedy: true },
};
const envelope = { run, endpoint: "get-run" };
const before = JSON.stringify(envelope);
const value = normalizeRun(envelope);
if(value.raw !== run || value.envelope_raw !== envelope) throw new Error("raw run/envelope not preserved");
if(value.identity !== run.identity || value.captured_identity !== run.identity) throw new Error("identity copied or lost");
if(value.prompt.messages !== run.assembled_messages) throw new Error("assembled messages not preferred");
if(value.answer.trace !== run.trace) throw new Error("trace not preserved");
if(JSON.stringify(envelope) !== before) throw new Error("normalizer mutated its input");
console.log(JSON.stringify(value));
""")
    assert out["kind"] == "run"
    assert out["id"] == "run-2" and out["model"] == "qwen-local"
    assert out["prompt"]["source"] == "final_prompt"
    assert out["prompt"]["text"] == "<exact rendered prompt>"
    assert out["prompt"]["delivered_messages"][0]["content"] == "raw question"
    assert out["answer"]["text"] == "recorded answer"
    assert out["parent"] == {"id": "run-1", "changes": {"greedy": True}}


def test_normalize_run_tolerates_legacy_fields_without_fabricating_missing_values():
    out = _node(r"""
const legacy = {
  run_id: 7, model_name: "legacy-model", prompt: "legacy prompt",
  answer: { text: "legacy answer" }, captured_identity: { engine_build: "old" }, parent: "run-6",
};
console.log(JSON.stringify(normalizeRun(legacy)));
""")
    assert out["id"] == "7"
    assert out["model"] == "legacy-model"
    assert out["prompt"]["text"] == "legacy prompt"
    assert out["prompt"]["source"] == "prompt"
    assert out["answer"]["text"] == "legacy answer"
    assert out["parent_id"] == "run-6"
    assert out["answer"]["trace"] is None


def test_evidence_is_complete_ordered_and_preserves_explicit_status_controls_and_raw():
    out = _node(r"""
const influence = {
  schema: "clozn.context_answer_influence.v1", status: "ok",
  method: { name: "teacher_forced" }, controls: [{ kind: "matched_filler" }],
  timing: { total_ms: 12.5 }, artifact_sha256: "map-sha",
};
const nestedForced = {
  mode: "forced", causal_verified: true, answer_tokens: ["A"], deltas: [0.4],
  null_floor: { kind: "filler" },
};
const receipt = { mode: "regen", causal_verified: true, null: null, forced: nestedForced };
const directForced = { mode: "forced", causal_verified: false, answer_tokens: ["B"], deltas: [0.1] };
const workspace = { type: "workspace_readout", provider: "mock", position: 0 };
const jlens = { available: false, layer: 21, readouts: [], reason: "not fitted" };
const truth = {
  schema: "clozn.calibration_profile.v1", model: "qwen-local", task: "retrieval",
  provenance: { kind: "outcome_grounded_calibration" },
};
const trust = { available: false, reason: "no matching fit" };
const record = {
  id: "run-evidence", influence_map: influence,
  receipts: { receipts: [receipt], forced_receipts: [directForced] },
  trace: { workspace_readouts: [workspace] }, _jlens: jlens,
  truth_calibration: truth, trust,
};
const before = JSON.stringify(record);
const values = normalizeEvidence(record);
const expectedRaw = [influence, receipt, nestedForced, directForced, workspace, jlens, truth, trust];
if(values.some((value, index) => value.raw !== expectedRaw[index])) throw new Error("raw artifact not preserved");
if(JSON.stringify(record) !== before) throw new Error("normalizer mutated stored evidence");
console.log(JSON.stringify(values));
""")
    assert [item["evidence_type"] for item in out] == [
        "influence_map", "receipt", "forced_receipt", "forced_receipt",
        "workspace_readout", "jlens_readout", "truth_calibration", "trust",
    ]
    assert all(item["kind"] == "evidence" for item in out)
    assert [item["status"] for item in out] == [
        "ok", "verified", "verified", "unverified", None, "unavailable", None, "unavailable",
    ]
    assert out[0]["method"] == {"name": "teacher_forced"}
    assert out[0]["controls"] == [{"kind": "matched_filler"}]
    assert out[0]["latency"] == 12.5 and out[0]["artifact_hash"] == "map-sha"
    assert out[2]["controls"] == {"kind": "filler"}
    assert out[1]["controls"] is None  # explicit missing null remains missing
    assert all(item["source_run_id"] == "run-evidence" for item in out)


def test_legacy_evidence_does_not_invent_availability_controls_latency_or_hash():
    out = _node(r"""
const raw = { mode: "forced", answer_tokens: ["A"], deltas: [0.01] };
const values = normalizeEvidence(raw, "run-legacy");
if(values[0].raw !== raw) throw new Error("raw not preserved");
console.log(JSON.stringify(values[0]));
""")
    assert out["evidence_type"] == "forced_receipt"
    assert out["status"] is None
    assert out["method"] == "forced"
    assert out["controls"] is None
    assert out["latency"] is None
    assert out["artifact_hash"] is None
    assert out["source_run_id"] == "run-legacy"


def test_experiment_normalization_handles_current_and_flat_legacy_envelopes():
    out = _node(r"""
const receipt = { id: "child-2", parent_run_id: "source-1", causal_verified: true };
const current = {
  run_id: "source-1", change: { type: "ablate_memory" }, method: "receipt:both",
  baseline: { reply: "old" },
  result: { changed_reply: "new", causal_verified: true, null: null, receipt },
};
const before = JSON.stringify(current);
const normalized = normalizeExperiment(current);
if(normalized.raw !== current || normalized.result !== current.result || normalized.receipt !== receipt)
  throw new Error("current envelope artifacts not preserved");
if(JSON.stringify(current) !== before) throw new Error("experiment input mutated");
const legacy = normalizeExperiment({
  type: "reroll", source_id: "source-3", child_id: "child-4",
  baseline: "old legacy", result: "new legacy",
});
const failed = normalizeExperiment({
  __status: 503, change: { type: "swap_concept" }, error: "worker unavailable",
});
console.log(JSON.stringify({ normalized, legacy, failed }));
""")
    current = out["normalized"]
    assert current["kind"] == "experiment"
    assert current["type"] == "ablate_memory"
    assert current["method"] == "receipt:both"
    assert current["status"] == "verified"
    assert current["source_run_id"] == "source-1"
    assert current["child_run_id"] == "child-2"
    assert current["baseline"] == {"reply": "old"}
    assert current["result"]["changed_reply"] == "new"

    legacy = out["legacy"]
    assert legacy["type"] == "reroll"
    assert legacy["status"] is None
    assert legacy["source_run_id"] == "source-3" and legacy["child_run_id"] == "child-4"
    assert legacy["baseline"] == "old legacy" and legacy["result"] == "new legacy"
    assert legacy["receipt"] is None
    assert out["failed"]["status"] == "error"
