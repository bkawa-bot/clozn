from __future__ import annotations

import json

import pytest

from clozn.cli.commands import qualify_chat_io as q
from clozn.runs.identity import template_fingerprint
from clozn.server.structured_io import NATIVE_WORKER_PIPELINE, QUALIFICATION_SCHEMA


class _Client:
    def __init__(self, *, pipeline=None):
        self.pipeline = dict(pipeline or NATIVE_WORKER_PIPELINE)
        self.calls = 0

    def health(self):
        return {"model_sha256": "a" * 64, "native_chat_io": {
            "available": True, "atomic": True, "buffered": True, **self.pipeline}}

    def apply_template(self, messages):
        return json.dumps(messages, sort_keys=True)

    def complete_chat(self, messages, **kwargs):
        outputs = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "lookup_weather", "arguments": '{"city":"Paris"}'}}]},
            {"role": "assistant", "content": "It is 21 C."},
            {"role": "assistant", "content": '{"ok":true}'},
            {"role": "assistant", "content": '{"city":"Paris","temperature_c":21}'},
        ]
        message = outputs[self.calls]
        self.calls += 1
        return {"chat_io": {"model_sha256": "a" * 64,
                            "pipeline": dict(NATIVE_WORKER_PIPELINE), "message": message}}


def test_fixed_battery_covers_tools_continuation_and_json_modes():
    client = _Client()
    cases, passed = q.run_qualification(
        client, model_sha256="a" * 64,
        template_sha256=template_fingerprint(client.apply_template))
    assert set(cases) == {"tool_call", "tool_result_continuation", "json_object", "json_schema"}
    assert passed == {"pipeline": 1, "tools": 2, "json_object": 1, "json_schema": 1}


def test_pipeline_drift_fails_before_generation():
    client = _Client(pipeline={**NATIVE_WORKER_PIPELINE, "parser_id": "drifted"})
    with pytest.raises(ValueError, match="pipeline mismatch"):
        q.run_qualification(client, model_sha256="a" * 64,
                            template_sha256=template_fingerprint(client.apply_template))
    assert client.calls == 0


def test_emits_contract_valid_artifact_and_registry(tmp_path):
    identity = {"sha256": "a" * 64, "architecture": "fixture", "hidden_size": 8,
                "layer_count": 2, "vocab_size": 16, "tokenizer_sha256": "b" * 64}
    artifact = tmp_path / "artifact"
    registry = tmp_path / "registry.json"
    q.emit_qualification(
        identity=identity, fingerprint="c" * 16,
        cases={"tool_call": "tool_call", "tool_result_continuation": "message",
               "json_object": "json_object", "json_schema": "json_schema"},
        passed={"pipeline": 1, "tools": 2, "json_object": 1, "json_schema": 1},
        artifact_dir=artifact, registry_path=registry, source_id="fixture")
    assert (artifact / "manifest.json").is_file()
    assert json.loads(registry.read_text())["schema_version"] == QUALIFICATION_SCHEMA
