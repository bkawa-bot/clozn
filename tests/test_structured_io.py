"""Model-free contract tests for Phase 2.8's fail-closed structured-I/O core."""
from __future__ import annotations

import json

import pytest

from clozn.server import structured_io as sio


def test_native_pipeline_ids_match_cpp_worker_constants():
    import re
    from pathlib import Path

    header = (Path(__file__).resolve().parents[1] / "engine" / "core" / "serve" /
              "chat_template_renderer.hpp").read_text(encoding="utf-8")
    expected = {
        "NATIVE_CHAT_EXECUTOR_ID": sio.NATIVE_EXECUTOR_ID,
        "NATIVE_CHAT_RENDERER_ID": sio.NATIVE_RENDERER_ID,
        "NATIVE_CHAT_GRAMMAR_ID": sio.NATIVE_GRAMMAR_ID,
        "NATIVE_CHAT_PARSER_ID": sio.NATIVE_PARSER_ID,
    }
    for name, value in expected.items():
        match = re.search(rf'{name}\s*=\s*\n?\s*"([^"]+)"', header)
        assert match, f"missing C++ pipeline constant {name}"
        assert match.group(1) == value


def _schema():
    return {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
        },
        "required": ["city"],
        "additionalProperties": False,
    }


def _tool(name="weather"):
    return {
        "type": "function",
        "function": {"name": name, "description": "Get weather", "parameters": _schema()},
    }


def _tools_contract(**extra):
    return sio.normalize_contract({"tools": [_tool()], "parallel_tool_calls": False, **extra})


def _registry(features=None):
    features = features or ["tools", "json_object", "json_schema"]
    schema_subsets = {}
    if "tools" in features:
        schema_subsets["tool_parameters"] = sio.JSON_SCHEMA_SUBSET_ID
    if "json_schema" in features:
        schema_subsets["json_schema"] = sio.JSON_SCHEMA_SUBSET_ID
    return {
        "schema_version": sio.QUALIFICATION_SCHEMA,
        "entries": [{
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16,
            "features": features,
            "schema_subsets": schema_subsets,
            "pipeline": dict(sio.NATIVE_QUALIFICATION_PIPELINE),
            "evidence": {
                "schema_version": sio.QUALIFICATION_EVIDENCE_SCHEMA,
                "suite_id": sio.QUALIFICATION_SUITE_ID,
                "artifact_version": 2,
                "payload_sha256": "c" * 64,
            },
        }],
    }


def test_normalize_tools_is_exact_strict_and_single_call():
    contract = _tools_contract()
    assert contract["mode"] == "tools"
    assert contract["tool_choice"] == "auto"
    fn = contract["active_tools"][0]["function"]
    assert fn["strict"] is True
    assert fn["parameters"]["additionalProperties"] is False
    assert len(sio.normalize_contract({"tools": [_tool(), _tool("forecast")]})["active_tools"]) == 2

    for body, param in [
        ({"tools": [_tool()], "tool_choice": "required"}, "tool_choice"),
        ({"tools": [_tool()], "parallel_tool_calls": True}, "parallel_tool_calls"),
        ({"tools": [_tool(), _tool()]}, "tools"),
    ]:
        with pytest.raises(sio.StructuredIOError) as caught:
            sio.normalize_contract(body)
        assert caught.value.param == param


def test_tool_choice_none_is_an_explicit_text_bypass():
    contract = sio.normalize_contract({"tools": [_tool()], "tool_choice": "none",
                                       "parallel_tool_calls": True})
    assert contract["mode"] is None
    assert contract["active_tools"] == []


def test_tool_definition_unknown_fields_and_non_strict_fail():
    bad = _tool()
    bad["function"]["strict"] = False
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_contract({"tools": [bad]})
    assert caught.value.param == "tools[0].function.strict"

    bad = _tool()
    bad["function"]["magic"] = 1
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_contract({"tools": [bad]})
    assert caught.value.code == "unsupported_parameter"
    assert caught.value.param.endswith(".magic")


def test_json_object_and_strict_json_schema_contracts():
    obj = sio.normalize_contract({"response_format": {"type": "json_object"}})
    assert obj["mode"] == "json_object"

    schema = sio.normalize_contract({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": _schema()},
    }})
    assert schema["mode"] == "json_schema"
    assert schema["response_format"]["json_schema"]["schema"]["required"] == ["city"]

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_contract({"tools": [_tool()],
                                "response_format": {"type": "json_object"}})
    assert caught.value.param == "response_format"


def test_schema_subset_rejects_unknown_keywords_and_non_strict_objects():
    bad = _schema()
    bad["properties"]["city"]["pattern"] = ".*"
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_json_schema(bad)
    assert caught.value.code == "unsupported_schema_keyword"
    assert caught.value.param.endswith(".pattern")

    bad = _schema()
    bad["additionalProperties"] = True
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_json_schema(bad)
    assert caught.value.param.endswith(".additionalProperties")


def test_schema_instance_validation_is_typed_and_does_not_treat_bool_as_integer():
    schema = sio.normalize_json_schema(_schema())
    sio.validate_instance({"city": "Paris", "days": 2}, schema)
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_instance({"city": "", "days": True, "extra": 1}, schema)
    assert caught.value.code == "schema_validation_failed"
    paths = {item["path"] for item in caught.value.evidence["errors"]}
    assert {"$.city", "$.days", "$.extra"}.issubset(paths)


def test_tool_history_is_preserved_for_journal_and_lowered_to_role_text():
    contract = _tools_contract()
    history = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "user", "content": "Summarize."},
    ]
    plan = sio.render_messages(history, contract)
    assert plan["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'
    assert plan["messages"][2]["role"] == "tool"
    assert plan["generation_messages"][0]["role"] == "system"
    assert plan["generation_messages"][2]["role"] == "assistant"
    assert plan["generation_messages"][3]["role"] == "user"
    assert "tool_result" in plan["generation_messages"][3]["content"]
    assert sio.PYTHON_ENVELOPE_RENDERER_ID in plan["renderer_instruction"]
    assert plan["renderer_instruction"].find(sio.NATIVE_RENDERER_ID) == -1


def test_tool_history_rejects_orphans_mismatch_and_parallel_calls():
    contract = _tools_contract()
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_and_lower_messages(
            [{"role": "tool", "tool_call_id": "call_missing", "content": "x"}], contract
        )
    assert caught.value.param == "messages[0].tool_call_id"

    call = {"id": "call_1", "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_and_lower_messages(
            [{"role": "assistant", "content": None, "tool_calls": [call, call]}], contract
        )
    assert caught.value.code == "unsupported_parameter"

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_and_lower_messages(
            [{"role": "assistant", "content": None, "tool_calls": [call]}], contract
        )
    assert caught.value.param == "messages"

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_and_lower_messages([
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "user", "content": "skip the result"},
            {"role": "tool", "tool_call_id": "call_1", "content": "late"},
        ], contract)
    assert caught.value.param == "messages[1].role"


def test_tool_history_can_continue_to_text_without_redeclaring_tools():
    contract = sio.normalize_contract({})
    plan = sio.normalize_and_lower_messages([
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ], contract)
    assert plan["messages"][1]["role"] == "tool"
    assert "tool_result" in plan["generation_messages"][1]["content"]


def test_default_qualification_is_empty_and_exact_identity_is_required():
    identity = {"model_sha256": "a" * 64, "template_fingerprint": "b" * 16}
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.require_qualification(identity, "tools", environ={})
    assert caught.value.code == "model_not_qualified"
    assert caught.value.evidence["reason"] == "identity_not_qualified"

    entry = sio.require_qualification(
        identity, "tools", registry=_registry(),
        runtime_pipeline=sio.NATIVE_WORKER_PIPELINE,
    )
    assert entry["pipeline"] == sio.NATIVE_QUALIFICATION_PIPELINE

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.require_qualification({**identity, "template_fingerprint": "c" * 16},
                                  "tools", registry=_registry())
    assert caught.value.evidence["reason"] == "identity_not_qualified"


def test_registry_loads_only_from_explicit_environment_path(tmp_path):
    path = tmp_path / "qualifications.json"
    path.write_text(json.dumps(_registry()), encoding="utf-8")
    loaded = sio.load_qualification_registry(environ={sio.QUALIFICATIONS_ENV: str(path)})
    assert loaded["entries"][0]["model_sha256"] == "a" * 64

    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.load_qualification_registry(environ={sio.QUALIFICATIONS_ENV: str(path)})
    assert caught.value.code == "qualification_registry_invalid"


def test_v2_registry_rejects_python_envelope_and_requires_exact_runtime_pipeline():
    legacy = _registry()
    legacy["schema_version"] = "clozn.structured_io.qualifications.v1"
    legacy["entries"][0].pop("pipeline")
    legacy["entries"][0]["renderer_id"] = sio.PYTHON_ENVELOPE_RENDERER_ID
    legacy["entries"][0]["parser_id"] = sio.PYTHON_ENVELOPE_PARSER_ID
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_qualification_registry(legacy)
    assert caught.value.code == "qualification_registry_invalid"

    identity = {"model_sha256": "a" * 64, "template_fingerprint": "b" * 16}
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.require_qualification(identity, "tools", registry=_registry())
    assert caught.value.evidence["reason"] == "missing_runtime_pipeline"

    drifted = dict(sio.NATIVE_WORKER_PIPELINE)
    drifted["grammar_id"] += ".drift"
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.require_qualification(
            identity, "tools", registry=_registry(), runtime_pipeline=drifted,
        )
    assert caught.value.evidence["reason"] == "runtime_pipeline_mismatch"
    assert caught.value.evidence["field"] == "grammar_id"

    identity["chat_io_pipeline"] = dict(sio.NATIVE_WORKER_PIPELINE)
    entry = sio.require_qualification(identity, "tools", registry=_registry())
    assert entry["pipeline"]["validator_id"] == sio.NATIVE_MESSAGE_VALIDATOR_ID


def test_parser_accepts_message_and_valid_tool_call_and_serializes_openai_shapes():
    contract = _tools_contract()
    message = sio.parse_output('{"type":"message","content":"No tool needed."}', contract)
    assert sio.serialize_openai_result(message) == {
        "message": {"role": "assistant", "content": "No tool needed."},
        "finish_reason": "stop",
    }

    call = sio.parse_output(
        '{"type":"tool_call","name":"weather","arguments":{"city":"Paris","days":2}}',
        contract,
    )
    wire = sio.serialize_openai_result(call, call_id="call_test")
    assert wire["finish_reason"] == "tool_calls"
    assert wire["message"]["content"] is None
    assert wire["message"]["tool_calls"][0]["function"] == {
        "name": "weather", "arguments": '{"city":"Paris","days":2}'
    }
    stream = sio.openai_stream_deltas(call, call_id="call_test")
    assert stream["deltas"][0] == {"role": "assistant"}
    assert stream["deltas"][1]["tool_calls"][0]["index"] == 0


def test_native_message_validator_rechecks_tool_name_arguments_and_schema():
    contract = _tools_contract()
    native = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should check the weather.",
        "tool_calls": [{
            "id": "native_ignored",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": '{"days":2,"city":"Paris"}',
            },
        }],
    }
    parsed = sio.validate_native_message(native, contract)
    assert parsed["kind"] == "tool_call"
    assert parsed["arguments_json"] == '{"city":"Paris","days":2}'
    assert parsed["evidence"]["validator_id"] == sio.NATIVE_MESSAGE_VALIDATOR_ID
    assert sio.serialize_openai_result(parsed, call_id="call_public")["message"][
        "tool_calls"
    ][0]["id"] == "call_public"

    unknown = json.loads(json.dumps(native))
    unknown["tool_calls"][0]["function"]["name"] = "undeclared"
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_native_message(unknown, contract)
    assert caught.value.code == "unknown_tool"

    invalid = json.loads(json.dumps(native))
    invalid["tool_calls"][0]["function"]["arguments"] = '{"city":"Paris","days":9}'
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_native_message(invalid, contract)
    assert caught.value.code == "schema_validation_failed"
    assert caught.value.evidence["tool"] == "weather"

    duplicate = json.loads(json.dumps(native))
    duplicate["tool_calls"][0]["function"]["arguments"] = (
        '{"city":"Paris","city":"Lyon"}'
    )
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_native_message(duplicate, contract)
    assert caught.value.code == "malformed_model_output"


def test_native_message_validator_rejects_multiple_mixed_and_expanded_tool_shapes():
    contract = _tools_contract()
    call = {
        "id": "native_1", "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
    }
    for native in [
        {"role": "assistant", "content": None, "tool_calls": [call, call]},
        {"role": "assistant", "content": "also text", "tool_calls": [call]},
        {"role": "assistant", "content": None, "tool_calls": [{**call, "future": True}]},
        {"role": "assistant", "content": "ok", "refusal": None},
        {"role": "tool", "content": "not an assistant"},
    ]:
        with pytest.raises(sio.StructuredIOError) as caught:
            sio.validate_native_message(native, contract)
        assert caught.value.code == "malformed_model_output"


def test_native_message_validator_strictly_validates_json_modes_without_repair():
    object_contract = sio.normalize_contract({
        "response_format": {"type": "json_object"}
    })
    parsed = sio.validate_native_message(
        {"role": "assistant", "content": '{"value":2}'}, object_contract
    )
    assert parsed["content"] == '{"value":2}'

    for content in ('```json\n{"value":2}\n```', '[1,2]', '{"value":1e400}'):
        with pytest.raises(sio.StructuredIOError) as caught:
            sio.validate_native_message(
                {"role": "assistant", "content": content}, object_contract
            )
        assert caught.value.code == "malformed_model_output"

    schema_contract = sio.normalize_contract({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": _schema()},
    }})
    valid = sio.validate_native_message(
        {"role": "assistant", "content": '{"city":"Rome","days":3}'},
        schema_contract,
    )
    assert valid["kind"] == "json_schema"
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_native_message(
            {"role": "assistant", "content": '{"city":"Rome","extra":true}'},
            schema_contract,
        )
    assert caught.value.code == "schema_validation_failed"

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_native_message({
            "role": "assistant", "content": '{"city":"Rome"}',
            "tool_calls": [{
                "type": "function",
                "function": {"name": "weather", "arguments": "{}"},
            }],
        }, schema_contract)
    assert caught.value.code == "malformed_model_output"


def test_parser_only_recovers_one_outer_lowercase_json_fence():
    contract = _tools_contract()
    parsed = sio.parse_output(
        '```json\n{"type":"message","content":"ok"}\n```', contract
    )
    assert parsed["evidence"]["normalization"] == "strip_single_json_fence"

    for raw in [
        'prefix {"type":"message","content":"ok"}',
        '```\n{"type":"message","content":"ok"}\n```',
        '```JSON\n{"type":"message","content":"ok"}\n```',
        '{"type":"message","type":"tool_call","name":"weather","arguments":{}}',
    ]:
        with pytest.raises(sio.StructuredIOError) as caught:
            sio.parse_output(raw, contract)
        assert caught.value.code == "malformed_model_output"


def test_think_hygiene_cannot_repair_names_or_argument_values_inside_json():
    raw_name = ('{"type":"tool_call","name":"<think>junk</think>weather",'
                '"arguments":{"city":"Paris"}}')
    sanitized_name = ('{"type":"tool_call","name":"weather",'
                      '"arguments":{"city":"Paris"}}')
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.structured_parser_input(raw_name, sanitized_name)
    assert caught.value.code == "malformed_model_output"

    raw_arg = ('{"type":"tool_call","name":"weather",'
               '"arguments":{"city":"<think>Oslo</think>Paris"}}')
    sanitized_arg = ('{"type":"tool_call","name":"weather",'
                     '"arguments":{"city":"Paris"}}')
    with pytest.raises(sio.StructuredIOError):
        sio.structured_parser_input(raw_arg, sanitized_arg)

    raw_prefix = '<think>consider weather</think>\n{"type":"message","content":"ok"}'
    parser_input, normalization = sio.structured_parser_input(
        raw_prefix, '\n{"type":"message","content":"ok"}'
    )
    assert parser_input == '{"type":"message","content":"ok"}'
    assert normalization == "strip_leading_think_blocks"


def test_parser_rejects_exponent_overflow_and_resource_failures_are_typed():
    contract = sio.normalize_contract({"response_format": {"type": "json_object"}})
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.parse_output('{"value":1e400}', contract)
    assert caught.value.code == "malformed_model_output"

    malformed_registry = _registry()
    malformed_registry["entries"][0]["features"] = [{"not": "a string"}]
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.validate_qualification_registry(malformed_registry)
    assert caught.value.code == "qualification_registry_invalid"


def test_tool_parser_never_invents_names_or_repairs_invalid_arguments():
    contract = _tools_contract()
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.parse_output('{"type":"tool_call","name":"other","arguments":{}}', contract)
    assert caught.value.code == "unknown_tool"

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.parse_output(
            '{"type":"tool_call","name":"weather","arguments":{"days":9}}', contract
        )
    assert caught.value.code == "schema_validation_failed"
    assert caught.value.as_error()["type"] == "model_output_error"
    assert caught.value.evidence["tool"] == "weather"


def test_json_schema_output_is_validated_and_canonicalized():
    contract = sio.normalize_contract({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": _schema()},
    }})
    parsed = sio.parse_output('{ "days": 2, "city": "Paris" }', contract)
    assert parsed["content"] == '{"city":"Paris","days":2}'

    with pytest.raises(sio.StructuredIOError) as caught:
        sio.parse_output('{"city":"Paris","extra":true}', contract)
    assert caught.value.code == "schema_validation_failed"
    assert caught.value.evidence["mode"] == "json_schema"


def test_typed_error_payload_keeps_param_code_and_evidence():
    with pytest.raises(sio.StructuredIOError) as caught:
        sio.normalize_contract({"tools": [_tool()], "parallel_tool_calls": True})
    payload = caught.value.as_error()
    assert payload["type"] == "invalid_request_error"
    assert payload["param"] == "parallel_tool_calls"
    assert payload["code"] == "unsupported_parameter"


def test_qualification_evidence_hashes_the_normalized_request_contract():
    contract = _tools_contract()
    entry = sio.require_qualification(
        {"model_sha256": "a" * 64, "template_fingerprint": "b" * 16},
        "tools", registry=_registry(), runtime_pipeline=sio.NATIVE_WORKER_PIPELINE,
    )
    evidence = sio.qualification_evidence(entry, "tools", contract)
    assert evidence["schema"] == sio.EVIDENCE_SCHEMA
    assert evidence["pipeline"] == sio.NATIVE_QUALIFICATION_PIPELINE
    assert evidence["schema_subsets"] == {
        "tool_parameters": sio.JSON_SCHEMA_SUBSET_ID,
        "json_schema": sio.JSON_SCHEMA_SUBSET_ID,
    }
    assert evidence["qualification_evidence"]["suite_id"] == sio.QUALIFICATION_SUITE_ID
    assert len(evidence["request_contract_sha256"]) == 64

    changed = _tools_contract()
    changed["active_tools"][0]["function"]["parameters"]["properties"]["city"]["minLength"] = 2
    changed_evidence = sio.qualification_evidence(entry, "tools", changed)
    assert changed_evidence["request_contract_sha256"] != evidence["request_contract_sha256"]
