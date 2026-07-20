"""test_validate_export -- clozn/cli/commands/validate_export.py (`clozn validate-export`, Phase-1 §4.5):
deployment-equivalence check v0, GGUF-side only.

Model-free / GPU-free throughout, mirroring tests/test_diff_model.py's / tests/test_artifact_contracts.py's
own discipline: no real GGUF bytes, no engine, no GPU. `contracts.gguf_identity` and
`validate_export._read_raw_metadata` are monkeypatched wherever `cmd_validate_export` needs them, exactly
like tests/test_artifact_contracts.py monkeypatches `contracts.gguf_header_from_path`; `run_known_answers_
check` (the LIVE two-engine path behind --known-answers) is monkeypatched exactly how tests/test_ci_check.py
monkeypatches `ci_check.run_diff_check`, and is never invoked for real by this suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.artifacts.contracts as contracts  # noqa: E402
import clozn.cli.commands.validate_export as ve  # noqa: E402
from clozn.cli.main import CloznError  # noqa: E402


# ==================================================================================== fixture identities

def _identity(**overrides) -> dict:
    base = {
        "name": "Fixture 3B Instruct", "architecture": "fixture", "hidden_size": 64, "layer_count": 8,
        "vocab_size": 32000, "tokenizer_sha256": "a" * 64, "chat_template_sha256": "b" * 64,
        "quantization": "Q8_0", "file_size": 123456, "filename": "fixture-Q8_0.gguf", "sha256": "c" * 64,
    }
    base.update(overrides)
    return base


# ==================================================================================== compare_field

def test_compare_field_match():
    r = ve.compare_field("vocab_size", 32000, 32000)
    assert r == {"field": "vocab_size", "status": "MATCH", "expected": 32000, "exported": 32000}


def test_compare_field_mismatch():
    r = ve.compare_field("architecture", "qwen2", "llama")
    assert r["status"] == "MISMATCH"
    assert r["expected"] == "qwen2" and r["exported"] == "llama"


def test_compare_field_unknown_when_both_missing():
    r = ve.compare_field("hidden_size", None, None)
    assert r["status"] == "UNKNOWN"


def test_compare_field_one_sided_missing_is_mismatch_not_unknown():
    # one side resolved a real value, the other didn't -- that's drift, not "nothing to compare".
    r = ve.compare_field("chat_template_sha256", "a" * 64, None)
    assert r["status"] == "MISMATCH"


# ==================================================================================== compare_identities

def test_compare_identities_all_match_is_ok_even_with_quant_difference():
    expected = _identity(quantization="F16")
    exported = _identity(quantization="Q4_K_M")
    out = ve.compare_identities(expected, exported)
    assert out["ok"] is True
    assert out["mismatched_fields"] == []
    assert out["quantization"] == {"expected": "F16", "exported": "Q4_K_M", "same": False}


@pytest.mark.parametrize("field,bad", [
    ("tokenizer_sha256", "z" * 64),
    ("chat_template_sha256", "z" * 64),
    ("vocab_size", 31999),
    ("architecture", "other"),
    ("hidden_size", 128),
    ("layer_count", 9),
])
def test_compare_identities_flags_each_field_mismatch(field, bad):
    expected = _identity()
    exported = _identity(**{field: bad})
    out = ve.compare_identities(expected, exported)
    assert out["ok"] is False
    assert out["mismatched_fields"] == [field]
    bad_field = next(f for f in out["fields"] if f["field"] == field)
    assert bad_field["status"] == "MISMATCH"
    assert bad_field["expected"] == _identity()[field]
    assert bad_field["exported"] == bad


def test_compare_identities_unknown_field_does_not_fail_the_gate():
    expected = _identity(hidden_size=None)
    exported = _identity(hidden_size=None)
    out = ve.compare_identities(expected, exported)
    assert out["ok"] is True
    hs = next(f for f in out["fields"] if f["field"] == "hidden_size")
    assert hs["status"] == "UNKNOWN"


# ==================================================================================== single-file: chat template

def test_check_chat_template_presence_ok():
    r = ve.check_chat_template_presence("{{ messages }}")
    assert r["status"] == "OK"


@pytest.mark.parametrize("template", [None, "", "   "])
def test_check_chat_template_presence_warns_when_missing(template):
    r = ve.check_chat_template_presence(template)
    assert r["status"] == "WARN"
    assert "degenerate text" in r["detail"]


# ==================================================================================== single-file: role markers

def test_check_template_markers_unknown_when_no_template():
    r = ve.check_template_markers(None, "chatml")
    assert r["status"] == "UNKNOWN"


def test_check_template_markers_ok_for_known_family_chatml():
    tpl = "<|im_start|>system\n{{ system }}<|im_end|>\n<|im_start|>user\n{{ user }}<|im_end|>"
    r = ve.check_template_markers(tpl, "chatml")
    assert r["status"] == "OK"


def test_check_template_markers_ok_for_known_family_llama3():
    tpl = "<|start_header_id|>user<|end_header_id|>\n{{ user }}<|eot_id|>"
    r = ve.check_template_markers(tpl, "llama3")
    assert r["status"] == "OK"


def test_check_template_markers_warns_on_wrong_family_template():
    # a mistral-style template, but the filename said this should be llama3 -- wrong-template export.
    tpl = "[INST] {{ user }} [/INST]"
    r = ve.check_template_markers(tpl, "llama3")
    assert r["status"] == "WARN"
    assert "llama3" in r["detail"]


def test_check_template_markers_generic_fallback_matches_unknown_family():
    tpl = "[INST] {{ user }} [/INST]"
    r = ve.check_template_markers(tpl, None)
    assert r["status"] == "OK"
    assert "GENERIC" in r["detail"]


def test_check_template_markers_generic_fallback_warns_when_nothing_recognized():
    tpl = "{{ user }} :: {{ assistant }}"
    r = ve.check_template_markers(tpl, None)
    assert r["status"] == "WARN"


# ==================================================================================== single-file: BOS/EOS

def test_check_bos_eos_ids_both_present():
    r = ve.check_bos_eos_ids({"tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2})
    assert r["status"] == "OK"


def test_check_bos_eos_ids_one_missing():
    r = ve.check_bos_eos_ids({"tokenizer.ggml.bos_token_id": 1})
    assert r["status"] == "WARN"
    assert "EOS" in r["detail"] and "BOS" not in r["detail"].split("missing")[1].split(")")[0]


def test_check_bos_eos_ids_both_missing():
    r = ve.check_bos_eos_ids({})
    assert r["status"] == "WARN"
    assert "BOS" in r["detail"] and "EOS" in r["detail"]


# ==================================================================================== single-file: vocab size

@pytest.mark.parametrize("vocab_size", [32000, 1])
def test_check_vocab_size_sane_ok(vocab_size):
    assert ve.check_vocab_size_sane(vocab_size)["status"] == "OK"


@pytest.mark.parametrize("vocab_size", [0, -1, None, "32000", True])
def test_check_vocab_size_sane_warns(vocab_size):
    assert ve.check_vocab_size_sane(vocab_size)["status"] == "WARN"


# ==================================================================================== _expected_tmpl_key

def test_expected_tmpl_key_known_llama3_family():
    assert ve._expected_tmpl_key("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") == "llama3"


def test_expected_tmpl_key_known_mistral_family():
    assert ve._expected_tmpl_key("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf") == "mistral"


def test_expected_tmpl_key_known_chat_model_with_no_explicit_tmpl_defaults_chatml():
    assert ve._expected_tmpl_key("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == "chatml"


def test_expected_tmpl_key_unrecognized_instruct_filename_guesses_chatml():
    assert ve._expected_tmpl_key("some-random-model-instruct.gguf") == "chatml"


def test_expected_tmpl_key_unrecognized_non_chat_filename_is_none():
    assert ve._expected_tmpl_key("some-random-base-model.gguf") is None


# ==================================================================================== single_file_findings

def test_single_file_findings_all_ok():
    identity = _identity(vocab_size=32000)
    metadata = {
        "tokenizer.chat_template": "<|im_start|>user\n{{ user }}<|im_end|>",
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }
    findings = ve.single_file_findings("Qwen2.5-7B-Instruct-Q4_K_M.gguf", identity, metadata)
    assert [f["check"] for f in findings] == [
        "chat_template_present", "template_role_markers", "bos_eos_ids_present", "vocab_size_sane",
    ]
    assert all(f["status"] == "OK" for f in findings)


def test_single_file_findings_flags_missing_template_and_ids():
    identity = _identity(vocab_size=32000)
    metadata = {}
    findings = ve.single_file_findings("Qwen2.5-7B-Instruct-Q4_K_M.gguf", identity, metadata)
    by_check = {f["check"]: f for f in findings}
    assert by_check["chat_template_present"]["status"] == "WARN"
    assert by_check["template_role_markers"]["status"] == "UNKNOWN"
    assert by_check["bos_eos_ids_present"]["status"] == "WARN"
    assert by_check["vocab_size_sane"]["status"] == "OK"


# ==================================================================================== rendering (pure)

def test_format_two_file_report_pass():
    identity = _identity()
    comparison = ve.compare_identities(identity, identity)
    report = {"mode": "two-file", "expected_path": "a.gguf", "exported_path": "b.gguf",
             "expected_identity": identity, "exported_identity": identity,
             "comparison": comparison, "known_answers": None}
    out = ve.format_two_file_report(report)
    assert "PASS" in out
    assert "a.gguf" in out and "b.gguf" in out
    assert "STATIC metadata gate" in out


def test_format_two_file_report_fail_names_the_mismatch():
    expected = _identity()
    exported = _identity(architecture="other")
    comparison = ve.compare_identities(expected, exported)
    report = {"mode": "two-file", "expected_path": "a.gguf", "exported_path": "b.gguf",
             "expected_identity": expected, "exported_identity": exported,
             "comparison": comparison, "known_answers": None}
    out = ve.format_two_file_report(report)
    assert "FAIL" in out
    assert "architecture" in out


def test_format_single_file_report_counts_warnings():
    identity = _identity()
    findings = [
        {"check": "chat_template_present", "status": "WARN", "detail": "missing"},
        {"check": "vocab_size_sane", "status": "OK", "detail": "fine"},
    ]
    report = {"mode": "single-file", "path": "x.gguf", "identity": identity, "findings": findings}
    out = ve.format_single_file_report(report)
    assert "1 warning(s)." in out
    assert "x.gguf" in out


# ==================================================================================== cmd_validate_export

def _args(gguf_a, gguf_b=None, *, known_answers=0, strict=False, cpu=False, json_=False):
    return argparse.Namespace(gguf_a=gguf_a, gguf_b=gguf_b, known_answers=known_answers, strict=strict,
                              cpu=cpu, json=json_)


@pytest.fixture
def two_files(tmp_path):
    a = tmp_path / "expected.gguf"; a.write_bytes(b"a")
    b = tmp_path / "exported.gguf"; b.write_bytes(b"b")
    return str(a), str(b)


def test_cmd_validate_export_two_file_match_exits_zero(tmp_path, monkeypatch, capsys, two_files):
    path_a, path_b = two_files
    identity = _identity()
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: dict(identity))
    rc = ve.cmd_validate_export(_args(path_a, path_b))
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_cmd_validate_export_two_file_mismatch_exits_one(monkeypatch, capsys, two_files):
    path_a, path_b = two_files

    def fake_identity(p, **kw):
        return _identity(tokenizer_sha256="z" * 64) if p == path_b else _identity()

    monkeypatch.setattr(contracts, "gguf_identity", fake_identity)
    rc = ve.cmd_validate_export(_args(path_a, path_b))
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "tokenizer_sha256" in out


def test_cmd_validate_export_two_file_json_is_valid_json(monkeypatch, capsys, two_files):
    path_a, path_b = two_files
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    rc = ve.cmd_validate_export(_args(path_a, path_b, json_=True))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["mode"] == "two-file"
    assert parsed["comparison"]["ok"] is True


def test_cmd_validate_export_known_answers_is_informational_not_gating(monkeypatch, capsys, two_files):
    path_a, path_b = two_files
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    calls = []

    def fake_known_answers(a, b, n, *, cpu=False):
        calls.append((a, b, n, cpu))
        return {
            "tokenizer_compat": {"compatible": True, "probes": []},
            "template_match": True,
            "agg": {"label_a": "expected", "label_b": "exported", "n_runs": 0, "n_verified": 0,
                   "n_skipped": 0, "total_tokens": 0, "total_preserved": 0, "total_flipped": 0,
                   "total_unknown": 0, "pct_preserved": None, "per_run": [], "top_flips": [],
                   "n_flips_total": 0, "caveat": None, "topk_note": None},
            "verdict": {"verdict": "CHANGED", "thresholds": {}, "total_tokens": 0, "total_flipped": 5,
                       "mean_abs_delta_nats_all_mean": None, "is_heuristic": True,
                       "message": "pretend it changed a lot"},
        }

    monkeypatch.setattr(ve, "run_known_answers_check", fake_known_answers)
    rc = ve.cmd_validate_export(_args(path_a, path_b, known_answers=4))
    # the static metadata still matches -- exit code must stay 0 even though the fake verdict says CHANGED,
    # per this module's own documented policy that --known-answers is informational, not gating.
    assert rc == 0
    assert calls == [(path_a, path_b, 4, False)]
    out = capsys.readouterr().out
    assert "informational" in out
    assert "CHANGED" in out


def test_cmd_validate_export_single_file_no_warnings_exits_zero(tmp_path, monkeypatch, capsys):
    path = tmp_path / "exported.gguf"; path.write_bytes(b"x")
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    monkeypatch.setattr(ve, "_read_raw_metadata", lambda p: {
        "tokenizer.chat_template": "<|im_start|>user\n{{ user }}<|im_end|>",
        "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2,
    })
    rc = ve.cmd_validate_export(_args(str(path)))
    assert rc == 0
    assert "no warnings." in capsys.readouterr().out


def test_cmd_validate_export_single_file_missing_template_warns_but_exits_zero_without_strict(
        tmp_path, monkeypatch, capsys):
    path = tmp_path / "exported.gguf"; path.write_bytes(b"x")
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    monkeypatch.setattr(ve, "_read_raw_metadata", lambda p: {})
    rc = ve.cmd_validate_export(_args(str(path)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "warning(s)." in out


def test_cmd_validate_export_single_file_missing_template_exits_one_with_strict(tmp_path, monkeypatch):
    path = tmp_path / "exported.gguf"; path.write_bytes(b"x")
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    monkeypatch.setattr(ve, "_read_raw_metadata", lambda p: {})
    rc = ve.cmd_validate_export(_args(str(path), strict=True))
    assert rc == 1


def test_cmd_validate_export_single_file_json_is_valid_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / "exported.gguf"; path.write_bytes(b"x")
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    monkeypatch.setattr(ve, "_read_raw_metadata", lambda p: {
        "tokenizer.chat_template": "<|im_start|>user\n{{ user }}<|im_end|>",
        "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2,
    })
    rc = ve.cmd_validate_export(_args(str(path), json_=True))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["mode"] == "single-file"
    assert len(parsed["findings"]) == 4


def test_cmd_validate_export_known_answers_without_second_gguf_refuses(tmp_path, monkeypatch):
    path = tmp_path / "exported.gguf"; path.write_bytes(b"x")
    monkeypatch.setattr(contracts, "gguf_identity", lambda p, **kw: _identity())
    with pytest.raises(CloznError, match="two-file mode only"):
        ve.cmd_validate_export(_args(str(path), known_answers=4))


def test_cmd_validate_export_unreadable_gguf_raises_clean_error(tmp_path, monkeypatch):
    path = tmp_path / "exported.gguf"; path.write_bytes(b"x")

    def boom(p, **kw):
        raise ValueError("not a GGUF file")

    monkeypatch.setattr(contracts, "gguf_identity", boom)
    with pytest.raises(CloznError):
        ve.cmd_validate_export(_args(str(path)))


# ==================================================================================== add_subparser / argparse

def _build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    ve.add_subparser(sub)
    return p


def test_add_subparser_defaults_single_file_mode():
    p = _build_parser()
    args = p.parse_args(["validate-export", "exported.gguf"])
    assert args.gguf_a == "exported.gguf"
    assert args.gguf_b is None
    assert args.known_answers == 0
    assert args.strict is False
    assert args.cpu is False
    assert args.json is False
    assert args.fn is ve.cmd_validate_export


def test_add_subparser_two_file_mode_with_overrides():
    p = _build_parser()
    args = p.parse_args(["validate-export", "expected.gguf", "exported.gguf",
                        "--known-answers", "8", "--strict", "--cpu", "--json"])
    assert args.gguf_a == "expected.gguf"
    assert args.gguf_b == "exported.gguf"
    assert args.known_answers == 8
    assert args.strict is True
    assert args.cpu is True
    assert args.json is True


def test_add_subparser_requires_at_least_one_model():
    import contextlib
    import io
    p = _build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):
            p.parse_args(["validate-export"])
