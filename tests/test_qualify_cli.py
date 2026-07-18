"""test_qualify_cli -- clozn/cli/commands/qualify.py (`clozn qualify-whitebox`): the honest per-feature
capability matrix built from contracts.gguf_identity + the wave1 qualification ledger + any locally
installed j-lens/SAE artifacts.

Model-free / GPU-free throughout, mirroring tests/test_quant_check.py's and
tests/test_artifact_contracts.py's own discipline: no engine, no GPU, no Torch. `gguf_identity` is never
actually invoked against a real GGUF -- either a synthetic identity dict is built directly (mirrors
test_artifact_contracts.py's own approach) or `contracts.gguf_header_from_path` is monkeypatched exactly
like that module's `model` fixture. `add_subparser`'s argparse wiring is exercised on a throwaway parser,
never touching clozn/cli/main.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.cli.commands.qualify as qw  # noqa: E402


# ==================================================================================== fixtures / helpers

def _identity(**overrides) -> dict:
    base = {
        "name": "Qwen2.5 7B Instruct",
        "architecture": "qwen2",
        "hidden_size": 3584,
        "layer_count": 28,
        "vocab_size": 152064,
        "tokenizer_sha256": "t" * 64,
        "chat_template_sha256": "c" * 64,
        "quantization": "Q4_K_M",
        "file_size": 4683074240,
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "sha256": "a" * 64,
    }
    base.update(overrides)
    return base


def _wave1_entry(**overrides) -> dict:
    entry = {
        "family": "qwen2.5",
        "source_id": "Qwen/Qwen2.5-7B-Instruct",
        "gguf": {
            "sha256": "a" * 64,
            "architecture": "qwen2",
            "tokenizer_sha256": "t" * 64,
        },
        "status": {
            "core": "passed_deep_cpu",
            "white_box": "partial",
            "dials": "legacy_global_requires_model_scoped_recalibration",
            "jlens": "qualified_q4_k_m",
        },
    }
    entry.update(overrides)
    return entry


def _manifest(identity, payload, artifact_type="jlens", **model_overrides):
    target = {
        "source_id": "owner/fixture",
        "architecture": identity["architecture"],
        "hidden_size": identity["hidden_size"],
        "layer_count": identity["layer_count"],
        "vocab_size": identity["vocab_size"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "compatible_gguf_sha256": [identity["sha256"]],
    }
    target.update(model_overrides)
    return {
        "contract_version": 1,
        "artifact_type": artifact_type,
        "artifact_version": 1,
        "model": target,
        "files": {
            payload.name: {
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }


def _install_artifact(tmp_path, identity, artifact_type, dirname="fixture-v1", corrupt=False):
    """Write a real, on-disk manifest.json + payload under tmp_path/<artifact_type>/<dirname>/ -- the same
    layout contracts.find_compatible_artifact discovers automatically. Returns the artifact_root
    (tmp_path)."""
    directory = tmp_path / artifact_type / dirname
    directory.mkdir(parents=True)
    payload = directory / "payload.bin"
    payload.write_bytes(b"artifact payload bytes")
    manifest = _manifest(identity, payload, artifact_type=artifact_type)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if corrupt:
        payload.write_bytes(b"corrupted after the manifest was written")
    return str(tmp_path)


# ============================================================================================ load_wave1

def test_load_wave1_missing_file_returns_empty(tmp_path):
    assert qw.load_wave1(str(tmp_path / "nope.json")) == []


def test_load_wave1_invalid_json_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert qw.load_wave1(str(path)) == []


def test_load_wave1_reads_models_list(tmp_path):
    path = tmp_path / "wave1.json"
    path.write_text(json.dumps({"models": [_wave1_entry()]}), encoding="utf-8")
    models = qw.load_wave1(str(path))
    assert len(models) == 1
    assert models[0]["source_id"] == "Qwen/Qwen2.5-7B-Instruct"


def test_load_wave1_default_path_reads_the_real_ledger():
    """Sanity: docs/qualification/wave1.json (the file this task's ledger lives in) actually exists and
    parses into a non-empty list of real entries."""
    models = qw.load_wave1()
    assert len(models) >= 1
    assert all("source_id" in m for m in models)


# ====================================================================================== find_wave1_match

def test_find_wave1_match_by_sha256():
    identity = _identity()
    entry = _wave1_entry()
    match, kind = qw.find_wave1_match(identity, wave1_models=[entry])
    assert match is entry
    assert kind == "sha256"


def test_find_wave1_match_by_family_when_sha256_differs():
    identity = _identity(sha256="b" * 64)  # different exact file, same arch + tokenizer
    entry = _wave1_entry()
    match, kind = qw.find_wave1_match(identity, wave1_models=[entry])
    assert match is entry
    assert kind == "family"


def test_find_wave1_match_by_checkpoint_when_nothing_else_matches():
    identity = _identity(sha256="b" * 64, tokenizer_sha256="z" * 64)
    entry = _wave1_entry()
    match, kind = qw.find_wave1_match(
        identity, checkpoint="Qwen/Qwen2.5-7B-Instruct", wave1_models=[entry]
    )
    assert match is entry
    assert kind == "checkpoint"


def test_find_wave1_match_checkpoint_architecture_mismatch():
    identity = _identity(sha256="b" * 64, tokenizer_sha256="z" * 64, architecture="llama")
    entry = _wave1_entry()  # gguf.architecture == "qwen2"
    match, kind = qw.find_wave1_match(
        identity, checkpoint="Qwen/Qwen2.5-7B-Instruct", wave1_models=[entry]
    )
    assert match is entry
    assert kind == "checkpoint_architecture_mismatch"


def test_find_wave1_match_no_match():
    identity = _identity(sha256="b" * 64, tokenizer_sha256="z" * 64, architecture="mystery-arch")
    match, kind = qw.find_wave1_match(identity, wave1_models=[_wave1_entry()])
    assert match is None
    assert kind is None


# =================================================================================== steer_qualification

def test_steer_unrecognized_family_not_qualified():
    result = qw.steer_qualification(None, None, None)
    assert result["feature"] == "steering"
    assert result["qualified"] is False
    assert "unrecognized model family" in result["reason"]


def test_steer_known_family_no_calibrated_layer_not_qualified():
    result = qw.steer_qualification("qwen3.5-9b", None, None)
    assert result["qualified"] is False
    assert "no calibrated steer tap" in result["reason"]


def test_steer_known_layer_no_ledger_entry_qualified():
    result = qw.steer_qualification("llama-3.2-1b", 8, None)
    assert result["qualified"] is True
    assert "8" in result["reason"]


def test_steer_known_layer_but_ledger_flags_legacy_dials_not_qualified():
    entry = _wave1_entry()  # dials: legacy_global_requires_model_scoped_recalibration
    result = qw.steer_qualification("qwen2.5-7b", 14, entry)
    assert result["qualified"] is False
    assert "legacy_global_requires_model_scoped_recalibration" in result["reason"]


def test_steer_known_layer_ledger_dials_passed_qualified():
    entry = _wave1_entry(status={**_wave1_entry()["status"], "dials": "passed_model_scoped"})
    result = qw.steer_qualification("qwen2.5-7b", 14, entry)
    assert result["qualified"] is True


# =================================================================================== jlens_qualification

def test_jlens_no_local_artifact_no_ledger_not_qualified(tmp_path):
    identity = _identity()
    result = qw.jlens_qualification(identity, str(tmp_path), None)
    assert result["qualified"] is False
    assert "unrecognized architecture" in result["reason"]


def test_jlens_no_local_artifact_but_ledger_says_qualified_still_not_qualified(tmp_path):
    identity = _identity()
    entry = _wave1_entry()  # status.jlens == "qualified_q4_k_m"
    result = qw.jlens_qualification(identity, str(tmp_path), entry)
    assert result["qualified"] is False
    assert "no local artifact installed" in result["reason"]
    assert "qualified_q4_k_m" in result["reason"]


def test_jlens_local_artifact_installed_and_valid_qualified(tmp_path):
    identity = _identity()
    root = _install_artifact(tmp_path, identity, "jlens")
    result = qw.jlens_qualification(identity, root, None)
    assert result["qualified"] is True
    assert "artifact_dir" in result["evidence"]


def test_jlens_local_artifact_corrupt_not_qualified(tmp_path):
    identity = _identity()
    root = _install_artifact(tmp_path, identity, "jlens", corrupt=True)
    result = qw.jlens_qualification(identity, root, None)
    assert result["qualified"] is False
    assert "failed contract validation" in result["reason"]


# ===================================================================================== sae_qualification

def test_sae_no_local_artifact_not_qualified(tmp_path):
    identity = _identity()
    result = qw.sae_qualification(identity, str(tmp_path))
    assert result["qualified"] is False
    assert "no SAE readout artifact is contract-qualified" in result["reason"]


def test_sae_local_artifact_installed_and_valid_qualified(tmp_path):
    identity = _identity()
    root = _install_artifact(tmp_path, identity, "sae")
    result = qw.sae_qualification(identity, root)
    assert result["qualified"] is True
    assert "artifact_dir" in result["evidence"]


# ============================================================================= _core_feature_qualification

@pytest.mark.parametrize("name", ["receipts", "explain", "rewrite"])
def test_core_features_always_qualified_regardless_of_ledger(name):
    assert qw._core_feature_qualification(name, None)["qualified"] is True
    assert qw._core_feature_qualification(name, _wave1_entry())["qualified"] is True


def test_core_feature_reason_mentions_untested_with_no_ledger_entry():
    result = qw._core_feature_qualification("receipts", None)
    assert "not been run through the wave1 smoke ladder" in result["reason"]


def test_core_feature_reason_mentions_wave1_status_when_matched():
    result = qw._core_feature_qualification("receipts", _wave1_entry())
    assert "passed_deep_cpu" in result["reason"]


# ============================================================================== build_capability_matrix

def test_build_capability_matrix_unknown_architecture_is_honest_about_steer_and_jlens(tmp_path):
    """The task's own bar: an unrecognized architecture must NOT claim steering or j-lens work."""
    identity = _identity(architecture="totally-unknown-arch", filename="mystery-model.gguf",
                         sha256="f" * 64, tokenizer_sha256="f" * 64)
    report = qw.build_capability_matrix(identity, artifact_root=str(tmp_path), wave1_models=[])
    summary = report["summary"]
    assert set(summary["qualified"]) == {"receipts", "explain", "rewrite"}
    assert set(summary["not_qualified"]) == {"steering", "jlens", "sae"}
    assert report["wave1_match"] is None
    assert report["model_family"] is None


def test_build_capability_matrix_known_family_with_wave1_match(tmp_path):
    identity = _identity()  # qwen2.5-7b filename, sha256 == "a"*64
    report = qw.build_capability_matrix(
        identity, artifact_root=str(tmp_path), wave1_models=[_wave1_entry()]
    )
    assert report["wave1_match"]["match_kind"] == "sha256"
    assert report["wave1_match"]["source_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert report["model_family"] == "qwen2.5-7b"
    by_name = {f["feature"]: f for f in report["features"]}
    assert by_name["receipts"]["qualified"] is True
    assert by_name["steering"]["qualified"] is False    # calibrated layer, but legacy dials
    assert by_name["jlens"]["qualified"] is False        # ledger says qualified, nothing installed locally
    assert by_name["sae"]["qualified"] is False


def test_build_capability_matrix_reports_checkpoint_architecture_mismatch(tmp_path):
    identity = _identity(sha256="b" * 64, tokenizer_sha256="z" * 64, architecture="llama")
    report = qw.build_capability_matrix(
        identity, checkpoint="Qwen/Qwen2.5-7B-Instruct",
        artifact_root=str(tmp_path), wave1_models=[_wave1_entry()],
    )
    assert report["wave1_match"] is None
    assert "does not match" in report["checkpoint_warning"]


def test_build_capability_matrix_features_are_json_serializable(tmp_path):
    identity = _identity()
    report = qw.build_capability_matrix(
        identity, artifact_root=str(tmp_path), wave1_models=[_wave1_entry()]
    )
    json.dumps(report)  # must never raise


# ========================================================================================= format_report

def test_format_report_lists_every_feature_and_status(tmp_path):
    identity = _identity(architecture="mystery", filename="mystery.gguf",
                         sha256="e" * 64, tokenizer_sha256="e" * 64)
    report = qw.build_capability_matrix(identity, artifact_root=str(tmp_path), wave1_models=[])
    text = qw.format_report(report)
    assert "qualify-whitebox" in text
    assert "capability matrix" in text
    for feature in ("receipts", "explain", "rewrite", "steering", "jlens", "sae"):
        assert feature in text
    assert "wave1 ledger: no match" in text


def test_format_report_shows_wave1_match_and_checkpoint_warning(tmp_path):
    identity = _identity(sha256="b" * 64, tokenizer_sha256="z" * 64, architecture="llama")
    report = qw.build_capability_matrix(
        identity, checkpoint="Qwen/Qwen2.5-7B-Instruct",
        artifact_root=str(tmp_path), wave1_models=[_wave1_entry()],
    )
    text = qw.format_report(report)
    assert "note:" in text
    assert "does not match" in text


# =========================================================================== add_subparser / argparse

def _build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    qw.add_subparser(sub)
    return p


def test_add_subparser_defaults():
    p = _build_parser()
    args = p.parse_args(["qualify-whitebox", "model.gguf"])
    assert args.gguf == "model.gguf"
    assert args.json is False
    assert args.checkpoint is None
    assert args.fn is qw.cmd_qualify


def test_add_subparser_parses_overrides():
    p = _build_parser()
    args = p.parse_args(["qualify-whitebox", "model.gguf", "--json", "--checkpoint", "org/model"])
    assert args.json is True
    assert args.checkpoint == "org/model"


def test_add_subparser_requires_gguf_arg():
    import contextlib
    import io
    p = _build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            p.parse_args(["qualify-whitebox"])
            raised = False
        except SystemExit:
            raised = True
    assert raised


# ================================================================================================ cmd_qualify

class _Args:
    def __init__(self, gguf, json=False, checkpoint=None):
        self.gguf = gguf
        self.json = json
        self.checkpoint = checkpoint


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """Point the default artifact root at an empty tmp dir so cmd_qualify tests never depend on -- or
    accidentally pick up -- whatever happens to live under the real ~/.clozn/artifacts on this machine."""
    monkeypatch.setenv("CLOZN_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    return tmp_path


def _patch_gguf(monkeypatch, identity):
    monkeypatch.setattr(qw.contracts, "gguf_identity", lambda _path: identity)
    monkeypatch.setattr("clozn.cli.commands.models.resolve_model", lambda arg: arg)


def test_cmd_qualify_text_output(monkeypatch, isolated_artifacts, capsys):
    _patch_gguf(monkeypatch, _identity())
    rc = qw.cmd_qualify(_Args("Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "qualify-whitebox" in out
    assert "Qwen2.5-7B-Instruct-Q4_K_M.gguf" in out


def test_cmd_qualify_json_output_is_valid_and_complete(monkeypatch, isolated_artifacts, capsys):
    _patch_gguf(monkeypatch, _identity())
    rc = qw.cmd_qualify(_Args("Qwen2.5-7B-Instruct-Q4_K_M.gguf", json=True))
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["identity"]["filename"] == "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    assert len(report["features"]) == 6
    assert set(report["summary"]["qualified"]) | set(report["summary"]["not_qualified"]) == {
        "receipts", "explain", "rewrite", "steering", "jlens", "sae"
    }


def test_cmd_qualify_unknown_architecture_end_to_end(monkeypatch, isolated_artifacts, capsys):
    _patch_gguf(monkeypatch, _identity(architecture="totally-unknown-arch", filename="mystery.gguf",
                                       sha256="d" * 64, tokenizer_sha256="d" * 64))
    rc = qw.cmd_qualify(_Args("mystery.gguf", json=True))
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["summary"]["not_qualified"]) == {"steering", "jlens", "sae"}
