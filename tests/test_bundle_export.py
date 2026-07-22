"""Model-free tests for ``clozn runs export-bundle`` (roadmap Phase 4.5 "Open export").

Covers: the bundle round-trip (build -> every artifact present and hash-matches), that hash verification
genuinely catches tampering, that the tensor writer's NumPy/raw-bin dispatch both produce readable data,
that the CLI wiring refuses/allows overwrite correctly, and that the export code itself never imports a
networking-capable module (the generated notebook's own opt-in live-reproduction cell is the sole,
deliberate exception -- see test_notebook_export.py).
"""
from __future__ import annotations

import ast
import json
import os
import struct
from pathlib import Path

import pytest

from clozn.cli import main as cli
from clozn.runs import bundle_export, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _record(*, prompt="hello", response="hello back", with_identity=True, **overrides):
    kwargs = dict(
        source="cli", client="sdk", model="qwen2.5-0.5b", substrate="engine",
        messages=[{"role": "user", "content": prompt}], response=response,
        trace={"tokens": ["hello", " back"], "confidence": [0.9, 0.8], "logprobs": [-0.1, -0.2]},
    )
    if with_identity:
        kwargs["identity"] = {
            "model_sha256": "a" * 64, "template_fingerprint": "tmpl-1",
            "engine_build": "build-1", "clozn_version": "0.1.0", "captured_at": "2026-01-01T00:00:00Z",
        }
    kwargs.update(overrides)
    return store.record(**kwargs)


def _attach_influence_map(run_id: str) -> dict:
    run = store.get_run(run_id)
    influence = {
        "schema": "clozn.context_answer_influence.v1",
        "method": {"name": "teacher_forced_matched_context_replacement",
                  "claim_limit": "behavioral dependence, not a percentage"},
        "identity": {"run_id": run_id, "prompt_view": "messages"},
        "baseline": {"logprobs": [-0.1, -0.2], "scored_once": True},
        "matrix": [[0.5, 0.1], [0.2, 0.05]],
        "matrix_shape": [2, 2],
        "links": [{"prompt_span": 0, "answer_span": 0, "delta": 0.5}],
        "artifact_sha256": "b" * 64,
    }
    run["influence_map"] = influence
    assert store.replace_run(run)
    return influence


# =========================================================================================== round trip

def test_export_bundle_writes_every_expected_artifact(isolated):
    run_id = _record()
    _attach_influence_map(run_id)
    out_dir = isolated / "bundle"

    manifest = bundle_export.export_bundle(run_id, str(out_dir))

    assert manifest["schema"] == bundle_export.SCHEMA_VERSION
    assert manifest["run_id"] == run_id
    assert manifest["identity"]["model_sha256"] == "a" * 64
    assert manifest["method"]["name"] == "teacher_forced_matched_context_replacement"
    assert manifest["scope"] == "messages"

    written = {p.name for p in out_dir.iterdir()}
    assert written == {
        "manifest.json", "receipt_bundle.json", "influence_map.json", "trace.json",
        "tensors.npz", "reproduce.ipynb",
    }
    paths_in_manifest = {a["path"] for a in manifest["artifacts"]}
    assert paths_in_manifest == written - {"manifest.json"}


def test_export_bundle_without_influence_map_omits_that_file(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    manifest = bundle_export.export_bundle(run_id, str(out_dir))
    assert not (out_dir / "influence_map.json").exists()
    assert manifest["method"] is None and manifest["scope"] is None
    kinds = {a["kind"] for a in manifest["artifacts"]}
    assert "influence_map_evidence" not in kinds


def test_manifest_artifact_hashes_match_the_actual_written_files(isolated):
    run_id = _record()
    _attach_influence_map(run_id)
    out_dir = isolated / "bundle"
    manifest = bundle_export.export_bundle(run_id, str(out_dir))
    for artifact in manifest["artifacts"]:
        from clozn.artifacts.contracts import sha256_file
        actual = sha256_file(out_dir / artifact["path"])
        assert actual == artifact["sha256"]
        assert len(artifact["sha256"]) == 64
        assert os.path.getsize(out_dir / artifact["path"]) == artifact["bytes"]


def test_receipt_bundle_json_round_trips_the_run_content(isolated):
    run_id = _record(prompt="what is 2+2", response="4")
    out_dir = isolated / "bundle"
    bundle_export.export_bundle(run_id, str(out_dir))
    receipt = json.loads((out_dir / "receipt_bundle.json").read_text(encoding="utf-8"))
    assert receipt["run"]["messages"][0]["content"] == "what is 2+2"
    assert receipt["run"]["response"] == "4"
    assert receipt["identity"]["model_sha256"] == "a" * 64


# ============================================================================== hash verification / tamper

def test_verify_bundle_reports_ok_for_an_untampered_export(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    bundle_export.export_bundle(run_id, str(out_dir))
    results = bundle_export.verify_bundle(str(out_dir))
    assert results and all(r["status"] == "OK" for r in results)


def test_verify_bundle_catches_a_single_flipped_byte(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    bundle_export.export_bundle(run_id, str(out_dir))
    target = out_dir / "trace.json"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF  # flip one bit -- SHA-256 must not shrug this off
    target.write_bytes(bytes(data))

    results = bundle_export.verify_bundle(str(out_dir))
    by_path = {r["path"]: r["status"] for r in results}
    assert by_path["trace.json"] == "TAMPERED"
    assert by_path["receipt_bundle.json"] == "OK"  # untouched files still verify clean


def test_verify_bundle_reports_missing_file(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    bundle_export.export_bundle(run_id, str(out_dir))
    (out_dir / "trace.json").unlink()
    results = bundle_export.verify_bundle(str(out_dir))
    by_path = {r["path"]: r["status"] for r in results}
    assert by_path["trace.json"] == "MISSING"


# ===================================================================================== tensor extraction

def test_extract_tensors_pulls_trace_and_influence_arrays(isolated):
    run_id = _record()
    _attach_influence_map(run_id)
    from clozn.receipts import bundle as receipt_bundle
    run = store.get_run(run_id)
    receipt = receipt_bundle.build(run)
    tensors = bundle_export._extract_tensors(receipt)
    assert tensors["trace.confidence"] == [0.9, 0.8]
    assert tensors["trace.logprobs"] == [-0.1, -0.2]
    assert tensors["influence_map.baseline.logprobs"] == [-0.1, -0.2]
    assert tensors["influence_map.matrix"] == [[0.5, 0.1], [0.2, 0.05]]


def test_extract_tensors_skips_ragged_or_non_numeric_fields():
    receipt = {"trace": {"confidence": ["not", "numbers"]},
              "influence_map": {"matrix": [[0.1, 0.2], [0.3]]}}  # ragged
    tensors = bundle_export._extract_tensors(receipt)
    assert tensors == {}


def test_write_tensors_raw_bin_produces_readable_little_endian_float32(tmp_path):
    arrays = {"trace.confidence": [0.25, 0.5, 0.75], "influence_map.matrix": [[1.0, 2.0], [3.0, 4.0]]}
    entries = bundle_export._write_tensors_raw_bin(str(tmp_path), arrays)
    by_name = {e["array_name"]: e for e in entries}
    assert by_name["trace.confidence"]["shape"] == [3]
    assert by_name["influence_map.matrix"]["shape"] == [2, 2]
    raw = (tmp_path / by_name["trace.confidence"]["path"]).read_bytes()
    values = struct.unpack("<3f", raw)
    assert values == pytest.approx((0.25, 0.5, 0.75))


def test_write_tensors_falls_back_to_raw_bin_when_numpy_import_fails(tmp_path, monkeypatch):
    def _broken_npz(*_a, **_k):
        raise ImportError("no numpy here")

    monkeypatch.setattr(bundle_export, "_write_tensors_npz", _broken_npz)
    entries = bundle_export._write_tensors(str(tmp_path), {"trace.confidence": [0.1, 0.2]})
    assert entries and entries[0]["format"] == "raw_f32_bin"
    assert (tmp_path / entries[0]["path"]).is_file()


def test_write_tensors_empty_input_writes_nothing(tmp_path):
    assert bundle_export._write_tensors(str(tmp_path), {}) == []
    assert list(tmp_path.iterdir()) == []


# ================================================================================================ refusals

def test_export_bundle_rejects_unknown_run_id(isolated):
    with pytest.raises(bundle_export.BundleExportError, match="not found"):
        bundle_export.export_bundle("run_missing", str(isolated / "out"))


def test_export_bundle_rejects_invalid_run_id_shape(isolated):
    with pytest.raises(bundle_export.BundleExportError, match="valid run ID"):
        bundle_export.export_bundle("../escape", str(isolated / "out"))


def test_export_bundle_refuses_nonempty_directory_without_force(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    bundle_export.export_bundle(run_id, str(out_dir))
    with pytest.raises(bundle_export.BundleExportError, match="not empty"):
        bundle_export.export_bundle(run_id, str(out_dir))


def test_export_bundle_force_overwrites(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    bundle_export.export_bundle(run_id, str(out_dir))
    manifest = bundle_export.export_bundle(run_id, str(out_dir), force=True)
    assert manifest["run_id"] == run_id


# =============================================================================================== honesty

def test_manifest_honesty_block_lists_every_hash_verified_file_and_states_the_live_limit(isolated):
    run_id = _record()
    out_dir = isolated / "bundle"
    manifest = bundle_export.export_bundle(run_id, str(out_dir))
    honesty = manifest["honesty"]
    assert set(honesty["hash_verified_offline"]) == {a["path"] for a in manifest["artifacts"]}
    assert honesty["live_reproduction"]["proven_offline"] is False
    assert "model_sha256" in " ".join(honesty["live_reproduction"]["requires"])


def test_engine_url_is_recorded_verbatim_when_supplied(isolated):
    run_id = _record()
    manifest = bundle_export.export_bundle(
        run_id, str(isolated / "bundle"), engine_url="http://10.0.0.5:8091")
    assert manifest["engine_url"] == "http://10.0.0.5:8091"


def test_engine_url_defaults_to_null_when_not_supplied(isolated):
    run_id = _record()
    manifest = bundle_export.export_bundle(run_id, str(isolated / "bundle"))
    assert manifest["engine_url"] is None


# ==================================================================================================== CLI

def test_cli_export_bundle_writes_files_and_prints_summary(isolated, capsys):
    run_id = _record()
    out_dir = isolated / "bundle_cli"
    assert cli.main(["runs", "export-bundle", run_id, "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "export bundle" in out
    assert (out_dir / "manifest.json").is_file()


def test_cli_export_bundle_refuses_then_succeeds_with_force(isolated, capsys):
    run_id = _record()
    out_dir = isolated / "bundle_cli"
    assert cli.main(["runs", "export-bundle", run_id, "--out", str(out_dir)]) == 0
    capsys.readouterr()
    assert cli.main(["runs", "export-bundle", run_id, "--out", str(out_dir)]) == 1
    assert "not empty" in capsys.readouterr().err
    assert cli.main(["runs", "export-bundle", run_id, "--out", str(out_dir), "--force", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["run_id"] == run_id


def test_cli_export_bundle_missing_run_is_a_user_error(isolated, capsys):
    assert cli.main(["runs", "export-bundle", "run_missing", "--out", str(isolated / "out")]) == 1
    assert "not found" in capsys.readouterr().err


# ============================================================================ no network in generator code

_NETWORK_MODULES = {
    "urllib", "urllib.request", "urllib2", "http", "http.client", "httplib", "socket",
    "ftplib", "smtplib", "requests", "httpx", "aiohttp", "websocket", "websockets",
    "xmlrpc", "telnetlib", "poplib", "imaplib", "nntplib",
}


def _imported_module_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", ["clozn.runs.bundle_export", "clozn.runs.notebook_export"])
def test_generator_module_imports_no_networking_module(module):
    imported = __import__(module, fromlist=["__file__"])
    path = Path(imported.__file__)
    hit = _imported_module_names(path) & _NETWORK_MODULES
    assert not hit, f"{module} imports networking module(s): {hit} -- bundle export must stay offline-only"
