"""Model-free tests for exact GGUF and lab-to-product artifact contracts."""
from __future__ import annotations

import hashlib
import json

import pytest

from clozn.artifacts import contracts
from clozn.cli.commands.models import KNOWN, PULLABLE


def _header(path):
    return {
        "name": "Fixture 3B Instruct",
        "arch": "fixture",
        "embedding_length": 64,
        "n_layers": 8,
        "quant": "Q4_K_M",
        "file_size_bytes": path.stat().st_size,
        "metadata": {
            "tokenizer.ggml.model": "fixture-bpe",
            "tokenizer.ggml.tokens": ["a", "b", "c"],
            "tokenizer.chat_template": "{{ messages }}",
        },
    }


@pytest.fixture
def model(tmp_path, monkeypatch):
    path = tmp_path / "fixture-Q4_K_M.gguf"
    path.write_bytes(b"exact gguf bytes")
    monkeypatch.setattr(contracts, "gguf_header_from_path", lambda _path: _header(path))
    return path, contracts.gguf_identity(path)


def _manifest(identity, payload, **model_overrides):
    target = {
        "source_id": "owner/fixture-3b-instruct",
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
        "artifact_type": "jlens",
        "artifact_version": 1,
        "model": target,
        "files": {
            payload.name: {
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }


def test_gguf_identity_pins_file_tokenizer_and_dimensions(model):
    path, identity = model
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["architecture"] == "fixture"
    assert identity["hidden_size"] == 64
    assert identity["layer_count"] == 8
    assert identity["vocab_size"] == 3
    assert identity["quantization"] == "Q4_K_M"
    assert len(identity["tokenizer_sha256"]) == 64
    assert len(identity["chat_template_sha256"]) == 64


def test_valid_artifact_checks_every_payload(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    manifest = _manifest(identity, payload)
    result = contracts.validate_artifact_manifest(
        manifest, identity, tmp_path, expected_type="jlens"
    )
    assert result == {
        "artifact_type": "jlens",
        "artifact_version": 1,
        "model_sha256": identity["sha256"],
        "files": ["J_layer4.f16"],
    }


@pytest.mark.parametrize("field,bad", [
    ("architecture", "other"),
    ("hidden_size", 128),
    ("layer_count", 9),
    ("vocab_size", 4),
    ("tokenizer_sha256", "0" * 64),
])
def test_model_contract_mismatch_fails_closed(model, tmp_path, field, bad):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    with pytest.raises(contracts.ArtifactContractError, match=field):
        contracts.validate_artifact_manifest(
            _manifest(identity, payload, **{field: bad}), identity, tmp_path
        )


def test_unqualified_gguf_digest_fails_closed(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "dial.bin"
    payload.write_bytes(b"direction")
    manifest = _manifest(identity, payload,
                         compatible_gguf_sha256=["1" * 64])
    with pytest.raises(contracts.ArtifactContractError, match="not qualified"):
        contracts.validate_artifact_manifest(manifest, identity, tmp_path)


def test_corrupt_payload_fails_closed(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    manifest = _manifest(identity, payload)
    payload.write_bytes(b"corruption!")
    with pytest.raises(contracts.ArtifactContractError, match="checksum mismatch"):
        contracts.validate_artifact_manifest(manifest, identity, tmp_path)


def test_manifest_path_cannot_escape_artifact_directory(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x")
    manifest = _manifest(identity, payload)
    spec = manifest["files"].pop(payload.name)
    manifest["files"]["../payload.bin"] = spec
    with pytest.raises(contracts.ArtifactContractError, match="escapes"):
        contracts.validate_artifact_manifest(manifest, identity, tmp_path / "artifact")


def test_wave_one_pull_aliases_are_exact_and_recognized():
    expected = {
        "qwen": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "qwen3.5": "Qwen3.5-9B-Q4_K_M.gguf",
        "llama": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "gemma4": "gemma-4-E4B-it-Q4_K_M.gguf",
        "ministral": "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    }
    for alias, filename in expected.items():
        assert PULLABLE[alias][1] == filename
        assert any(alias == friendly and fragment in filename.lower()
                   for fragment, friendly, _flags in KNOWN)


def test_manifest_example_is_json_serializable(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    json.dumps(_manifest(identity, payload))


def test_discovery_selects_only_the_artifact_qualified_for_this_gguf(model, tmp_path):
    _path, identity = model
    directory = tmp_path / "jlens" / "fixture-v1"
    directory.mkdir(parents=True)
    payload = directory / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    (directory / "manifest.json").write_text(
        json.dumps(_manifest(identity, payload)), encoding="utf-8"
    )
    assert contracts.find_compatible_artifact("jlens", identity, tmp_path) == str(directory.resolve())


def test_explicit_legacy_manifest_is_refused(model, tmp_path):
    _path, identity = model
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"model": "filename-only", "d_model": 64}), encoding="utf-8"
    )
    with pytest.raises(contracts.ArtifactContractError, match="contract_version"):
        contracts.find_compatible_artifact(
            "jlens", identity, tmp_path, explicit_dir=directory
        )
