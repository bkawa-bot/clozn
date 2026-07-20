"""Versioned, fail-closed contracts for forward-only product artifacts.

J-lenses, calibrated dial bundles, and future SAE/readout artifacts are not
portable merely because two model filenames look similar.  They are tied to an
exact tokenizer and residual-space contract, and their application must be
qualified against one or more exact GGUF files.  This module is deliberately
stdlib-only and does not import Torch or the lab runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from clozn.cli.fit_planner import gguf_header_from_path


CONTRACT_VERSION = 1
_MODEL_FIELDS = (
    "architecture",
    "hidden_size",
    "layer_count",
    "vocab_size",
    "tokenizer_sha256",
)


class ArtifactContractError(ValueError):
    """An artifact is malformed, corrupt, or incompatible with the model."""


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    """Return the lowercase SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_contract(metadata: Mapping[str, object]) -> dict:
    """The GGUF tokenizer payload, including its chat template when present."""
    return {
        key: metadata[key]
        for key in sorted(metadata)
        if key.startswith("tokenizer.")
    }


def gguf_identity(path: str | os.PathLike[str], *, include_file_hash: bool = True) -> dict:
    """Read an exact, model-agnostic identity from a local GGUF.

    Header fields protect the activation and tokenizer contract.  The whole-file
    digest pins the exact quantized checkpoint used for qualification.  Callers
    performing only inventory may disable the expensive whole-file hash; artifact
    validation always requires it.
    """
    resolved = os.path.abspath(os.fspath(path))
    header = gguf_header_from_path(resolved)
    metadata = header["metadata"]
    tokenizer = _tokenizer_contract(metadata)
    tokens = metadata.get("tokenizer.ggml.tokens")
    vocab_size = len(tokens) if isinstance(tokens, list) else metadata.get("tokenizer.ggml.vocab_size")
    identity = {
        "name": header.get("name"),
        "architecture": header.get("arch"),
        "hidden_size": header.get("embedding_length"),
        "layer_count": header.get("n_layers"),
        "vocab_size": vocab_size,
        "tokenizer_sha256": _json_digest(tokenizer),
        "chat_template_sha256": _json_digest(metadata.get("tokenizer.chat_template")),
        "quantization": header.get("quant"),
        "file_size": int(header["file_size_bytes"]),
        "filename": os.path.basename(resolved),
    }
    if include_file_hash:
        # Routed through the persistent (path, size, mtime_ns) cache: spawn_engine calls this on
        # EVERY engine boot, and an uncached whole-file SHA of a 5-8 GB GGUF costs tens of seconds
        # per launch of an UNCHANGED file. Lazy import -- clozn.runs.identity imports this module
        # for sha256_file, so a top-level import here would be a cycle. Falls back to the direct
        # (raising) hash when the cached path returns None, preserving the strict behavior
        # validation callers rely on: a real IO error still raises, never silently omits.
        from clozn.runs.identity import model_sha256
        cached = model_sha256(resolved)
        identity["sha256"] = cached if cached else sha256_file(resolved)
    return identity


def _require_hex_digest(value, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ArtifactContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def validate_artifact_manifest(
    manifest: Mapping[str, object],
    model_identity: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    *,
    expected_type: str | None = None,
) -> dict:
    """Validate a manifest and every declared payload against one loaded GGUF.

    The manifest shape is::

        {
          "contract_version": 1,
          "artifact_type": "jlens",
          "artifact_version": 1,
          "model": {
            "source_id": "Qwen/Qwen2.5-7B-Instruct",
            "architecture": "qwen2", "hidden_size": 3584,
            "layer_count": 28, "vocab_size": 152064,
            "tokenizer_sha256": "...",
            "compatible_gguf_sha256": ["..."]
          },
          "files": {"J_layer14.f16": {"sha256": "...", "bytes": 25690112}}
        }

    A lab artifact may be fitted on BF16/NF4 and transferred to more than one
    quant, but each product GGUF digest must be listed only after qualification.
    """
    if not isinstance(manifest, Mapping):
        raise ArtifactContractError("artifact manifest must be an object")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ArtifactContractError(
            f"unsupported artifact contract_version {manifest.get('contract_version')!r}; "
            f"expected {CONTRACT_VERSION}"
        )
    artifact_type = str(manifest.get("artifact_type") or "")
    if not artifact_type:
        raise ArtifactContractError("artifact_type is required")
    if expected_type is not None and artifact_type != expected_type:
        raise ArtifactContractError(
            f"artifact_type {artifact_type!r} does not match expected {expected_type!r}"
        )
    if not isinstance(manifest.get("artifact_version"), int):
        raise ArtifactContractError("artifact_version must be an integer")

    target = manifest.get("model")
    if not isinstance(target, Mapping):
        raise ArtifactContractError("model contract is required")
    for field in _MODEL_FIELDS:
        if target.get(field) is None:
            raise ArtifactContractError(f"model.{field} is required")
        if target[field] != model_identity.get(field):
            raise ArtifactContractError(
                f"model.{field} mismatch: artifact has {target[field]!r}, "
                f"GGUF has {model_identity.get(field)!r}"
            )

    actual_model_digest = _require_hex_digest(model_identity.get("sha256"), "GGUF sha256")
    compatible = target.get("compatible_gguf_sha256")
    if not isinstance(compatible, list) or not compatible:
        raise ArtifactContractError("model.compatible_gguf_sha256 must be a non-empty list")
    allowed = [_require_hex_digest(value, "compatible GGUF sha256") for value in compatible]
    if actual_model_digest not in allowed:
        raise ArtifactContractError(
            f"artifact is not qualified for GGUF sha256 {actual_model_digest}"
        )

    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ArtifactContractError("files must declare at least one payload")
    root = Path(artifact_dir).resolve()
    checked = []
    for relative, spec in files.items():
        if not isinstance(relative, str) or not relative or not isinstance(spec, Mapping):
            raise ArtifactContractError("each files entry must map a relative path to an object")
        payload = (root / relative).resolve()
        try:
            payload.relative_to(root)
        except ValueError:
            raise ArtifactContractError(f"artifact payload escapes its directory: {relative}") from None
        if not payload.is_file():
            raise ArtifactContractError(f"artifact payload is missing: {relative}")
        expected_bytes = spec.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ArtifactContractError(f"files[{relative!r}].bytes must be a non-negative integer")
        actual_bytes = payload.stat().st_size
        if actual_bytes != expected_bytes:
            raise ArtifactContractError(
                f"artifact payload size mismatch for {relative}: expected {expected_bytes}, got {actual_bytes}"
            )
        expected_digest = _require_hex_digest(spec.get("sha256"), f"files[{relative!r}].sha256")
        actual_digest = sha256_file(payload)
        if actual_digest != expected_digest:
            raise ArtifactContractError(
                f"artifact payload checksum mismatch for {relative}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        checked.append(relative)

    return {
        "artifact_type": artifact_type,
        "artifact_version": manifest["artifact_version"],
        "model_sha256": actual_model_digest,
        "files": checked,
    }


def find_compatible_artifact(
    artifact_type: str,
    model_identity: Mapping[str, object],
    root: str | os.PathLike[str],
    *,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    """Find exactly one valid artifact directory for a GGUF.

    An explicitly selected directory is strict: a missing, legacy, corrupt, or
    incompatible manifest is an error. Automatic discovery ignores artifacts
    qualified for other models, but refuses ambiguity or corruption in a
    candidate that otherwise claims this GGUF digest.
    """
    if explicit_dir is not None:
        candidates = [Path(explicit_dir)]
        strict = True
    else:
        base = Path(root) / artifact_type
        if not base.is_dir():
            return None
        candidates = sorted({path.parent for path in base.rglob("manifest.json")})
        strict = False

    actual_digest = str(model_identity.get("sha256") or "").lower()
    matches: list[str] = []
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            if strict:
                raise ArtifactContractError(f"artifact manifest is missing: {manifest_path}")
            continue
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as error:
            if strict:
                raise ArtifactContractError(
                    f"artifact manifest could not be read: {manifest_path}: {error}"
                ) from None
            continue

        model_contract = manifest.get("model") if isinstance(manifest, Mapping) else None
        claimed = (model_contract.get("compatible_gguf_sha256")
                   if isinstance(model_contract, Mapping) else None)
        claims_this_model = isinstance(claimed, list) and actual_digest in {
            str(value).lower() for value in claimed
        }
        if not strict and not claims_this_model:
            continue
        validate_artifact_manifest(
            manifest, model_identity, directory, expected_type=artifact_type
        )
        matches.append(str(directory.resolve()))

    if len(matches) > 1:
        raise ArtifactContractError(
            f"multiple {artifact_type} artifacts claim GGUF sha256 {actual_digest}: "
            + ", ".join(matches)
        )
    return matches[0] if matches else None
