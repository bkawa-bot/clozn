"""``clozn runs export-bundle`` (roadmap Phase 4.5 "Open export"): a self-contained, local directory
export of one run's evidence, plus a generated notebook that can check it.

This module is intentionally offline-only and imports nothing network-capable -- it reads an already
-stored run (``clozn.runs.store``) and writes local files. The ONE deliberate exception to "offline" in
the whole feature is a single, clearly-labeled, opt-in cell inside the GENERATED notebook itself (built by
``clozn.runs.notebook_export``) that may call out to a live engine if the user chooses to run it; nothing
in this module or in the notebook-generation code ever does that while the bundle is being built.

A bundle directory looks like::

    <out_dir>/
      manifest.json        -- clozn.export_bundle.v1: identity, method/scope (verbatim), per-file
                               SHA-256 + byte count for every OTHER artifact below, and an explicit
                               "honesty" block naming what is hash-verified offline vs. only checkable
                               with a live, reachable engine.
      receipt_bundle.json   -- clozn.receipts.bundle's full receipt_bundle.v1 (run, repro, identity,
                               trace, memory, explain, receipts, influence_map, ...).
      influence_map.json    -- present only if the run has a computed context<->answer influence map;
                               the SAME object already embedded in receipt_bundle.json, broken out as its
                               own file so it can be inspected/verified without loading the whole bundle.
      trace.json            -- present only if the run captured a non-empty trace; likewise a standalone
                               copy of receipt_bundle.json's own "trace" field.
      tensors.npz           -- numeric arrays (trace confidence/logprobs/topk_entropy, the influence
                               matrix, the influence baseline logprobs) as a NumPy archive, when NumPy is
                               importable at export time.
      <name>.f32.bin         -- the SAME arrays, one raw little-endian float32 file per array (row-major,
                               flattened), when NumPy is NOT importable. Each file's exact shape/dtype/
                               order is recorded in manifest.json's "arrays"/"shape" fields so a
                               stdlib-only reader can reconstruct it without guessing.
      reproduce.ipynb       -- a plain nbformat-v4 notebook (see notebook_export.py) that loads
                               manifest.json, hash-verifies every artifact above, reconstructs a readable
                               receipt, and offers the one optional live-reproduction cell.

Referential note: this is a snapshot export of ALREADY-COMPUTED evidence. It never triggers scoring,
generation, or influence-map computation as a side effect of exporting -- exactly like
``clozn.receipts.bundle.build``, which this module calls unmodified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
import struct
from typing import Any

from clozn._io import atomic_write_json
from clozn.artifacts.contracts import sha256_file

SCHEMA_VERSION = "clozn.export_bundle.v1"

# The run's own reproduction-identity block (roadmap S4.3) -- copied verbatim, never re-derived.
IDENTITY_KEYS = (
    "model_sha256", "model_path", "model_size_bytes", "template_fingerprint",
    "engine_build", "clozn_version", "captured_at",
)

# trace.* numeric fields worth exporting as tensors when present and well-formed (clozn/runs/trace.py's
# TRACE_KEYS); "tokens"/"alternatives"/"steps"/"reasoning_steps"/"workspace_readouts" are not flat numeric
# arrays and are already fully readable from trace.json, so they are left out of the tensor extraction.
_TRACE_TENSOR_KEYS = ("confidence", "logprobs", "topk_entropy")


class BundleExportError(ValueError):
    """A run's evidence could not be assembled into a portable export bundle."""


def _clozn_version() -> str:
    try:
        from clozn import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _write_json_artifact(out_dir: str, filename: str, obj: Any, *, kind: str,
                         schema: "str | None" = None) -> dict:
    path = os.path.join(out_dir, filename)
    try:
        atomic_write_json(path, obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BundleExportError(f"could not serialize {filename}: {exc}") from exc
    entry = {"path": filename, "kind": kind, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    if schema:
        entry["schema"] = schema
    return entry


def _finite_number(value) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _flat_float_row(value) -> "list[float] | None":
    """A 1-D list of finite numbers, or None if `value` isn't exactly that."""
    if not isinstance(value, list) or not value:
        return None
    out = []
    for item in value:
        number = _finite_number(item)
        if number is None:
            return None
        out.append(number)
    return out


def _rectangular_float_matrix(value) -> "list[list[float]] | None":
    """A non-empty, non-ragged 2-D list of finite numbers, or None if `value` isn't exactly that."""
    if not isinstance(value, list) or not value:
        return None
    rows = []
    width = None
    for row in value:
        parsed = _flat_float_row(row)
        if parsed is None:
            return None
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            return None  # ragged -- not a real matrix, leave it out rather than guess a shape
        rows.append(parsed)
    return rows


def _extract_tensors(receipt: dict) -> "dict[str, list]":
    """Pull already-present numeric evidence out of the receipt as name -> nested-list-of-floats.

    This is a convenience extraction, not new evidence: everything here is already readable from
    trace.json / influence_map.json. A field that isn't a clean rectangular finite-float array is simply
    left out (never guessed at or coerced) -- the JSON copies remain the authoritative record regardless.
    """
    tensors: dict[str, list] = {}
    trace = receipt.get("trace") if isinstance(receipt.get("trace"), dict) else {}
    for key in _TRACE_TENSOR_KEYS:
        row = _flat_float_row(trace.get(key))
        if row is not None:
            tensors[f"trace.{key}"] = row

    influence = receipt.get("influence_map") if isinstance(receipt.get("influence_map"), dict) else {}
    baseline = influence.get("baseline") if isinstance(influence.get("baseline"), dict) else {}
    baseline_row = _flat_float_row(baseline.get("logprobs"))
    if baseline_row is not None:
        tensors["influence_map.baseline.logprobs"] = baseline_row
    matrix = _rectangular_float_matrix(influence.get("matrix"))
    if matrix is not None:
        tensors["influence_map.matrix"] = matrix
    return tensors


def _safe_tensor_filename(name: str) -> str:
    return name.replace(".", "_") + ".f32.bin"


def _shape_of(values: list) -> "tuple[int, ...]":
    if values and isinstance(values[0], list):
        return (len(values), len(values[0]))
    return (len(values),)


def _flatten(values: list) -> "list[float]":
    if values and isinstance(values[0], list):
        return [item for row in values for item in row]
    return list(values)


def _write_tensors_npz(out_dir: str, arrays: "dict[str, list]") -> "list[dict]":
    import numpy as np

    path = os.path.join(out_dir, "tensors.npz")
    np_arrays = {name: np.asarray(values, dtype="<f4") for name, values in arrays.items()}
    np.savez(path, **np_arrays)
    entry = {
        "path": "tensors.npz", "kind": "tensor_payload", "format": "npz", "dtype": "float32",
        "sha256": sha256_file(path), "bytes": os.path.getsize(path),
        "arrays": {name: list(int(dim) for dim in arr.shape) for name, arr in np_arrays.items()},
    }
    return [entry]


def _write_tensors_raw_bin(out_dir: str, arrays: "dict[str, list]") -> "list[dict]":
    entries = []
    for name, values in sorted(arrays.items()):
        shape = _shape_of(values)
        flat = _flatten(values)
        filename = _safe_tensor_filename(name)
        path = os.path.join(out_dir, filename)
        payload = struct.pack(f"<{len(flat)}f", *flat)
        with open(path, "wb") as handle:
            handle.write(payload)
        entries.append({
            "path": filename, "kind": "tensor_payload", "format": "raw_f32_bin", "dtype": "float32",
            "array_name": name, "shape": list(shape), "order": "row_major",
            "sha256": sha256_file(path), "bytes": os.path.getsize(path),
        })
    return entries


def _write_tensors(out_dir: str, arrays: "dict[str, list]") -> "list[dict]":
    if not arrays:
        return []
    try:
        return _write_tensors_npz(out_dir, arrays)
    except ImportError:
        return _write_tensors_raw_bin(out_dir, arrays)


def export_bundle(run_id: str, out_dir: str, *, engine_url: "str | None" = None,
                  force: bool = False) -> dict:
    """Build a self-contained export bundle for one stored run under `out_dir`, returning the manifest.

    Refuses (typed `BundleExportError`) rather than silently overwriting an existing, non-empty
    directory unless `force=True`; even then, only this export's own filenames are (re)written -- an
    unrelated file already in `out_dir` is left alone.
    """
    from clozn.receipts import bundle as receipt_bundle
    from clozn.receipts import explain as explain_mod
    from . import store

    if not isinstance(run_id, str) or not store._valid_rid(run_id):
        raise BundleExportError("run_id must be an exact valid run ID")
    run = store.get_run(run_id)
    if run is None:
        raise BundleExportError(f"run {run_id!r} was not found")

    target = os.path.abspath(os.fspath(out_dir))
    if os.path.isdir(target) and os.listdir(target) and not force:
        raise BundleExportError(f"{target} already exists and is not empty; pass force=True to overwrite")
    os.makedirs(target, exist_ok=True)

    try:
        try:
            xr = explain_mod.explain(run)
        except Exception:
            xr = None  # M1 explain is best-effort evidence, never a reason to fail the whole export
        receipt = receipt_bundle.build(run, explain=xr)

        artifacts = [_write_json_artifact(
            target, "receipt_bundle.json", receipt,
            kind="receipt_bundle", schema=receipt.get("schema_version"))]

        influence = receipt.get("influence_map")
        if isinstance(influence, dict) and influence:
            artifacts.append(_write_json_artifact(
                target, "influence_map.json", influence,
                kind="influence_map_evidence", schema=influence.get("schema")))

        trace = receipt.get("trace")
        if isinstance(trace, dict) and trace:
            artifacts.append(_write_json_artifact(target, "trace.json", trace, kind="trace_evidence"))

        artifacts.extend(_write_tensors(target, _extract_tensors(receipt)))

        identity = {k: v for k, v in (receipt.get("identity") or {}).items() if k in IDENTITY_KEYS}
        method = influence.get("method") if isinstance(influence, dict) else None
        influence_identity = influence.get("identity") if isinstance(influence, dict) else None
        scope = (influence_identity.get("prompt_view")
                if isinstance(influence_identity, dict) else None)

        manifest = {
            "schema": SCHEMA_VERSION,
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clozn_version": _clozn_version(),
            "identity": identity,
            "method": method,
            "scope": scope,
            "engine_url": engine_url,
            "artifacts": artifacts,
            "honesty": {
                "note": (
                    "Every file listed in 'artifacts' is byte-for-byte SHA-256 verifiable offline from "
                    "this manifest alone -- see reproduce.ipynb's verification cell, which does exactly "
                    "that and nothing else over the network. Re-running the teacher-forced score to "
                    "check NUMERIC reproducibility additionally requires a reachable clozn engine "
                    "serving the exact model_sha256 recorded in 'identity'; this bundle records what "
                    "that check would need, but cannot itself prove a live result offline."
                ),
                "hash_verified_offline": [a["path"] for a in artifacts],
                "live_reproduction": {
                    "claim": "teacher_forced_sum_logprob_matches_recorded_baseline",
                    "requires": [
                        "a reachable clozn engine (POST /score)",
                        "identity.model_sha256 matching the engine's currently loaded model",
                        "the exact recorded prompt/continuation text (run.final_prompt / run.response)",
                    ],
                    "proven_offline": False,
                },
            },
        }

        from . import notebook_export
        notebook = notebook_export.build_reproduction_notebook(manifest, receipt)
        notebook_artifact = _write_json_artifact(
            target, "reproduce.ipynb", notebook, kind="notebook", schema="nbformat_v4")
        manifest["artifacts"].append(notebook_artifact)
        manifest["honesty"]["hash_verified_offline"].append(notebook_artifact["path"])

        atomic_write_json(os.path.join(target, "manifest.json"), manifest,
                          indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except BundleExportError:
        raise
    except Exception as exc:
        raise BundleExportError(f"could not export bundle for run {run_id!r}: "
                                f"{type(exc).__name__}: {exc}") from exc
    return manifest


def verify_bundle(out_dir: str) -> "list[dict]":
    """Re-hash every artifact manifest.json lists and report OK / TAMPERED / MISSING per file.

    Pure local read, no clozn.runs.store involved -- this is the same check `reproduce.ipynb` runs on its
    own (duplicated there in plain stdlib, since the notebook must not import clozn); this function exists
    so the identical check has a real, tested Python entry point too.
    """
    import json

    target = os.path.abspath(os.fspath(out_dir))
    manifest_path = os.path.join(target, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise BundleExportError(f"could not read {manifest_path}: {exc}") from exc

    results = []
    for artifact in manifest.get("artifacts") or []:
        path = os.path.join(target, str(artifact.get("path")))
        if not os.path.isfile(path):
            results.append({"path": artifact.get("path"), "status": "MISSING"})
            continue
        actual = sha256_file(path)
        expected = artifact.get("sha256")
        results.append({
            "path": artifact.get("path"),
            "status": "OK" if actual == expected else "TAMPERED",
            "sha256": actual,
        })
    return results


__all__ = ["BundleExportError", "SCHEMA_VERSION", "export_bundle", "verify_bundle"]
