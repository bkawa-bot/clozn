"""Deterministic batch replay for mixed Clozn experiment manifests."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._transport import CloznClientError
from .engine import EngineClient
from .manifests import InterventionManifest
from .models import JsonObject
from .patch_manifests import PatchSweepManifest

Manifest = InterventionManifest | PatchSweepManifest


@dataclass(frozen=True)
class BatchItemResult:
    path: str
    schema: str
    name: str
    manifest_sha256: str
    status: str
    result_path: str | None = None
    error: str | None = None

    def to_json_object(self) -> JsonObject:
        return {
            "path": self.path,
            "schema": self.schema,
            "name": self.name,
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
            "result_path": self.result_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class BatchRunResult:
    items: tuple[BatchItemResult, ...]
    output_dir: str
    metadata: JsonObject = field(default_factory=dict)

    @property
    def succeeded(self) -> int:
        return sum(item.status == "ok" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "error" for item in self.items)

    def to_json_object(self) -> JsonObject:
        return {
            "schema": "clozn.batch_run.v1",
            "output_dir": self.output_dir,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "items": [item.to_json_object() for item in self.items],
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, indent=indent, allow_nan=False) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def run_manifest_batch(
    engine: EngineClient,
    manifests: Iterable[tuple[str | Path, Manifest]],
    output_dir: str | Path,
    *,
    continue_on_error: bool = True,
) -> BatchRunResult:
    """Replay manifests in supplied order and write one hash-addressed result per item."""
    if not isinstance(engine, EngineClient):
        raise ValueError("engine must be an EngineClient")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    items: list[BatchItemResult] = []
    for source, manifest in manifests:
        if not isinstance(manifest, (InterventionManifest, PatchSweepManifest)):
            raise ValueError("batch entries must contain supported manifest objects")
        result_name = f"{manifest.sha256}.result.json"
        result_path = output / result_name
        try:
            if isinstance(manifest, InterventionManifest):
                text = engine.run_manifest(manifest).to_json()
            else:
                text = engine.run_patch_manifest(manifest).to_json()
            result_path.write_text(text, encoding="utf-8")
            items.append(BatchItemResult(
                path=str(source), schema=manifest.SCHEMA, name=manifest.name,
                manifest_sha256=manifest.sha256, status="ok", result_path=str(result_path),
            ))
        except (CloznClientError, OSError, ValueError) as exc:
            items.append(BatchItemResult(
                path=str(source), schema=manifest.SCHEMA, name=manifest.name,
                manifest_sha256=manifest.sha256, status="error", error=str(exc),
            ))
            if not continue_on_error:
                break
    return BatchRunResult(items=tuple(items), output_dir=str(output))
