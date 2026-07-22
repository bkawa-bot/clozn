"""Portable provenance records for reproducible Clozn experiment runs."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .engine import EngineClient
from .models import JsonObject, require_object


@dataclass(frozen=True)
class ReproducibilityRecord:
    """Environment and engine identity attached to a set of manifests.

    ``sha256`` is computed from the stable payload and intentionally excludes
    ``captured_at`` so repeated captures of the same environment have the same identity.
    """

    client_version: str
    python: str
    implementation: str
    platform: str
    engine_url: str
    engine_health: JsonObject
    manifest_sha256: tuple[str, ...]
    captured_at: str
    metadata: JsonObject = field(default_factory=dict)

    SCHEMA = "clozn.reproducibility.v1"

    def __post_init__(self) -> None:
        if not self.client_version:
            raise ValueError("client_version must be non-empty")
        if not self.engine_url:
            raise ValueError("engine_url must be non-empty")
        for digest in self.manifest_sha256:
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("manifest_sha256 entries must be sha256 strings")
        if len(set(self.manifest_sha256)) != len(self.manifest_sha256):
            raise ValueError("manifest_sha256 entries must be unique")

    def stable_json_object(self) -> JsonObject:
        return {
            "schema": self.SCHEMA,
            "client_version": self.client_version,
            "python": self.python,
            "implementation": self.implementation,
            "platform": self.platform,
            "engine_url": self.engine_url,
            "engine_health": dict(self.engine_health),
            "manifest_sha256": list(self.manifest_sha256),
            "metadata": dict(self.metadata),
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.stable_json_object(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_json_object(self) -> JsonObject:
        return {
            **self.stable_json_object(),
            "captured_at": self.captured_at,
            "sha256": self.sha256,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, indent=indent, allow_nan=False) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ReproducibilityRecord":
        obj = require_object(value, "reproducibility record")
        if obj.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported reproducibility schema: {obj.get('schema')!r}")
        health = require_object(obj.get("engine_health"), "engine_health")
        digests = obj.get("manifest_sha256")
        if not isinstance(digests, list) or not all(isinstance(x, str) for x in digests):
            raise ValueError("manifest_sha256 must be an array of strings")
        metadata = require_object(obj.get("metadata", {}), "metadata")
        record = cls(
            client_version=str(obj.get("client_version", "")),
            python=str(obj.get("python", "")),
            implementation=str(obj.get("implementation", "")),
            platform=str(obj.get("platform", "")),
            engine_url=str(obj.get("engine_url", "")),
            engine_health=health,
            manifest_sha256=tuple(digests),
            captured_at=str(obj.get("captured_at", "")),
            metadata=metadata,
        )
        supplied = obj.get("sha256")
        if supplied is not None and supplied != record.sha256:
            raise ValueError("reproducibility record sha256 does not match its contents")
        return record

    @classmethod
    def read(cls, path: str | Path) -> "ReproducibilityRecord":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def capture_reproducibility(
    engine: EngineClient,
    manifest_sha256: Iterable[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ReproducibilityRecord:
    """Capture a content-addressed environment and native-engine snapshot."""
    if not isinstance(engine, EngineClient):
        raise ValueError("engine must be an EngineClient")
    from . import __version__

    digests = tuple(sorted(set(manifest_sha256)))
    return ReproducibilityRecord(
        client_version=__version__,
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        engine_url=engine.base_url,
        engine_health=engine.health(),
        manifest_sha256=digests,
        captured_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        metadata=dict(metadata or {}),
    )
