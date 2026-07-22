"""Portable residual patch-sweep manifests and result artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ._transport import CloznProtocolError
from .models import JsonObject, Observation, PatchArm, PatchSweepResult, require_object


def _name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > 128 or any(ord(ch) < 32 for ch in text):
        raise ValueError(f"{label} must be a non-empty printable string up to 128 characters")
    return text


def _json_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values: {exc}") from None
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be an object")
    return decoded


def _positions(value: object, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CloznProtocolError(f"{label} must be an integer array")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise CloznProtocolError(f"{label} must contain non-negative integers")
        result.append(item)
    if not result:
        raise CloznProtocolError(f"{label} must not be empty")
    return tuple(result)


def _matrix(value: object, label: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CloznProtocolError(f"{label} must be a numeric matrix")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row in value:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise CloznProtocolError(f"{label} must be a numeric matrix")
        normalized: list[float] = []
        for item in row:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise CloznProtocolError(f"{label} must contain numbers")
            number = float(item)
            if not math.isfinite(number):
                raise CloznProtocolError(f"{label} must contain finite numbers")
            normalized.append(number)
        if not normalized:
            raise CloznProtocolError(f"{label} rows must not be empty")
        if width is None:
            width = len(normalized)
        elif len(normalized) != width:
            raise CloznProtocolError(f"{label} rows must have equal length")
        rows.append(tuple(normalized))
    if not rows:
        raise CloznProtocolError(f"{label} must not be empty")
    return tuple(rows)


@dataclass(frozen=True)
class PatchManifestArm:
    """One serializable residual patch arm."""

    name: str
    positions: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "arm.name"))
        try:
            positions = _positions(self.positions, "arm.positions")
            raw_matrix = _matrix(self.values, "arm.values")
        except CloznProtocolError as exc:
            raise ValueError(str(exc)) from None
        if len(raw_matrix) != len(positions):
            raise ValueError(
                f"arm.values row count {len(raw_matrix)} does not match {len(positions)} positions"
            )
        matrix = tuple(
            tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in row)
            for row in raw_matrix
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "metadata", _json_object(self.metadata, "arm.metadata"))

    def to_patch_arm(self) -> PatchArm:
        return PatchArm(self.name, self.positions, self.values, self.metadata)

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {
            "name": self.name,
            "positions": list(self.positions),
            "values": [list(row) for row in self.values],
        }
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_json(cls, value: object) -> "PatchManifestArm":
        obj = require_object(value, "patch manifest arm")
        metadata = require_object(obj.get("metadata", {}), "patch manifest arm.metadata")
        try:
            return cls(
                name=_name(obj.get("name"), "patch manifest arm.name"),
                positions=_positions(obj.get("positions"), "patch manifest arm.positions"),
                values=_matrix(obj.get("values"), "patch manifest arm.values"),
                metadata=metadata,
            )
        except ValueError as exc:
            raise CloznProtocolError(f"invalid patch manifest arm: {exc}") from None


@dataclass(frozen=True)
class PatchSweepManifest:
    """Versioned, hash-addressed residual patch experiment."""

    SCHEMA: ClassVar[str] = "clozn.patch_sweep_manifest.v1"

    name: str
    text: str
    arms: tuple[PatchManifestArm, ...]
    layer: int | None = None
    expected_health: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "manifest.name"))
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("manifest.text must be a non-empty string")
        if self.layer is not None and (
            not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0
        ):
            raise ValueError("manifest.layer must be a non-negative integer or null")
        arms = tuple(self.arms)
        if not arms or any(not isinstance(arm, PatchManifestArm) for arm in arms):
            raise ValueError("manifest.arms must contain PatchManifestArm objects")
        names = [arm.name for arm in arms]
        if len(set(names)) != len(names):
            raise ValueError("manifest arm names must be unique")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "expected_health", _json_object(self.expected_health, "expected_health"))
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {
            "schema": self.SCHEMA,
            "name": self.name,
            "text": self.text,
            "layer": self.layer,
            "arms": [arm.to_json_object() for arm in self.arms],
        }
        if self.expected_health:
            value["expected_health"] = dict(self.expected_health)
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, indent=indent, allow_nan=False) + "\n"

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, value: str | bytes | Mapping[str, Any]) -> "PatchSweepManifest":
        if isinstance(value, (str, bytes)):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise CloznProtocolError(f"invalid patch manifest JSON: {exc}") from None
        obj = require_object(value, "patch manifest")
        if obj.get("schema") != cls.SCHEMA:
            raise CloznProtocolError(f"unsupported patch manifest schema: {obj.get('schema')!r}")
        raw_arms = obj.get("arms")
        if isinstance(raw_arms, (str, bytes, bytearray)) or not isinstance(raw_arms, Sequence):
            raise CloznProtocolError("patch manifest.arms must be an array")
        layer = obj.get("layer")
        expected = require_object(obj.get("expected_health", {}), "patch manifest.expected_health")
        metadata = require_object(obj.get("metadata", {}), "patch manifest.metadata")
        try:
            return cls(
                name=_name(obj.get("name"), "patch manifest.name"),
                text=obj.get("text"),
                layer=layer,
                arms=tuple(PatchManifestArm.from_json(item) for item in raw_arms),
                expected_health=expected,
                metadata=metadata,
            )
        except ValueError as exc:
            raise CloznProtocolError(f"invalid patch manifest: {exc}") from None

    @classmethod
    def read(cls, path: str | Path) -> "PatchSweepManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PatchSweepArtifact:
    """Portable summary of a completed patch sweep."""

    SCHEMA: ClassVar[str] = "clozn.patch_sweep_result.v1"

    manifest_sha256: str
    layer: int
    tokens: tuple[str, ...]
    n_embd: int
    arms: tuple[JsonObject, ...]
    engine_health: JsonObject = field(default_factory=dict)
    contract_binding: JsonObject = field(default_factory=dict)
    capture_statistics: JsonObject = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        manifest: PatchSweepManifest,
        result: PatchSweepResult,
        *,
        engine_health: Mapping[str, Any] | None = None,
    ) -> "PatchSweepArtifact":
        arms: list[JsonObject] = []
        for arm in result.arms:
            observation = arm.observation
            arms.append({
                "name": arm.name,
                "positions": list(arm.positions),
                "metadata": dict(arm.metadata),
                "observation": _observation_json(observation),
            })
        from .contracts import CaptureStatistics, bind_contract_evidence
        binding = bind_contract_evidence(manifest, capture=result.harvest)
        capture_statistics = CaptureStatistics.from_harvest(result.harvest)
        return cls(
            manifest_sha256=manifest.sha256,
            layer=result.harvest.layer,
            tokens=result.harvest.tokens,
            n_embd=result.harvest.n_embd,
            arms=tuple(arms),
            engine_health=_json_object(engine_health or {}, "engine_health"),
            contract_binding=binding.to_json_object(),
            capture_statistics=capture_statistics.to_json_object(),
        )

    def to_json_object(self) -> JsonObject:
        return {
            "schema": self.SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "layer": self.layer,
            "tokens": list(self.tokens),
            "n_embd": self.n_embd,
            "arms": [dict(arm) for arm in self.arms],
            "engine_health": dict(self.engine_health),
            "contract_binding": dict(self.contract_binding),
            "capture_statistics": dict(self.capture_statistics),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, indent=indent, allow_nan=False) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _observation_json(observation: Observation) -> JsonObject:
    return {
        "applied": observation.applied,
        "layer": observation.layer,
        "moved_l2": observation.moved_l2,
        "shifted": observation.shifted,
        "baseline_top": [dict(item) for item in observation.baseline_top],
        "edited_top": [dict(item) for item in observation.edited_top],
        "error": observation.error,
    }
