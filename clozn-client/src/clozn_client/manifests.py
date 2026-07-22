"""Versioned, replayable native scoring experiment manifests.

The manifest deliberately models only contracts the installable client can replay today:
teacher-forced scoring, optional steering, and explicit attention-edge knockouts. It does not
pretend that an arbitrary hook vocabulary is already supported by the engine.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ._transport import CloznProtocolError
from .models import AttentionKnockout, JsonObject, ScoreResult, require_object


def _ids(value: object, label: str) -> tuple[int, ...]:
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


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CloznProtocolError(f"{label} must be a numeric array")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise CloznProtocolError(f"{label} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise CloznProtocolError(f"{label} must contain finite numbers")
        result.append(number)
    if not result:
        raise CloznProtocolError(f"{label} must not be empty")
    return tuple(result)


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


@dataclass(frozen=True)
class ScoreRequest:
    """The immutable teacher-forced request shared by every experiment arm."""

    prompt: str | None = None
    prompt_ids: tuple[int, ...] | None = None
    continuation: str | None = None
    continuation_ids: tuple[int, ...] | None = None
    topk: int = 0

    def __post_init__(self) -> None:
        if (self.prompt is None) == (self.prompt_ids is None):
            raise ValueError("provide exactly one of prompt or prompt_ids")
        if (self.continuation is None) == (self.continuation_ids is None):
            raise ValueError("provide exactly one of continuation or continuation_ids")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if self.continuation is not None and not isinstance(self.continuation, str):
            raise ValueError("continuation must be a string")
        try:
            if self.prompt_ids is not None:
                object.__setattr__(self, "prompt_ids", _ids(self.prompt_ids, "prompt_ids"))
            if self.continuation_ids is not None:
                object.__setattr__(
                    self,
                    "continuation_ids",
                    _ids(self.continuation_ids, "continuation_ids"),
                )
        except CloznProtocolError as exc:
            raise ValueError(str(exc)) from None
        if not isinstance(self.topk, int) or isinstance(self.topk, bool) or self.topk < 0:
            raise ValueError("topk must be a non-negative integer")

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {"topk": self.topk}
        if self.prompt_ids is not None:
            value["prompt_ids"] = list(self.prompt_ids)
        else:
            value["prompt"] = self.prompt
        if self.continuation_ids is not None:
            value["continuation_ids"] = list(self.continuation_ids)
        else:
            value["continuation"] = self.continuation
        return value

    @classmethod
    def from_json(cls, value: object) -> "ScoreRequest":
        obj = require_object(value, "manifest.request")
        topk = obj.get("topk", 0)
        if not isinstance(topk, int) or isinstance(topk, bool) or topk < 0:
            raise CloznProtocolError("manifest.request.topk must be a non-negative integer")
        prompt_ids = (
            None if "prompt_ids" not in obj
            else _ids(obj["prompt_ids"], "manifest.request.prompt_ids")
        )
        continuation_ids = (
            None if "continuation_ids" not in obj
            else _ids(obj["continuation_ids"], "manifest.request.continuation_ids")
        )
        try:
            return cls(
                prompt=obj.get("prompt"),
                prompt_ids=prompt_ids,
                continuation=obj.get("continuation"),
                continuation_ids=continuation_ids,
                topk=topk,
            )
        except ValueError as exc:
            raise CloznProtocolError(f"invalid manifest.request: {exc}") from None


@dataclass(frozen=True)
class InterventionArm:
    """One named intervention condition compared with an automatic no-intervention baseline."""

    name: str
    attention_knockout: tuple[AttentionKnockout, ...] = ()
    steer: JsonObject | None = None
    steer_vec: tuple[float, ...] | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "arm.name"))
        specs = tuple(self.attention_knockout)
        if any(not isinstance(spec, AttentionKnockout) for spec in specs):
            raise ValueError("attention_knockout entries must be AttentionKnockout objects")
        object.__setattr__(self, "attention_knockout", specs)
        if self.steer is not None:
            normalized_steer = _json_object(self.steer, "steer")
            if not normalized_steer:
                raise ValueError("steer must be a non-empty object")
            object.__setattr__(self, "steer", normalized_steer)
        if self.steer_vec is not None:
            try:
                object.__setattr__(self, "steer_vec", _float_tuple(self.steer_vec, "steer_vec"))
            except CloznProtocolError as exc:
                raise ValueError(str(exc)) from None
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))
        if not specs and self.steer is None and self.steer_vec is None:
            raise ValueError("an intervention arm must define a knockout, steer, or steer_vec")

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {"name": self.name}
        if self.attention_knockout:
            value["attention_knockout"] = [spec.to_wire() for spec in self.attention_knockout]
        if self.steer is not None:
            value["steer"] = dict(self.steer)
        if self.steer_vec is not None:
            value["steer_vec"] = list(self.steer_vec)
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_json(cls, value: object) -> "InterventionArm":
        obj = require_object(value, "manifest.arm")
        knockout_value = obj.get("attention_knockout", [])
        if isinstance(knockout_value, (str, bytes, bytearray)) or not isinstance(knockout_value, Sequence):
            raise CloznProtocolError("manifest.arm.attention_knockout must be an array")
        knockouts: list[AttentionKnockout] = []
        for index, raw_spec in enumerate(knockout_value):
            spec = require_object(raw_spec, f"manifest.arm.attention_knockout[{index}]")
            queries = _ids(spec.get("queries"), f"manifest.arm.attention_knockout[{index}].queries")
            keys = _ids(spec.get("keys"), f"manifest.arm.attention_knockout[{index}].keys")
            layer = spec.get("layer")
            renormalize = spec.get("renormalize", True)
            if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
                raise CloznProtocolError(
                    f"manifest.arm.attention_knockout[{index}].layer must be a non-negative integer"
                )
            if not isinstance(renormalize, bool):
                raise CloznProtocolError(
                    f"manifest.arm.attention_knockout[{index}].renormalize must be a bool"
                )
            knockouts.append(AttentionKnockout(layer, queries, keys, renormalize))
        steer = obj.get("steer")
        if steer is not None:
            steer = require_object(steer, "manifest.arm.steer")
        steer_vec = (
            None if "steer_vec" not in obj
            else _float_tuple(obj["steer_vec"], "manifest.arm.steer_vec")
        )
        metadata = require_object(obj.get("metadata", {}), "manifest.arm.metadata")
        try:
            return cls(
                name=_name(obj.get("name"), "manifest.arm.name"),
                attention_knockout=tuple(knockouts),
                steer=steer,
                steer_vec=steer_vec,
                metadata=metadata,
            )
        except ValueError as exc:
            raise CloznProtocolError(f"invalid manifest.arm: {exc}") from None


@dataclass(frozen=True)
class InterventionManifest:
    """Portable v1 experiment manifest with one baseline and named intervention arms."""

    SCHEMA: ClassVar[str] = "clozn.intervention_manifest.v1"

    name: str
    request: ScoreRequest
    arms: tuple[InterventionArm, ...]
    expected_health: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "manifest.name"))
        if not isinstance(self.request, ScoreRequest):
            raise ValueError("request must be a ScoreRequest")
        arms = tuple(self.arms)
        if not arms:
            raise ValueError("manifest must contain at least one intervention arm")
        if any(not isinstance(arm, InterventionArm) for arm in arms):
            raise ValueError("arms must contain InterventionArm objects")
        names = [arm.name for arm in arms]
        if len(set(names)) != len(names):
            raise ValueError("manifest arm names must be unique")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(
            self, "expected_health", _json_object(self.expected_health, "expected_health")
        )
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {
            "schema": self.SCHEMA,
            "name": self.name,
            "request": self.request.to_json_object(),
            "arms": [arm.to_json_object() for arm in self.arms],
        }
        if self.expected_health:
            value["expected_health"] = dict(self.expected_health)
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_json_object(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_json_object(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    @classmethod
    def from_json(cls, value: str | bytes | bytearray | Mapping[str, Any]) -> "InterventionManifest":
        if isinstance(value, Mapping):
            obj = dict(value)
        else:
            try:
                obj = require_object(json.loads(value), "manifest")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise CloznProtocolError(f"manifest is not valid JSON: {exc}") from None
        schema = obj.get("schema")
        if schema != cls.SCHEMA:
            raise CloznProtocolError(f"unsupported manifest schema {schema!r}; expected {cls.SCHEMA!r}")
        arms_value = obj.get("arms")
        if isinstance(arms_value, (str, bytes, bytearray)) or not isinstance(arms_value, Sequence):
            raise CloznProtocolError("manifest.arms must be an array")
        try:
            return cls(
                name=_name(obj.get("name"), "manifest.name"),
                request=ScoreRequest.from_json(obj.get("request")),
                arms=tuple(InterventionArm.from_json(item) for item in arms_value),
                expected_health=require_object(obj.get("expected_health", {}), "manifest.expected_health"),
                metadata=require_object(obj.get("metadata", {}), "manifest.metadata"),
            )
        except ValueError as exc:
            raise CloznProtocolError(f"invalid manifest: {exc}") from None

    save = write

    @classmethod
    def read(cls, path: str | Path) -> "InterventionManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    load = read


@dataclass(frozen=True)
class InterventionArmResult:
    name: str
    score: ScoreResult
    support_drop: float
    metadata: JsonObject = field(default_factory=dict)

    def to_json_object(self) -> JsonObject:
        return {
            "name": self.name,
            "support_drop": self.support_drop,
            "score": self.score.raw,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InterventionRunResult:
    """The replay result. Positive support_drop means the intervention hurt the continuation."""

    manifest_sha256: str
    baseline: ScoreResult
    arms: tuple[InterventionArmResult, ...]
    engine_health: JsonObject = field(default_factory=dict)

    def to_json_object(self) -> JsonObject:
        return {
            "schema": "clozn.intervention_run.v1",
            "manifest_sha256": self.manifest_sha256,
            "baseline": self.baseline.raw,
            "arms": [arm.to_json_object() for arm in self.arms],
            "engine_health": dict(self.engine_health),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_json_object(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
