"""Stable typed portions of Clozn gateway and engine responses."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._transport import CloznProtocolError

JsonObject = dict[str, Any]


def require_object(value: Any, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise CloznProtocolError(f"{label} must be a JSON object")
    return dict(value)


def _objects(value: Any, label: str) -> tuple[JsonObject, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CloznProtocolError(f"{label} must be a JSON array")
    return tuple(require_object(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class Run:
    id: str
    source: str | None
    model: str | None
    response: str | None
    created_at: str | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "Run":
        obj = require_object(value, "run")
        run_id = obj.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise CloznProtocolError("run.id must be a non-empty string")
        return cls(
            id=run_id,
            source=obj.get("source") if isinstance(obj.get("source"), str) else None,
            model=obj.get("model") if isinstance(obj.get("model"), str) else None,
            response=obj.get("response") if isinstance(obj.get("response"), str) else None,
            created_at=obj.get("created_at") if isinstance(obj.get("created_at"), str) else None,
            raw=obj,
        )


@dataclass(frozen=True)
class RunPage:
    runs: tuple[Run, ...]
    cursor: str | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "RunPage":
        obj = require_object(value, "run page")
        rows = obj.get("runs", [])
        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
            raise CloznProtocolError("run page.runs must be an array")
        cursor = obj.get("cursor") or obj.get("next_cursor")
        return cls(
            runs=tuple(Run.from_json(row) for row in rows),
            cursor=cursor if isinstance(cursor, str) else None,
            raw=obj,
        )


@dataclass(frozen=True)
class LatestRun:
    available: bool
    run: Run | None
    association: JsonObject
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "LatestRun":
        obj = require_object(value, "latest run response")
        available = bool(obj.get("available"))
        run_value = obj.get("run")
        run = None if run_value is None else Run.from_json(run_value)
        if available != (run is not None):
            raise CloznProtocolError("latest run availability disagrees with the run payload")
        association = obj.get("association", {})
        return cls(available, run, require_object(association, "association"), obj)


@dataclass(frozen=True)
class Timeline:
    run_id: str
    events: tuple[JsonObject, ...]
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "Timeline":
        obj = require_object(value, "timeline")
        run_id = obj.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise CloznProtocolError("timeline.run_id must be a non-empty string")
        return cls(run_id, _objects(obj.get("events", []), "timeline.events"), obj)


@dataclass(frozen=True)
class ReceiptBundle:
    raw: JsonObject

    @classmethod
    def from_json(cls, value: Any) -> "ReceiptBundle":
        return cls(require_object(value, "receipt bundle"))


@dataclass(frozen=True)
class AttentionKnockout:
    """One native attention-edge cut applied during teacher-forced scoring."""

    layer: int
    queries: tuple[int, ...]
    keys: tuple[int, ...]
    renormalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not self.queries:
            raise ValueError("queries must not be empty")
        if not self.keys:
            raise ValueError("keys must not be empty")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (*self.queries, *self.keys)):
            raise ValueError("query/key positions must be non-negative integers")
        if not isinstance(self.renormalize, bool):
            raise ValueError("renormalize must be a bool")

    def to_wire(self) -> JsonObject:
        return {
            "layer": int(self.layer),
            "queries": sorted({int(value) for value in self.queries}),
            "keys": sorted({int(value) for value in self.keys}),
            "renormalize": self.renormalize,
        }


@dataclass(frozen=True)
class ScoreToken:
    id: int
    piece: str
    logprob: float
    topk: tuple[JsonObject, ...]
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "ScoreToken":
        obj = require_object(value, "score token")
        token_id = obj.get("id")
        piece = obj.get("piece")
        logprob = obj.get("logprob")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise CloznProtocolError("score token.id must be an integer")
        if not isinstance(piece, str):
            raise CloznProtocolError("score token.piece must be a string")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise CloznProtocolError("score token.logprob must be a number")
        return cls(token_id, piece, float(logprob), _objects(obj.get("topk", []), "score token.topk"), obj)


@dataclass(frozen=True)
class ScoreResult:
    n_prompt: int
    n_cont: int
    tokens: tuple[ScoreToken, ...]
    sum_logprob: float
    boundary_approximate: bool
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "ScoreResult":
        obj = require_object(value, "score response")
        n_prompt = obj.get("n_prompt")
        n_cont = obj.get("n_cont")
        sum_logprob = obj.get("sum_logprob")
        if not isinstance(n_prompt, int) or isinstance(n_prompt, bool) or n_prompt < 0:
            raise CloznProtocolError("score.n_prompt must be a non-negative integer")
        if not isinstance(n_cont, int) or isinstance(n_cont, bool) or n_cont < 0:
            raise CloznProtocolError("score.n_cont must be a non-negative integer")
        if not isinstance(sum_logprob, (int, float)) or isinstance(sum_logprob, bool):
            raise CloznProtocolError("score.sum_logprob must be a number")
        rows = obj.get("tokens", [])
        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
            raise CloznProtocolError("score.tokens must be an array")
        tokens = tuple(ScoreToken.from_json(row) for row in rows)
        if len(tokens) != n_cont:
            raise CloznProtocolError("score.n_cont disagrees with score.tokens")
        return cls(n_prompt, n_cont, tokens, float(sum_logprob),
                   bool(obj.get("boundary_approximate", False)), obj)


@dataclass(frozen=True)
class HarvestResult:
    """Residual activations returned by ``POST /harvest``."""

    tokens: tuple[str, ...]
    layer: int
    activations: Any = field(repr=False, compare=False)
    raw: JsonObject = field(repr=False, compare=False)

    @property
    def n_tokens(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_embd(self) -> int:
        return int(self.activations.shape[1])

    @classmethod
    def from_json(cls, value: Any) -> "HarvestResult":
        from .tensors import decode_float32_tensor

        obj = require_object(value, "harvest response")
        layer = obj.get("layer")
        tokens_value = obj.get("tokens")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise CloznProtocolError("harvest.layer must be a non-negative integer")
        if (isinstance(tokens_value, (str, bytes, bytearray))
                or not isinstance(tokens_value, Sequence)
                or any(not isinstance(token, str) for token in tokens_value)):
            raise CloznProtocolError("harvest.tokens must be a string array")
        activations = decode_float32_tensor(obj.get("activations"), label="harvest.activations")
        if activations.ndim != 2:
            raise CloznProtocolError("harvest.activations must be a rank-2 tensor")
        tokens = tuple(tokens_value)
        if activations.shape[0] != len(tokens):
            raise CloznProtocolError("harvest token count disagrees with activation rows")
        return cls(tokens=tokens, layer=layer, activations=activations, raw=obj)


@dataclass(frozen=True)
class Observation:
    """Prediction movement reported by one native residual write."""

    applied: bool
    layer: int
    moved_l2: float
    baseline_top: tuple[JsonObject, ...]
    edited_top: tuple[JsonObject, ...]
    error: str | None
    raw: JsonObject = field(repr=False)

    @property
    def shifted(self) -> bool:
        return bool(
            self.applied and self.baseline_top and self.edited_top
            and self.baseline_top[0].get("token") != self.edited_top[0].get("token")
        )

    @classmethod
    def from_json(cls, value: Any) -> "Observation":
        obj = require_object(value, "state observation")
        applied = obj.get("applied")
        layer = obj.get("layer")
        moved_l2 = obj.get("moved_l2")
        error = obj.get("error")
        if not isinstance(applied, bool):
            raise CloznProtocolError("state.applied must be a bool")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise CloznProtocolError("state.layer must be a non-negative integer")
        if not isinstance(moved_l2, (int, float)) or isinstance(moved_l2, bool):
            raise CloznProtocolError("state.moved_l2 must be a number")
        if error is not None and not isinstance(error, str):
            raise CloznProtocolError("state.error must be a string or null")
        return cls(
            applied=applied,
            layer=layer,
            moved_l2=float(moved_l2),
            baseline_top=_objects(obj.get("baseline_top", []), "state.baseline_top"),
            edited_top=_objects(obj.get("edited_top", []), "state.edited_top"),
            error=error,
            raw=obj,
        )


@dataclass(frozen=True)
class PatchArm:
    """One named residual write in a patch sweep."""

    name: str
    positions: tuple[int, ...]
    values: Any = field(repr=False, compare=False)
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("patch arm name must be a non-empty string")
        if not self.positions:
            raise ValueError("patch arm positions must not be empty")
        if any(not isinstance(pos, int) or isinstance(pos, bool) or pos < 0 for pos in self.positions):
            raise ValueError("patch arm positions must be non-negative integers")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("patch arm metadata must be an object")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PatchArmResult:
    name: str
    positions: tuple[int, ...]
    observation: Observation
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class PatchSweepResult:
    harvest: HarvestResult
    arms: tuple[PatchArmResult, ...]
