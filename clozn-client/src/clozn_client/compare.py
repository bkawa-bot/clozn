"""Deterministic comparison and regression gates for Clozn batch results."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._transport import CloznProtocolError
from .models import JsonObject, require_object


@dataclass(frozen=True)
class MetricDelta:
    arm: str
    metric: str
    baseline: float | bool
    candidate: float | bool
    delta: float | None
    regressed: bool

    def to_json_object(self) -> JsonObject:
        return {
            "arm": self.arm, "metric": self.metric, "baseline": self.baseline,
            "candidate": self.candidate, "delta": self.delta, "regressed": self.regressed,
        }


@dataclass(frozen=True)
class ExperimentComparison:
    manifest_sha256: str
    name: str
    schema: str
    status: str
    metrics: tuple[MetricDelta, ...] = ()
    error: str | None = None

    @property
    def regressions(self) -> int:
        return sum(metric.regressed for metric in self.metrics) + (1 if self.status == "regressed" and not self.metrics else 0)

    def to_json_object(self) -> JsonObject:
        return {
            "manifest_sha256": self.manifest_sha256, "name": self.name,
            "schema": self.schema, "status": self.status,
            "regressions": self.regressions,
            "metrics": [metric.to_json_object() for metric in self.metrics], "error": self.error,
        }


@dataclass(frozen=True)
class BatchComparison:
    baseline_index: str
    candidate_index: str
    experiments: tuple[ExperimentComparison, ...]
    max_metric_delta: float
    metadata: JsonObject = field(default_factory=dict)

    SCHEMA = "clozn.batch_comparison.v1"

    @property
    def regressions(self) -> int:
        return sum(item.regressions for item in self.experiments)

    @property
    def unchanged(self) -> int:
        return sum(item.status == "ok" for item in self.experiments)

    def to_json_object(self) -> JsonObject:
        return {
            "schema": self.SCHEMA, "baseline_index": self.baseline_index,
            "candidate_index": self.candidate_index, "max_metric_delta": self.max_metric_delta,
            "regressions": self.regressions, "unchanged": self.unchanged,
            "experiments": [item.to_json_object() for item in self.experiments],
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, indent=indent, allow_nan=False) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _read_json(path: Path, label: str) -> JsonObject:
    try:
        return require_object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise CloznProtocolError(f"invalid {label} JSON: {exc}") from None


def _items(index_path: Path) -> dict[str, JsonObject]:
    obj = _read_json(index_path, "batch index")
    if obj.get("schema") != "clozn.batch_run.v1":
        raise CloznProtocolError(f"unsupported batch index schema: {obj.get('schema')!r}")
    rows = obj.get("items")
    if not isinstance(rows, list):
        raise CloznProtocolError("batch index.items must be an array")
    result: dict[str, JsonObject] = {}
    for i, raw in enumerate(rows):
        row = require_object(raw, f"batch index.items[{i}]")
        digest = row.get("manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CloznProtocolError(f"batch index.items[{i}].manifest_sha256 must be a sha256")
        if digest in result:
            raise CloznProtocolError(f"duplicate manifest_sha256 in batch index: {digest}")
        result[digest] = row
    return result


def _result_path(index_path: Path, row: JsonObject) -> Path:
    value = row.get("result_path")
    if not isinstance(value, str) or not value:
        raise CloznProtocolError("successful batch item is missing result_path")
    path = Path(value)
    return path if path.is_absolute() else index_path.parent / path


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CloznProtocolError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CloznProtocolError(f"{label} must be finite")
    return number


def _extract_metrics(payload: JsonObject) -> dict[tuple[str, str], float | bool]:
    schema = payload.get("schema")
    rows = payload.get("arms")
    if not isinstance(rows, list):
        raise CloznProtocolError("result.arms must be an array")
    metrics: dict[tuple[str, str], float | bool] = {}
    for i, raw in enumerate(rows):
        arm = require_object(raw, f"result.arms[{i}]")
        name = arm.get("name")
        if not isinstance(name, str) or not name:
            raise CloznProtocolError(f"result.arms[{i}].name must be a non-empty string")
        if schema == "clozn.intervention_run.v1":
            metrics[(name, "support_drop")] = _number(arm.get("support_drop"), f"arm {name}.support_drop")
        elif schema == "clozn.patch_sweep_result.v1":
            observation = require_object(arm.get("observation"), f"arm {name}.observation")
            metrics[(name, "moved_l2")] = _number(observation.get("moved_l2"), f"arm {name}.moved_l2")
            shifted = observation.get("shifted")
            if not isinstance(shifted, bool):
                raise CloznProtocolError(f"arm {name}.shifted must be a bool")
            metrics[(name, "shifted")] = shifted
        else:
            raise CloznProtocolError(f"unsupported result schema: {schema!r}")
    return metrics


def compare_batch_runs(
    baseline_index: str | Path,
    candidate_index: str | Path,
    *,
    max_metric_delta: float = 0.0,
) -> BatchComparison:
    """Compare matching manifest hashes; positive numeric movement above tolerance regresses."""
    if not isinstance(max_metric_delta, (int, float)) or isinstance(max_metric_delta, bool) or max_metric_delta < 0:
        raise ValueError("max_metric_delta must be a non-negative number")
    tolerance = float(max_metric_delta)
    base_path, cand_path = Path(baseline_index), Path(candidate_index)
    base, cand = _items(base_path), _items(cand_path)
    experiments: list[ExperimentComparison] = []
    for digest in sorted(set(base) | set(cand)):
        left, right = base.get(digest), cand.get(digest)
        row = right or left or {}
        name = str(row.get("name", ""))
        schema = str(row.get("schema", ""))
        if left is None or right is None:
            experiments.append(ExperimentComparison(digest, name, schema, "regressed", error="manifest missing from one batch"))
            continue
        if left.get("status") != "ok" or right.get("status") != "ok":
            same = left.get("status") == right.get("status") and left.get("error") == right.get("error")
            experiments.append(ExperimentComparison(digest, name, schema, "ok" if same else "regressed", error=None if same else "batch item status/error changed"))
            continue
        left_payload = _read_json(_result_path(base_path, left), "baseline result")
        right_payload = _read_json(_result_path(cand_path, right), "candidate result")
        if left_payload.get("schema") != right_payload.get("schema"):
            experiments.append(ExperimentComparison(digest, name, schema, "regressed", error="result schema changed"))
            continue
        lm, rm = _extract_metrics(left_payload), _extract_metrics(right_payload)
        metric_rows: list[MetricDelta] = []
        for key in sorted(set(lm) | set(rm)):
            if key not in lm or key not in rm:
                metric_rows.append(MetricDelta(key[0], key[1], lm.get(key, False), rm.get(key, False), None, True))
                continue
            lv, rv = lm[key], rm[key]
            if isinstance(lv, bool) or isinstance(rv, bool):
                regressed = bool(rv) and not bool(lv)
                metric_rows.append(MetricDelta(key[0], key[1], bool(lv), bool(rv), None, regressed))
            else:
                delta = float(rv) - float(lv)
                metric_rows.append(MetricDelta(key[0], key[1], float(lv), float(rv), delta, delta > tolerance))
        status = "regressed" if any(metric.regressed for metric in metric_rows) else "ok"
        experiments.append(ExperimentComparison(digest, name, schema, status, tuple(metric_rows)))
    return BatchComparison(str(base_path), str(cand_path), tuple(experiments), tolerance)
