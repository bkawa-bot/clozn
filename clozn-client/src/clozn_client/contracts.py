"""Minimal public intervention vocabulary for receipt-backed Clozn experiments.

This module intentionally describes only operations already exposed by the native worker.  It is
not a generic hook registry.  New operations belong here only when a Phase 1-3 receipt or
regression workflow consumes them and their native semantics can be qualified.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from ._transport import CloznProtocolError
from .models import JsonObject




class ProtocolCompatibility(str, Enum):
    """Whether the engine explicitly proves compatibility with this client contract."""

    VERIFIED = "verified"
    UNADVERTISED = "unadvertised"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class EngineProtocolAdvertisement:
    """Versioned protocol identity expected in native ``GET /health`` responses."""

    SCHEMA: ClassVar[str] = "clozn.engine_protocol.v1"
    protocol_version: str
    intervention_contract_schema: str
    intervention_contract_sha256: str

    @classmethod
    def current(cls, contract: "InterventionContract") -> "EngineProtocolAdvertisement":
        return cls("1.0", contract.SCHEMA, contract.sha256)

    @classmethod
    def from_health(cls, health: Mapping[str, Any]) -> "EngineProtocolAdvertisement | None":
        raw = health.get("protocol")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise CloznProtocolError("health.protocol must be an object")
        schema = raw.get("schema")
        if schema != cls.SCHEMA:
            raise CloznProtocolError(f"unsupported health.protocol.schema: {schema!r}")
        contract = raw.get("intervention_contract")
        if not isinstance(contract, Mapping):
            raise CloznProtocolError("health.protocol.intervention_contract must be an object")
        version = raw.get("version")
        contract_schema = contract.get("schema")
        digest = contract.get("sha256")
        if not isinstance(version, str) or not version:
            raise CloznProtocolError("health.protocol.version must be a non-empty string")
        if not isinstance(contract_schema, str) or not contract_schema:
            raise CloznProtocolError("health.protocol.intervention_contract.schema must be a non-empty string")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CloznProtocolError("health.protocol.intervention_contract.sha256 must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise CloznProtocolError("health.protocol.intervention_contract.sha256 must be hexadecimal") from exc
        return cls(version, contract_schema, digest.lower())

    def to_json_object(self) -> JsonObject:
        return {
            "schema": self.SCHEMA,
            "version": self.protocol_version,
            "intervention_contract": {
                "schema": self.intervention_contract_schema,
                "sha256": self.intervention_contract_sha256,
            },
        }


class InterventionOperation(str, Enum):
    """Stable operation identifiers in ``clozn.intervention_contract.v1``."""

    SCORE_TEACHER_FORCED = "score.teacher_forced"
    ATTENTION_KNOCKOUT = "intervention.attention_knockout"
    CAPTURE_RESIDUAL = "capture.residual.layer_output"
    REPLACE_RESIDUAL = "intervention.residual.replace_rows"


class ReplayClass(str, Enum):
    """Honest replay guarantees; never infer bit identity from deterministic inputs alone."""

    REQUEST_REPLAY = "request_replay"
    RE_PREFILLED = "re_prefilled"


@dataclass(frozen=True)
class OperationSpec:
    operation: InterventionOperation
    endpoint: str
    semantics: str
    tensor_contract: str | None
    replay_class: ReplayClass
    qualification_requirement: str
    health_capability: str | None = None
    limits: tuple[str, ...] = ()

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {
            "operation": self.operation.value,
            "endpoint": self.endpoint,
            "semantics": self.semantics,
            "replay_class": self.replay_class.value,
            "qualification_requirement": self.qualification_requirement,
            "limits": list(self.limits),
        }
        if self.tensor_contract is not None:
            value["tensor_contract"] = self.tensor_contract
        if self.health_capability is not None:
            value["health_capability"] = self.health_capability
        return value


@dataclass(frozen=True)
class InterventionContract:
    """The deliberately small, versioned contract consumed by receipt workflows."""

    SCHEMA: ClassVar[str] = "clozn.intervention_contract.v1"
    operations: tuple[OperationSpec, ...]
    scope: str = (
        "Minimum native operations required for receipt-backed scoring, attention knockout, "
        "and residual patch experiments; not a general-purpose hook API."
    )

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("contract must define at least one operation")
        names = [spec.operation for spec in operations]
        if len(set(names)) != len(names):
            raise ValueError("contract operation identifiers must be unique")
        object.__setattr__(self, "operations", operations)

    def to_json_object(self) -> JsonObject:
        return {
            "schema": self.SCHEMA,
            "scope": self.scope,
            "operations": [spec.to_json_object() for spec in self.operations],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_json_object(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def spec(self, operation: InterventionOperation | str) -> OperationSpec:
        try:
            target = operation if isinstance(operation, InterventionOperation) else InterventionOperation(operation)
        except ValueError as exc:
            raise KeyError(str(operation)) from exc
        for spec in self.operations:
            if spec.operation is target:
                return spec
        raise KeyError(target.value)


MINIMAL_INTERVENTION_CONTRACT = InterventionContract(operations=(
    OperationSpec(
        operation=InterventionOperation.SCORE_TEACHER_FORCED,
        endpoint="POST /score",
        semantics=(
            "Prefill the supplied prompt and score the supplied continuation token by token without "
            "sampling a replacement continuation."
        ),
        tensor_contract=None,
        replay_class=ReplayClass.REQUEST_REPLAY,
        qualification_requirement="Exact model, tokenizer/template, engine build, and request identity must be retained.",
        limits=(
            "Text continuations may report an approximate prompt/continuation token boundary.",
            "A repeated request is not labeled bit-identical unless the engine proves checkpoint identity.",
        ),
    ),
    OperationSpec(
        operation=InterventionOperation.ATTENTION_KNOCKOUT,
        endpoint="POST /score",
        semantics=(
            "During teacher-forced scoring, remove the named query-to-key attention edges at one "
            "zero-based layer; optionally renormalize the surviving attention weights."
        ),
        tensor_contract="queries and keys are zero-based token positions in the scored sequence",
        replay_class=ReplayClass.REQUEST_REPLAY,
        qualification_requirement="Engine health must advertise capabilities.attn_knockout=true.",
        health_capability="attn_knockout",
        limits=(
            "Requires the native no-flash-attention execution path.",
            "Renormalized and non-renormalized cuts are distinct interventions and must not be conflated.",
        ),
    ),
    OperationSpec(
        operation=InterventionOperation.CAPTURE_RESIDUAL,
        endpoint="POST /harvest",
        semantics=(
            "Prefill the supplied text and return one residual row per token at the native worker's "
            "named layer-output capture seam."
        ),
        tensor_contract="little-endian float32 matrix shaped [n_tokens, n_embd]",
        replay_class=ReplayClass.RE_PREFILLED,
        qualification_requirement="The receipt must retain layer, shape, dtype, model, template, and engine identity.",
        limits=(
            "The v1 contract does not claim pre-norm, post-norm, or arbitrary graph-node capture.",
            "Capture is a fresh prefill, not checkpoint restoration.",
        ),
    ),
    OperationSpec(
        operation=InterventionOperation.REPLACE_RESIDUAL,
        endpoint="POST /state",
        semantics=(
            "Prefill the supplied text, replace complete residual rows at selected token positions at "
            "one layer-output seam, and observe next-token movement."
        ),
        tensor_contract="position-major float32 rows shaped [len(positions), n_embd]",
        replay_class=ReplayClass.RE_PREFILLED,
        qualification_requirement="Replacement values must match the captured layer width and receipt identity.",
        limits=(
            "The v1 operation replaces complete rows; it is not arbitrary tensor surgery.",
            "The returned observation is next-token evidence, not a full generated continuation.",
        ),
    ),
))


@dataclass(frozen=True)
class CaptureBudget:
    """Client-side ceiling for residual captures used in portable receipts.

    The native worker may later advertise stronger server-side limits.  Until then the client
    validates every decoded capture before it can be attached to an evidence artifact.
    """

    max_tokens: int = 4096
    max_elements: int = 8_388_608
    max_bytes: int = 33_554_432

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_elements", "max_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def check(self, *, n_tokens: int, n_embd: int, itemsize: int = 4) -> None:
        elements = n_tokens * n_embd
        n_bytes = elements * itemsize
        failures = []
        if n_tokens > self.max_tokens:
            failures.append(f"tokens {n_tokens} > {self.max_tokens}")
        if elements > self.max_elements:
            failures.append(f"elements {elements} > {self.max_elements}")
        if n_bytes > self.max_bytes:
            failures.append(f"bytes {n_bytes} > {self.max_bytes}")
        if failures:
            raise CloznProtocolError("capture exceeds client budget: " + "; ".join(failures))

    def to_json_object(self) -> JsonObject:
        return {
            "max_tokens": self.max_tokens,
            "max_elements": self.max_elements,
            "max_bytes": self.max_bytes,
        }


DEFAULT_CAPTURE_BUDGET = CaptureBudget()


@dataclass(frozen=True)
class CaptureStatistics:
    """Shape and storage facts printed beside residual evidence."""

    layer: int
    n_tokens: int
    n_embd: int
    elements: int
    n_bytes: int
    dtype: str = "float32-le"

    @classmethod
    def from_harvest(cls, harvest: Any) -> "CaptureStatistics":
        n_tokens = int(harvest.n_tokens)
        n_embd = int(harvest.n_embd)
        return cls(
            layer=int(harvest.layer),
            n_tokens=n_tokens,
            n_embd=n_embd,
            elements=n_tokens * n_embd,
            n_bytes=n_tokens * n_embd * 4,
        )

    def to_json_object(self) -> JsonObject:
        return {
            "layer": self.layer,
            "n_tokens": self.n_tokens,
            "n_embd": self.n_embd,
            "elements": self.elements,
            "n_bytes": self.n_bytes,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class ContractEvidenceBinding:
    """Receipt-portable identity and replay labels for the operations actually used."""

    SCHEMA: ClassVar[str] = "clozn.intervention_contract_binding.v1"
    contract_sha256: str
    operations: tuple[InterventionOperation, ...]
    replay_classes: tuple[ReplayClass, ...]
    capture: CaptureStatistics | None = None

    def to_json_object(self) -> JsonObject:
        value: JsonObject = {
            "schema": self.SCHEMA,
            "contract_sha256": self.contract_sha256,
            "operations": [item.value for item in self.operations],
            "replay_classes": [item.value for item in self.replay_classes],
        }
        if self.capture is not None:
            value["capture"] = self.capture.to_json_object()
        return value


def bind_contract_evidence(value: Any, *, capture: Any | None = None) -> ContractEvidenceBinding:
    """Bind a supported manifest to the exact stable contract operations it consumes."""
    operations = required_operations(value)
    replay = tuple(dict.fromkeys(MINIMAL_INTERVENTION_CONTRACT.spec(item).replay_class for item in operations))
    statistics = None if capture is None else CaptureStatistics.from_harvest(capture)
    return ContractEvidenceBinding(
        contract_sha256=MINIMAL_INTERVENTION_CONTRACT.sha256,
        operations=operations,
        replay_classes=replay,
        capture=statistics,
    )


@dataclass(frozen=True)
class ContractCheck:
    operation: InterventionOperation
    available: bool
    reason: str

    def to_json_object(self) -> JsonObject:
        return {"operation": self.operation.value, "available": self.available, "reason": self.reason}


@dataclass(frozen=True)
class ContractReport:
    """Engine-specific availability report for a selected set of stable operations."""

    contract_sha256: str
    checks: tuple[ContractCheck, ...]
    protocol_compatibility: ProtocolCompatibility
    protocol_reason: str
    engine_health: JsonObject = field(default_factory=dict)

    @property
    def compatible(self) -> bool:
        return self.protocol_compatibility is ProtocolCompatibility.VERIFIED and all(
            check.available for check in self.checks
        )

    def require_compatible(self) -> None:
        failures = [check for check in self.checks if not check.available]
        details = []
        if self.protocol_compatibility is not ProtocolCompatibility.VERIFIED:
            details.append(f"protocol: {self.protocol_reason}")
        details.extend(f"{check.operation.value}: {check.reason}" for check in failures)
        if details:
            raise CloznProtocolError("engine does not satisfy intervention contract: " + "; ".join(details))

    def to_json_object(self) -> JsonObject:
        return {
            "schema": "clozn.intervention_contract_report.v1",
            "contract_sha256": self.contract_sha256,
            "compatible": self.compatible,
            "protocol_compatibility": self.protocol_compatibility.value,
            "protocol_reason": self.protocol_reason,
            "checks": [check.to_json_object() for check in self.checks],
            "engine_health": dict(self.engine_health),
        }


def required_operations(value: Any) -> tuple[InterventionOperation, ...]:
    """Return the stable operations required by a supported client manifest."""
    from .manifests import InterventionManifest
    from .patch_manifests import PatchSweepManifest

    if isinstance(value, InterventionManifest):
        operations = {InterventionOperation.SCORE_TEACHER_FORCED}
        if any(arm.attention_knockout for arm in value.arms):
            operations.add(InterventionOperation.ATTENTION_KNOCKOUT)
        # Generic steering remains supported by the legacy manifest but deliberately has no stable
        # v1 hook-contract identifier.
        return tuple(sorted(operations, key=lambda item: item.value))
    if isinstance(value, PatchSweepManifest):
        return (InterventionOperation.CAPTURE_RESIDUAL, InterventionOperation.REPLACE_RESIDUAL)
    raise ValueError("value must be an InterventionManifest or PatchSweepManifest")


def check_contract(
    health: Mapping[str, Any],
    operations: Sequence[InterventionOperation | str],
    *,
    contract: InterventionContract = MINIMAL_INTERVENTION_CONTRACT,
) -> ContractReport:
    """Check capability-advertised requirements without inventing unsupported health fields."""
    if not isinstance(health, Mapping):
        raise ValueError("health must be an object")
    normalized: list[InterventionOperation] = []
    for operation in operations:
        try:
            normalized.append(
                operation if isinstance(operation, InterventionOperation) else InterventionOperation(operation)
            )
        except ValueError as exc:
            raise ValueError(f"operation is not in {contract.SCHEMA}: {operation!r}") from exc
    if not normalized:
        raise ValueError("operations must not be empty")
    capabilities = health.get("capabilities")
    capability_map = capabilities if isinstance(capabilities, Mapping) else {}
    checks: list[ContractCheck] = []
    for operation in dict.fromkeys(normalized):
        spec = contract.spec(operation)
        if spec.health_capability is None:
            checks.append(ContractCheck(operation, True, "operation is exposed by the selected native endpoint"))
        elif capability_map.get(spec.health_capability) is True:
            checks.append(ContractCheck(operation, True, f"capabilities.{spec.health_capability}=true"))
        else:
            checks.append(ContractCheck(operation, False, f"capabilities.{spec.health_capability}=true is required"))
    advertisement = EngineProtocolAdvertisement.from_health(health)
    if advertisement is None:
        protocol_compatibility = ProtocolCompatibility.UNADVERTISED
        protocol_reason = "health.protocol is required; endpoint presence alone is not compatibility proof"
    elif advertisement.protocol_version != "1.0":
        protocol_compatibility = ProtocolCompatibility.INCOMPATIBLE
        protocol_reason = f"protocol version {advertisement.protocol_version!r} is not supported (expected '1.0')"
    elif advertisement.intervention_contract_schema != contract.SCHEMA:
        protocol_compatibility = ProtocolCompatibility.INCOMPATIBLE
        protocol_reason = (
            f"contract schema {advertisement.intervention_contract_schema!r} does not match {contract.SCHEMA!r}"
        )
    elif advertisement.intervention_contract_sha256 != contract.sha256:
        protocol_compatibility = ProtocolCompatibility.INCOMPATIBLE
        protocol_reason = "advertised intervention contract digest does not match this client"
    else:
        protocol_compatibility = ProtocolCompatibility.VERIFIED
        protocol_reason = "engine explicitly advertises the client contract schema and digest"
    return ContractReport(
        contract.sha256, tuple(checks), protocol_compatibility, protocol_reason, dict(health)
    )
