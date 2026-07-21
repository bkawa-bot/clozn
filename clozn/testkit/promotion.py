"""Pure promotion of journaled runs into editable regression-suite drafts.

The module owns no journal or suite storage.  Callers resolve selected runs, edit or
redact the returned JSON-compatible draft, then freeze it before execution.  A
freeze is an integrity contract rather than a signature: loaders recompute the
canonical SHA-256 before making any model call.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import re
from typing import Any


REGRESSION_SUITE_SCHEMA = "clozn.regression_suite.v1"
_TEXT_ROLES = frozenset({"system", "developer", "user", "assistant"})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CASE_KEYS = frozenset({
    "name", "prompt", "messages", "model", "max_tokens", "expect",
    "sampling", "prove", "prove_mode", "warnings", "source", "freeze",
})
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "seed"})
_UNSEEDED_WARNING = "captured sampling had no fixed seed; repeated results may vary"


class PromotionError(ValueError):
    """A run or suite cannot satisfy the reproducible promotion contract."""


def _canonical(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PromotionError(f"value is not canonical JSON: {exc}") from None
    return encoded.encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    # Canonicalization validates the complete tree; deepcopy then retains tuples only
    # where a malformed caller supplied them, so round-trip through JSON for one exact
    # artifact representation.
    raw = _canonical(value)
    return json.loads(raw.decode("utf-8"))


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionError(f"{label} must be a non-empty string")
    return value


def _text_messages(value: Any, label: str = "messages") -> list[dict[str, str]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or not value:
        raise PromotionError(f"{label} must be a non-empty array of text messages")
    messages = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise PromotionError(f"{label}[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in _TEXT_ROLES or not isinstance(content, str):
            raise PromotionError(
                f"{label}[{index}] must have a supported text role and string content")
        # Tool calls and other message extensions change request semantics.  Refuse
        # instead of silently producing a superficially similar text-only case.
        if set(message) != {"role", "content"}:
            raise PromotionError(f"{label}[{index}] contains unsupported message fields")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        raise PromotionError(f"{label} must end with a user message")
    return messages


def _max_tokens(run: Mapping[str, Any]) -> int:
    meta = run.get("meta")
    value = meta.get("max_tokens") if isinstance(meta, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PromotionError("run.meta.max_tokens must be a positive integer")
    return value


def _sampling(run: Mapping[str, Any]) -> tuple[dict[str, int | float], list[str]]:
    meta = run.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    sampling: dict[str, int | float] = {}
    for key in ("temperature", "top_p"):
        if key not in meta:
            continue
        value = meta[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            raise PromotionError(f"run.meta.{key} must be a finite number when recorded")
        value = float(value)
        if key == "temperature" and value < 0:
            raise PromotionError("run.meta.temperature must be non-negative")
        if key == "top_p" and not 0 < value <= 1:
            raise PromotionError("run.meta.top_p must be in (0, 1]")
        sampling[key] = value
    seed = meta.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise PromotionError("run.meta.seed must be an integer when recorded")
        if seed >= 0:
            sampling["seed"] = seed
    sampled = meta.get("sampler_mode") == "sample" or meta.get("sampling") is True
    warnings = [_UNSEEDED_WARNING] if sampled and "seed" not in sampling else []
    return sampling, warnings


def _source_material(run: Mapping[str, Any]) -> dict[str, Any]:
    run_id = _nonempty_string(run.get("id"), "run.id")
    identity = run.get("identity")
    if not isinstance(identity, Mapping) or not identity:
        raise PromotionError(f"run {run_id!r} has no immutable identity evidence")
    response = run.get("response")
    if not isinstance(response, str):
        raise PromotionError(f"run {run_id!r} response must be a string")
    if run.get("error"):
        raise PromotionError(f"run {run_id!r} ended with an error and cannot be promoted")
    sampling, warnings = _sampling(run)
    material = {
        "run_id": run_id,
        "identity": _json_copy(dict(identity)),
        "messages": _text_messages(run.get("messages"), f"run {run_id!r}.messages"),
        "model": _nonempty_string(run.get("model"), f"run {run_id!r}.model"),
        "max_tokens": _max_tokens(run),
        "response": response,
    }
    if sampling:
        material["sampling"] = sampling
    if warnings:
        material["warnings"] = warnings
    return material


def _case_from_run(run: Mapping[str, Any]) -> dict[str, Any]:
    material = _source_material(run)
    case = {
        "name": material["run_id"],
        "messages": copy.deepcopy(material["messages"]),
        "model": material["model"],
        "max_tokens": material["max_tokens"],
        "expect": {"equals": material["response"]},
        # The identity and captured traffic remain in the local journal.  The artifact
        # carries only a join key and a digest that verify_source() can later compare.
        "source": {"run_id": material["run_id"], "sha256": _digest(material)},
    }
    if material.get("sampling"):
        case["sampling"] = copy.deepcopy(material["sampling"])
    if material.get("warnings"):
        case["warnings"] = list(material["warnings"])
    return case


def create_suite_draft(name: str, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert selected full run records into one deterministic, storage-free draft."""
    name = _nonempty_string(name, "suite name")
    if isinstance(runs, (str, bytes, bytearray)) or not isinstance(runs, Sequence) or not runs:
        raise PromotionError("runs must be a non-empty ordered array of run records")
    cases = []
    seen = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise PromotionError(f"runs[{index}] must be an object")
        case = _case_from_run(run)
        if case["name"] in seen:
            raise PromotionError(f"duplicate selected run id: {case['name']!r}")
        seen.add(case["name"])
        cases.append(case)
    draft = {"schema_version": REGRESSION_SUITE_SCHEMA, "name": name,
             "state": "draft", "cases": cases}
    return validate_suite(draft, require_frozen=False)


def _validate_source(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"run_id", "sha256"}:
        raise PromotionError(f"{label}.source must contain exactly run_id and sha256")
    run_id = _nonempty_string(value.get("run_id"), f"{label}.source.run_id")
    sha = value.get("sha256")
    if not isinstance(sha, str) or not _DIGEST_RE.fullmatch(sha):
        raise PromotionError(f"{label}.source.sha256 must be a lowercase SHA-256 digest")
    return {"run_id": run_id, "sha256": sha}


def _without_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "freeze"}


def _validate_freeze(value: Any, actual: str, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "sha256"}:
        raise PromotionError(f"{label}.freeze must contain exactly algorithm and sha256")
    if value.get("algorithm") != "sha256":
        raise PromotionError(f"{label}.freeze.algorithm must be 'sha256'")
    claimed = value.get("sha256")
    if not isinstance(claimed, str) or not _DIGEST_RE.fullmatch(claimed):
        raise PromotionError(f"{label}.freeze.sha256 must be a lowercase SHA-256 digest")
    if claimed != actual:
        raise PromotionError(f"{label} changed after it was frozen")
    return {"algorithm": "sha256", "sha256": claimed}


def _validate_case(value: Any, *, require_frozen: bool, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} must be an object")
    extra = set(value) - _CASE_KEYS
    if extra:
        raise PromotionError(f"{label} has unsupported fields: {sorted(extra)!r}")
    case = _json_copy(dict(value))
    case["name"] = _nonempty_string(case.get("name"), f"{label}.name")
    has_prompt, has_messages = "prompt" in case, "messages" in case
    if has_prompt == has_messages:
        raise PromotionError(f"{label} must contain exactly one of prompt or messages")
    if has_prompt:
        case["prompt"] = _nonempty_string(case["prompt"], f"{label}.prompt")
    else:
        case["messages"] = _text_messages(case["messages"], f"{label}.messages")
    case["model"] = _nonempty_string(case.get("model"), f"{label}.model")
    maximum = case.get("max_tokens")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise PromotionError(f"{label}.max_tokens must be a positive integer")
    if "sampling" in case:
        sampling = case["sampling"]
        if not isinstance(sampling, dict) or not sampling or set(sampling) - _SAMPLING_KEYS:
            raise PromotionError(
                f"{label}.sampling must contain only temperature, top_p, and/or seed")
        for key in ("temperature", "top_p"):
            if key not in sampling:
                continue
            value = sampling[key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                raise PromotionError(f"{label}.sampling.{key} must be a finite number")
            value = float(value)
            if key == "temperature" and value < 0:
                raise PromotionError(f"{label}.sampling.temperature must be non-negative")
            if key == "top_p" and not 0 < value <= 1:
                raise PromotionError(f"{label}.sampling.top_p must be in (0, 1]")
            sampling[key] = value
        if "seed" in sampling and (
                not isinstance(sampling["seed"], int) or isinstance(sampling["seed"], bool)
                or sampling["seed"] < 0):
            raise PromotionError(f"{label}.sampling.seed must be a non-negative integer")
    expect = case.get("expect")
    if not isinstance(expect, dict):
        raise PromotionError(f"{label}.expect must be an object")
    prove = case.get("prove", False)
    if not isinstance(prove, bool):
        raise PromotionError(f"{label}.prove must be a boolean")
    if not expect and not prove:
        raise PromotionError(f"{label} must define expect checks or prove=true")
    if "prove_mode" in case:
        if not prove or case["prove_mode"] not in ("regen", "forced", "both"):
            raise PromotionError(f"{label}.prove_mode requires prove=true and regen, forced, or both")
    if "warnings" in case:
        warnings = case["warnings"]
        if (not isinstance(warnings, list) or not warnings
                or any(not isinstance(item, str) or not item for item in warnings)):
            raise PromotionError(f"{label}.warnings must be a non-empty array of strings")
    case["source"] = _validate_source(case.get("source"), label)

    if "freeze" in case:
        actual = _digest(_without_freeze(case))
        case["freeze"] = _validate_freeze(case["freeze"], actual, label)
    elif require_frozen:
        raise PromotionError(f"{label} is not frozen")
    return case


def edit_case(case: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    """Return an edited draft case; source provenance is intentionally immutable."""
    if not isinstance(case, Mapping):
        raise PromotionError("case must be an object")
    if "freeze" in case:
        raise PromotionError("cannot edit a frozen case")
    editable = {"name", "prompt", "messages", "model", "max_tokens", "sampling",
                "expect", "prove", "prove_mode", "warnings"}
    unknown = set(changes) - editable
    if unknown:
        raise PromotionError(f"unsupported case edits: {sorted(unknown)!r}")
    candidate = _json_copy(dict(case))
    if "prompt" in changes:
        candidate.pop("messages", None)
    if "messages" in changes:
        candidate.pop("prompt", None)
    for key, value in changes.items():
        if value is None and key in ("prove", "prove_mode"):
            candidate.pop(key, None)
        else:
            candidate[key] = value
    return _validate_case(candidate, require_frozen=False, label="case")


def _replacement_pattern(replacements: Mapping[str, str]) -> re.Pattern[str]:
    if not isinstance(replacements, Mapping) or not replacements:
        raise PromotionError("replacements must be a non-empty string-to-string object")
    for source, replacement in replacements.items():
        if not isinstance(source, str) or not source or not isinstance(replacement, str):
            raise PromotionError("redaction replacements must map non-empty strings to strings")
    # Longest-first makes overlap deterministic.  One regex pass prevents replacement
    # text from being scanned again (and accidentally cascading into another rule).
    ordered = sorted(replacements, key=lambda item: (-len(item), item))
    return re.compile("|".join(re.escape(item) for item in ordered))


def _redact_value(value: Any, pattern: re.Pattern[str], replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return pattern.sub(lambda match: replacements[match.group(0)], value)
    if isinstance(value, list):
        return [_redact_value(item, pattern, replacements) for item in value]
    if isinstance(value, dict):
        return {key: (item if key == "role" else _redact_value(item, pattern, replacements))
                for key, item in value.items()}
    return value


def redact_case(case: Mapping[str, Any], replacements: Mapping[str, str]) -> dict[str, Any]:
    """Redact literal text in runnable inputs/expectations without retaining the secrets."""
    pattern = _replacement_pattern(replacements)
    candidate = _json_copy(dict(case))
    if "freeze" in candidate:
        raise PromotionError("cannot redact a frozen case")
    changes = {
        key: _redact_value(candidate[key], pattern, replacements)
        for key in ("prompt", "messages", "model", "expect") if key in candidate
    }
    return edit_case(candidate, **changes)


def redact_suite(draft: Mapping[str, Any], replacements: Mapping[str, str]) -> dict[str, Any]:
    """Apply one deterministic literal-redaction map to every case in a draft suite."""
    if not isinstance(draft, Mapping):
        raise PromotionError("suite must be an object")
    if draft.get("state") == "frozen" or "freeze" in draft:
        raise PromotionError("cannot redact a frozen suite")
    suite = validate_suite(draft, require_frozen=False)
    suite["cases"] = [redact_case(case, replacements) for case in suite["cases"]]
    return validate_suite(suite, require_frozen=False)


def _freeze_case(case: Mapping[str, Any], label: str) -> dict[str, Any]:
    if "freeze" in case:
        raise PromotionError(f"{label} is already frozen")
    frozen = _validate_case(case, require_frozen=False, label=label)
    frozen["freeze"] = {"algorithm": "sha256", "sha256": _digest(frozen)}
    return frozen


def freeze_suite(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every executable case and the ordered suite envelope exactly once."""
    if not isinstance(draft, Mapping):
        raise PromotionError("suite must be an object")
    if draft.get("state") == "frozen" or "freeze" in draft:
        raise PromotionError("suite is already frozen")
    suite = validate_suite(draft, require_frozen=False)
    suite["cases"] = [_freeze_case(case, f"cases[{index}]")
                      for index, case in enumerate(suite["cases"])]
    suite["state"] = "frozen"
    suite["freeze"] = {"algorithm": "sha256", "sha256": _digest(suite)}
    return validate_suite(suite, require_frozen=True)


def validate_suite(value: Any, *, require_frozen: bool = True) -> dict[str, Any]:
    """Validate and copy a draft or frozen runnable suite, including freeze hashes."""
    if not isinstance(value, Mapping):
        raise PromotionError("suite must be an object")
    suite = _json_copy(dict(value))
    state = suite.get("state")
    allowed = {"schema_version", "name", "state", "cases"}
    if state == "frozen":
        allowed.add("freeze")
    if set(suite) != allowed:
        raise PromotionError(f"suite fields are invalid: expected {sorted(allowed)!r}")
    if suite.get("schema_version") != REGRESSION_SUITE_SCHEMA:
        raise PromotionError(f"schema_version must be {REGRESSION_SUITE_SCHEMA!r}")
    suite["name"] = _nonempty_string(suite.get("name"), "suite.name")
    if state not in ("draft", "frozen"):
        raise PromotionError("suite.state must be 'draft' or 'frozen'")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise PromotionError("suite.cases must be a non-empty array")
    suite["cases"] = [
        _validate_case(case, require_frozen=state == "frozen", label=f"cases[{index}]")
        for index, case in enumerate(cases)
    ]
    names = [case["name"] for case in suite["cases"]]
    if len(names) != len(set(names)):
        raise PromotionError("suite case names must be unique")
    if state == "frozen":
        actual = _digest(_without_freeze(suite))
        suite["freeze"] = _validate_freeze(suite.get("freeze"), actual, "suite")
    elif require_frozen:
        raise PromotionError("suite is not frozen")
    return suite


def verify_source(case: Mapping[str, Any], run: Mapping[str, Any]) -> bool:
    """Check that a current local run still matches a case's captured source digest."""
    try:
        validated = _validate_case(case, require_frozen=False, label="case")
        material = _source_material(run)
        return (validated["source"]["run_id"] == material["run_id"]
                and validated["source"]["sha256"] == _digest(material))
    except PromotionError:
        return False


__all__ = [
    "REGRESSION_SUITE_SCHEMA", "PromotionError", "create_suite_draft", "edit_case",
    "freeze_suite", "redact_case", "redact_suite", "validate_suite", "verify_source",
]
