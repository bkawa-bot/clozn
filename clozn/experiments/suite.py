"""Versioned case x variant x seed experiments for model developers.

This is deliberately separate from :mod:`clozn.experiments.experiment`, which is the
single-run "change one thing" receipt API used by Studio.  A suite experiment composes
many ordinary, fully instrumented OpenAI-compatible calls and keeps their run records.
It never calls a model engine directly.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid

from clozn.testkit import ci as testkit_ci

MANIFEST_SCHEMA = "clozn.experiment.v0"
RESULT_SCHEMA = "clozn.experiment.result.v0"
VARIANT_KINDS = frozenset({"base", "tuned", "quant", "prompt", "dial"})
DEFAULT_URL = "http://127.0.0.1:8080"


class ManifestError(ValueError):
    pass


class ExperimentClientError(RuntimeError):
    pass


def _nonempty(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_messages(value, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{label} must be a non-empty list")
    for i, message in enumerate(value):
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant"):
            raise ManifestError(f"{label}[{i}] must be a text message with a supported role")
        if not isinstance(message.get("content"), str):
            raise ManifestError(f"{label}[{i}].content must be a string")


def validate_manifest(raw: dict) -> dict:
    """Validate and normalize a v0 manifest without mutating the caller's object."""
    if not isinstance(raw, dict):
        raise ManifestError("experiment manifest must be a JSON object")
    manifest = copy.deepcopy(raw)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ManifestError(f"schema_version must be {MANIFEST_SCHEMA!r}")
    manifest["name"] = _nonempty(manifest.get("name"), "name")
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ManifestError("defaults must be an object")
    manifest["defaults"] = defaults

    seeds = manifest.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ManifestError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ManifestError("seeds must not contain duplicates")
    manifest["seeds"] = seeds

    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise ManifestError("variants must contain at least two variants to compare")
    names = set()
    for i, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ManifestError(f"variants[{i}] must be an object")
        name = _nonempty(variant.get("name"), f"variants[{i}].name")
        if name in names:
            raise ManifestError(f"duplicate variant name: {name!r}")
        names.add(name)
        kind = variant.get("kind")
        if kind not in VARIANT_KINDS:
            raise ManifestError(f"variant {name!r} kind must be one of {sorted(VARIANT_KINDS)}")
        if kind == "dial" and "dials" not in variant:
            raise ManifestError(f"dial variant {name!r} requires a dials object")
        if "dials" in variant:
            dials = variant["dials"]
            if not isinstance(dials, dict):
                raise ManifestError(f"variant {name!r} dials must be an object")
            for dial, value in dials.items():
                if not isinstance(dial, str) or not dial or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ManifestError(f"variant {name!r} has an invalid dial value for {dial!r}")
        for field in ("base_url", "model", "system_prompt", "prompt_prefix", "prompt_suffix"):
            if field in variant and not isinstance(variant[field], str):
                raise ManifestError(f"variant {name!r} {field} must be a string")

    baseline = manifest.get("baseline_variant", variants[0]["name"])
    if baseline not in names:
        raise ManifestError("baseline_variant must name one of variants")
    manifest["baseline_variant"] = baseline

    suites = manifest.get("suites")
    if not isinstance(suites, dict):
        raise ManifestError("suites must be an object containing target and guard")
    for suite_name in ("target", "guard"):
        suite = suites.get(suite_name)
        if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list) or not suite["cases"]:
            raise ManifestError(f"suites.{suite_name}.cases must be a non-empty list")
        case_names = set()
        for i, case in enumerate(suite["cases"]):
            if not isinstance(case, dict):
                raise ManifestError(f"suites.{suite_name}.cases[{i}] must be an object")
            case_name = _nonempty(case.get("name"), f"suites.{suite_name}.cases[{i}].name")
            if case_name in case_names:
                raise ManifestError(f"duplicate {suite_name} case name: {case_name!r}")
            case_names.add(case_name)
            has_prompt, has_messages = isinstance(case.get("prompt"), str), "messages" in case
            if has_prompt == has_messages:
                raise ManifestError(f"case {case_name!r} must define exactly one of prompt or messages")
            if has_prompt and not case["prompt"].strip():
                raise ManifestError(f"case {case_name!r} prompt must not be empty")
            if has_messages:
                _validate_messages(case["messages"], f"case {case_name!r}.messages")
            if "expect" in case and not isinstance(case["expect"], dict):
                raise ManifestError(f"case {case_name!r} expect must be an object")
    return manifest


def load_manifest(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as exc:
        raise ManifestError(f"could not read experiment manifest {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"experiment manifest is not valid JSON: {exc}") from exc
    return validate_manifest(raw)


class ExperimentClient:
    """Small HTTP client. All generation goes through /v1/chat/completions."""
    def __init__(self, base_url: str, *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ExperimentClientError(f"HTTP {exc.code} {path}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ExperimentClientError(f"request failed for {self.base_url}{path}: {exc}") from exc

    def get_run(self, run_id: str) -> dict:
        run = self.request("GET", f"/runs/{run_id}")
        if not isinstance(run, dict) or not run:
            raise ExperimentClientError(f"run {run_id!r} was not available after generation")
        return run

    def receipts(self, run_id: str, mode: str = "regen") -> dict:
        return self.request("POST", f"/runs/{run_id}/receipts", {"mode": mode})

    def generate(self, messages: list, *, model: str, params: dict) -> tuple[dict, dict]:
        body = {"model": model, "messages": messages, "stream": False, **params}
        response = self.request("POST", "/v1/chat/completions", body)
        run_id = response.get("clozn_run_id") if isinstance(response, dict) else None
        if not run_id:
            rows = self.request("GET", "/runs")
            candidates = rows.get("runs") if isinstance(rows, dict) else None
            run_id = candidates[0].get("id") if candidates else None
        if not run_id:
            raise ExperimentClientError("completion returned no clozn_run_id and the run journal was empty")
        return response, self.get_run(run_id)

    @contextmanager
    def dial_state(self, desired: dict | None):
        """Temporarily apply an exact dial state and always restore the gateway's prior state."""
        if desired is None:
            yield
            return
        axes_response = self.request("POST", "/steer/axes", {})
        axes = axes_response.get("axes") if isinstance(axes_response, dict) else None
        if not isinstance(axes, list):
            raise ExperimentClientError("gateway did not expose /steer/axes for a dial variant")
        prior = {a.get("name"): float(a.get("value", 0.0)) for a in axes if isinstance(a, dict) and a.get("name")}
        unknown = sorted(set(desired) - set(prior))
        if unknown:
            raise ExperimentClientError(f"gateway does not provide requested dial(s): {', '.join(unknown)}")
        try:
            for name in prior:
                self.request("POST", "/steer/set", {"name": name, "value": float(desired.get(name, 0.0))})
            yield
        finally:
            for name, value in prior.items():
                try:
                    self.request("POST", "/steer/set", {"name": name, "value": value})
                except Exception:
                    pass


def _messages(case: dict, variant: dict) -> list:
    messages = copy.deepcopy(case.get("messages") or [{"role": "user", "content": case["prompt"]}])
    prefix, suffix = variant.get("prompt_prefix", ""), variant.get("prompt_suffix", "")
    if prefix or suffix:
        for message in reversed(messages):
            if message.get("role") == "user":
                message["content"] = prefix + message["content"] + suffix
                break
    if variant.get("system_prompt"):
        messages.insert(0, {"role": "system", "content": variant["system_prompt"]})
    return messages


class _RecordedClient:
    """Adapter that lets testkit.ci evaluate an already-generated run without regenerating it."""
    def __init__(self, run: dict, client):
        self.run, self.client = run, client

    def chat(self, _prompt):
        return self.run

    def receipts(self, run_id, mode="regen"):
        return self.client.receipts(run_id, mode)


def _cell_error(suite_name, case, variant, seed, exc) -> dict:
    return {"suite": suite_name, "case": case["name"], "variant": variant["name"],
            "variant_kind": variant["kind"], "seed": seed, "status": "error", "run_id": None,
            "response": None, "assertions": [], "min_confidence": None, "receipts": None,
            "error": f"{type(exc).__name__}: {exc}", "run": None}


def _summarize(cells: list, baseline: str, variant_names: list[str]) -> dict:
    aggregates = {}
    for variant in variant_names:
        aggregates[variant] = {}
        for suite in ("target", "guard"):
            selected = [c for c in cells if c["variant"] == variant and c["suite"] == suite]
            counts = {s: sum(c["status"] == s for c in selected) for s in ("pass", "fail", "error", "unscored")}
            scored = counts["pass"] + counts["fail"]
            aggregates[variant][suite] = {"runs": len(selected), "counts": counts,
                                                   "pass_rate": round(counts["pass"] / scored, 6) if scored else None}

    by_key = {(c["suite"], c["case"], c["seed"], c["variant"]): c for c in cells}
    comparisons = []
    for variant in variant_names:
        if variant == baseline:
            continue
        change = {"variant": variant, "baseline": baseline, "target_gains": [], "target_regressions": [],
                  "guard_regressions": [], "guard_fixes": [], "changed_unscored": []}
        for (suite, case, seed, name), base in by_key.items():
            if name != baseline:
                continue
            candidate = by_key.get((suite, case, seed, variant))
            if not candidate:
                continue
            label = {"case": case, "seed": seed}
            if base["status"] == "fail" and candidate["status"] == "pass":
                change["target_gains" if suite == "target" else "guard_fixes"].append(label)
            elif base["status"] == "pass" and candidate["status"] == "fail":
                change["target_regressions" if suite == "target" else "guard_regressions"].append(label)
            elif base["status"] == candidate["status"] == "unscored" and base["response"] != candidate["response"]:
                change["changed_unscored"].append({"suite": suite, **label})
        comparisons.append(change)
    return {"baseline_variant": baseline, "aggregates": aggregates, "comparisons": comparisons}


def run_manifest(raw_manifest: dict, *, default_url: str = DEFAULT_URL, seeds_override: int | None = None,
                 client_factory=ExperimentClient) -> dict:
    manifest = validate_manifest(raw_manifest)
    seeds = list(range(seeds_override)) if seeds_override is not None else manifest["seeds"]
    if not seeds:
        raise ManifestError("seeds override must be at least 1")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    experiment_id = "exp_" + uuid.uuid4().hex[:16]
    clients, cells = {}, []
    defaults = manifest["defaults"]

    for variant in manifest["variants"]:
        url = variant.get("base_url") or defaults.get("base_url") or default_url
        client = clients.setdefault(url, client_factory(url))
        try:
            dial_context = client.dial_state(variant.get("dials") if "dials" in variant else None)
            with dial_context:
                for suite_name in ("target", "guard"):
                    for case in manifest["suites"][suite_name]["cases"]:
                        for seed in seeds:
                            try:
                                params = {k: defaults[k] for k in ("max_tokens", "temperature", "top_p", "top_k", "repeat_penalty") if k in defaults}
                                params.update({k: variant[k] for k in ("max_tokens", "temperature", "top_p", "top_k", "repeat_penalty") if k in variant})
                                params["seed"] = seed
                                model = variant.get("model") or defaults.get("model") or "clozn"
                                messages = _messages(case, variant)
                                _response, run = client.generate(messages, model=model, params=params)
                                if case.get("expect") or case.get("prove"):
                                    eval_case = dict(case)
                                    eval_case["prompt"] = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "experiment case")
                                    eval_case.pop("messages", None)
                                    evaluated = testkit_ci.run_case(eval_case, _RecordedClient(run, client))
                                    status, assertions = evaluated.status, evaluated.assertions
                                    min_conf, receipts, error = evaluated.min_confidence, evaluated.receipts, evaluated.error or evaluated.prove_error
                                else:
                                    status, assertions, min_conf, receipts, error = "unscored", [], None, None, None
                                cells.append({"suite": suite_name, "case": case["name"], "variant": variant["name"],
                                              "variant_kind": variant["kind"], "seed": seed, "status": status,
                                              "run_id": run.get("id"), "response": run.get("response"),
                                              "assertions": assertions, "min_confidence": min_conf,
                                              "receipts": receipts, "error": error, "run": run})
                            except Exception as exc:
                                cells.append(_cell_error(suite_name, case, variant, seed, exc))
        except Exception as exc:
            for suite_name in ("target", "guard"):
                for case in manifest["suites"][suite_name]["cases"]:
                    for seed in seeds:
                        cells.append(_cell_error(suite_name, case, variant, seed, exc))

    return {"schema_version": RESULT_SCHEMA, "experiment_id": experiment_id, "name": manifest["name"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest_sha256": digest, "manifest": manifest, "seeds": seeds, "cells": cells,
            "summary": _summarize(cells, manifest["baseline_variant"], [v["name"] for v in manifest["variants"]])}


def default_result_path(result: dict, directory: str | None = None) -> str:
    directory = directory or os.path.expanduser("~/.clozn/experiments")
    return os.path.join(directory, result["experiment_id"] + ".json")


def format_summary(result: dict) -> str:
    lines = [f"clozn experiment {result.get('experiment_id')}  {result.get('name')}",
             f"  baseline: {result.get('summary', {}).get('baseline_variant')}  runs: {len(result.get('cells') or [])}"]
    for name, suites in (result.get("summary", {}).get("aggregates") or {}).items():
        parts = []
        for suite in ("target", "guard"):
            row = suites.get(suite, {})
            rate = "unscored" if row.get("pass_rate") is None else f"{100 * row['pass_rate']:.1f}% pass"
            parts.append(f"{suite} {rate}, {row.get('counts', {}).get('error', 0)} error")
        lines.append(f"  {name}: " + " | ".join(parts))
    for comparison in result.get("summary", {}).get("comparisons") or []:
        lines.append(f"  vs {comparison['baseline']} -> {comparison['variant']}: "
                     f"target +{len(comparison['target_gains'])}/-{len(comparison['target_regressions'])}; "
                     f"guard regressions {len(comparison['guard_regressions'])}")
    return "\n".join(lines)


def select_cells(result: dict, *, suite=None, case=None, variant=None, seed=None) -> list:
    return [c for c in (result.get("cells") or [])
            if (suite is None or c.get("suite") == suite)
            and (case is None or c.get("case") == case)
            and (variant is None or c.get("variant") == variant)
            and (seed is None or c.get("seed") == seed)]


def format_cells(cells: list) -> str:
    lines = []
    for c in cells:
        lines.append(f"[{c.get('status', '?').upper()}] {c.get('suite')}/{c.get('case')}  "
                     f"variant={c.get('variant')} seed={c.get('seed')} run={c.get('run_id') or '-'}")
        if c.get("response") is not None:
            lines.append("  " + str(c["response"]).replace("\n", "\n  "))
        for assertion in c.get("assertions") or []:
            lines.append(f"  - {assertion.get('status')}: {assertion.get('check')}")
        if c.get("error"):
            lines.append(f"  error: {c['error']}")
    return "\n".join(lines) if lines else "no matching experiment cells"
