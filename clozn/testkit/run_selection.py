"""Deterministic local-journal selection for captured-run suite promotion.

This module stops at resolution: it returns complete stored run records and does
not redact, edit, freeze, or serialize a suite.  Explicit ids are all-or-nothing;
filtered selection is newest-first by journal insertion order.  Derived runs are
rejected unless the caller opts in deliberately.
"""
from __future__ import annotations

from collections.abc import Sequence

import clozn.runs.store as _runlog


DERIVED_SOURCES = frozenset({"replay", "branch", "fork"})


class RunSelectionError(ValueError):
    """A selector is invalid or cannot be resolved without widening its intent."""

    def __init__(self, code: str, message: str, *, run_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.run_id = run_id


def _derived(run: dict) -> bool:
    return bool(run.get("parent_run_id") or str(run.get("source") or "") in DERIVED_SOURCES)


def _clean_optional(value, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise RunSelectionError("invalid_selector", f"{name} must not be empty")
    return text


def _clean_ids(run_ids) -> tuple[str, ...]:
    if run_ids is None:
        return ()
    if isinstance(run_ids, (str, bytes, bytearray)) or not isinstance(run_ids, Sequence):
        raise RunSelectionError("invalid_selector", "run_ids must be an array of run ids")
    out = []
    seen = set()
    for value in run_ids:
        rid = str(value).strip()
        if not rid:
            raise RunSelectionError("invalid_selector", "run_ids must not contain empty values")
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return tuple(out)


def _wanted_project(value: str | None) -> str | None:
    if value is None:
        return None
    from clozn.runs.association import AssociationValueError, project_key, validate_selector
    try:
        validate_selector(value, "project")
    except AssociationValueError as exc:
        raise RunSelectionError("invalid_selector", str(exc)) from exc
    key = project_key(value)
    if key is None:
        raise RunSelectionError("invalid_selector", "project must not be empty")
    return key


def _matches(run: dict, *, source: str | None, client: str | None,
             project_key: str | None) -> bool:
    if source is not None and str(run.get("source") or "").casefold() != source.casefold():
        return False
    if client is not None and str(run.get("client") or "").casefold() != client.casefold():
        return False
    if project_key is not None and run.get("project_key") != project_key:
        return False
    return True


def resolve_runs(*, run_ids=None, latest: bool = False, count: int | None = None,
                 source: str | None = None, client: str | None = None,
                 project: str | None = None, include_derived: bool = False,
                 journal=None) -> list[dict]:
    """Resolve one explicit-id set or one filtered newest-first journal selection.

    ``run_ids`` preserves caller order and removes duplicate ids. It cannot be
    combined with latest/count/filter selection. Explicit ids are resolved
    atomically: one missing or disallowed-derived record fails the whole request.

    Without explicit ids, ``latest`` selects newest-first and defaults to one record.
    ``count`` raises that limit to N; filters without either option also resolve the latest matching run.
    The default excludes replay/branch/fork children and any record with a parent.
    ``project`` accepts the caller-known project id or an already-opaque key.
    """
    store = journal or _runlog
    ids = _clean_ids(run_ids)
    source = _clean_optional(source, "source")
    client = _clean_optional(client, "client")
    project = _clean_optional(project, "project")
    filters_present = any(value is not None for value in (source, client, project))

    if ids:
        if latest or count is not None or filters_present:
            raise RunSelectionError(
                "invalid_selector", "explicit run ids cannot be combined with latest/count filters")
        selected = []
        for rid in ids:
            try:
                run = store.get_run(rid)
            except Exception as exc:
                raise RunSelectionError(
                    "journal_unavailable", f"could not read run {rid}: {type(exc).__name__}",
                    run_id=rid) from exc
            if not isinstance(run, dict):
                raise RunSelectionError("run_not_found", f"run not found: {rid}", run_id=rid)
            if _derived(run) and not include_derived:
                raise RunSelectionError(
                    "derived_run_not_allowed",
                    f"run {rid} is derived; pass include_derived=True to select it",
                    run_id=rid,
                )
            selected.append(run)
        return selected

    if not (latest or count is not None or filters_present):
        raise RunSelectionError(
            "invalid_selector", "select explicit run ids, latest, a count, or at least one filter")
    if not isinstance(latest, bool):
        raise RunSelectionError("invalid_selector", "latest must be a boolean")
    if count is None:
        wanted_count = 1
    elif isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RunSelectionError("invalid_selector", "count must be a positive integer")
    else:
        wanted_count = count
    if source is not None and source.casefold() in DERIVED_SOURCES and not include_derived:
        raise RunSelectionError(
            "derived_run_not_allowed",
            f"source {source!r} is derived; pass include_derived=True to select it",
        )

    wanted_project = _wanted_project(project)
    try:
        summaries = store.find_runs(
            limit=max(1, int(getattr(store, "KEEP", 1000))), include_derived=True)
    except Exception as exc:
        raise RunSelectionError(
            "journal_unavailable", f"could not scan the run journal: {type(exc).__name__}") from exc

    selected = []
    for summary in summaries:
        rid = summary.get("id") if isinstance(summary, dict) else None
        if not rid:
            continue
        try:
            run = store.get_run(rid)
        except Exception as exc:
            raise RunSelectionError(
                "journal_unavailable", f"could not read run {rid}: {type(exc).__name__}",
                run_id=str(rid)) from exc
        if not isinstance(run, dict):
            # The bounded journal may be pruned between scan and read. Skipping a vanished row keeps
            # resolution stable over the records that still exist without fabricating a partial object.
            continue
        if _derived(run) and not include_derived:
            continue
        if not _matches(run, source=source, client=client, project_key=wanted_project):
            continue
        selected.append(run)
        if len(selected) >= wanted_count:
            break
    if not selected:
        raise RunSelectionError("no_matching_runs", "no matching recorded runs found")
    return selected


__all__ = ["DERIVED_SOURCES", "RunSelectionError", "resolve_runs"]
