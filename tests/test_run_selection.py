"""Focused model-free tests for Phase 3.4 captured-run selection."""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.runs.association import project_key
from clozn.testkit.run_selection import RunSelectionError, resolve_runs


@pytest.fixture
def store(tmp_path):
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


def _record(store, *, source="openai_api", client="aider", project=None,
            parent=None, started=1000.0, response="answer"):
    return store.record(
        source=source,
        client=client,
        project_key=project_key(project) if project else None,
        parent_run_id=parent,
        messages=[{"role": "user", "content": "question"}],
        response=response,
        started=started,
    )


def test_explicit_ids_resolve_full_records_in_caller_order_and_deduplicate(store):
    first = _record(store, response="first")
    second = _record(store, response="second")

    selected = resolve_runs(run_ids=[second, first, second], journal=store)

    assert [run["id"] for run in selected] == [second, first]
    assert [run["response"] for run in selected] == ["second", "first"]
    assert all("messages" in run and "trace" in run for run in selected)


def test_explicit_selection_fails_atomically_on_missing_run(store):
    existing = _record(store)

    with pytest.raises(RunSelectionError) as exc:
        resolve_runs(run_ids=[existing, "run_missing"], journal=store)

    assert exc.value.code == "run_not_found"
    assert exc.value.run_id == "run_missing"


def test_explicit_derived_run_requires_opt_in(store):
    parent = _record(store)
    child = _record(store, source="replay", parent=parent)

    with pytest.raises(RunSelectionError) as exc:
        resolve_runs(run_ids=[child], journal=store)
    assert exc.value.code == "derived_run_not_allowed"

    assert resolve_runs(run_ids=[child], include_derived=True, journal=store)[0]["id"] == child


def test_latest_uses_journal_insertion_order_and_skips_newer_derived_run(store):
    first = _record(store, started=2000.0, response="first inserted")
    latest_organic = _record(store, started=1000.0, response="second inserted")
    _record(store, source="branch", parent=first, started=3000.0, response="derived")

    selected = resolve_runs(latest=True, journal=store)

    assert [run["id"] for run in selected] == [latest_organic]


def test_count_and_source_client_project_filters_compose(store):
    project = "workspace-one"
    wanted_old = _record(store, source="openai_api", client="Aider", project=project,
                         started=3000.0, response="old")
    _record(store, source="studio_chat", client="studio", project=project,
            started=4000.0, response="other")
    wanted_new = _record(store, source="openai_api", client="aider", project=project,
                         started=1000.0, response="new")
    _record(store, source="openai_api", client="aider", project="different",
            started=5000.0, response="wrong project")

    selected = resolve_runs(
        latest=True, count=2, source="OPENAI_API", client="AIDER", project=project, journal=store)

    assert [run["id"] for run in selected] == [wanted_new, wanted_old]


def test_filtered_selection_accepts_an_already_opaque_project_key(store):
    opaque = project_key("workspace-one")
    wanted = _record(store, project="workspace-one")

    assert resolve_runs(project=opaque, journal=store)[0]["id"] == wanted


def test_filtered_selection_fails_cleanly_when_nothing_matches(store):
    _record(store, project="workspace-one")

    with pytest.raises(RunSelectionError) as exc:
        resolve_runs(project="missing-project", journal=store)

    assert exc.value.code == "no_matching_runs"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"run_ids": "run_one"},
        {"count": 0},
        {"source": "   "},
        {"project": "contains space"},
        {"run_ids": ["run_one"], "latest": True},
    ],
)
def test_invalid_selector_shapes_fail_with_a_stable_code(store, kwargs):
    with pytest.raises(RunSelectionError) as exc:
        resolve_runs(journal=store, **kwargs)

    assert exc.value.code == "invalid_selector"


def test_derived_source_filter_requires_explicit_allow(store):
    parent = _record(store)
    child = _record(store, source="replay", parent=parent)

    with pytest.raises(RunSelectionError) as exc:
        resolve_runs(source="replay", journal=store)
    assert exc.value.code == "derived_run_not_allowed"

    assert resolve_runs(source="replay", include_derived=True, journal=store)[0]["id"] == child
