"""research/tests conftest -- the `-m model` gate.

research/tests had no pytest config, so a bare `@pytest.mark.model` test would still RUN in the plain
suite (`pytest research/tests/ -q`) -- and these tests load a real checkpoint onto the GPU and TTT-train
a soft prefix for minutes. This conftest makes the marker actually gate: model-marked tests are skipped
unless the run's mark expression names "model" (i.e. you opted in with `-m model`). Mirrors the
inspector's convention (its pyproject registers the same marker; its gated tests additionally
pytest.skip at runtime when the resource is missing -- ours do too).
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "model: loads a real checkpoint on the GPU and trains (slow); deselected unless run with -m model")


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", "") or ""
    if "model" in markexpr:            # explicit opt-in (`-m model`) or exclusion (`-m "not model"`):
        return                         # let pytest's own -m selection decide
    skip = pytest.mark.skip(reason="model-gated (loads a GPU checkpoint): run with -m model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
