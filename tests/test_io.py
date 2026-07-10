"""Model-free tests for clozn._io.atomic_write_json -- the shared atomic-write helper.

Round-2 pressure test #1 (HIGH, data loss): every user-data JSON store used to `open(path, "w")` then
`json.dump(obj, f)` straight against the real path. If `obj` turned out to contain something json can't
serialize, `json.dump` raised AFTER the file had already been truncated to empty -- so a single bad write
silently destroyed everything that was there before (confirmed repro: 3 memory cards -> 1 bad update() ->
0 cards). This module is the shared fix every affected store now routes through: serialize to a string
FIRST (so a bad value raises before the real file is ever opened for writing), then write that string to
a temp file in the same directory and atomically os.replace() it over the real path.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
from clozn._io import atomic_write_json  # noqa: E402


def test_writes_json_readable_back(tmp_path):
    path = str(tmp_path / "x.json")
    atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"a": 1, "b": [1, 2, 3]}


def test_forwards_dump_kwargs_like_indent_and_ensure_ascii(tmp_path):
    path = str(tmp_path / "x.json")
    atomic_write_json(path, {"a": 1}, indent=2, ensure_ascii=False)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "\n" in text                       # indent=2 produced multi-line output
    assert json.loads(text) == {"a": 1}


def test_makes_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "x.json")
    atomic_write_json(path, {"a": 1})
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"a": 1}


def test_overwrites_an_existing_file_with_good_data(tmp_path):
    path = str(tmp_path / "x.json")
    atomic_write_json(path, {"v": 1})
    atomic_write_json(path, {"v": 2})
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"v": 2}


def test_non_serializable_value_raises_before_touching_an_existing_file(tmp_path):
    """The load-bearing guarantee: a bad write must never destroy prior good data."""
    path = str(tmp_path / "x.json")
    atomic_write_json(path, {"good": "data"})             # prior good content on disk
    with pytest.raises(TypeError):
        atomic_write_json(path, {"bad": {1, 2, 3}})       # a set isn't JSON-serializable
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"good": "data"}           # untouched -- the prior write survives intact


def test_non_serializable_value_never_creates_a_file_when_none_existed(tmp_path):
    path = str(tmp_path / "brand_new.json")
    with pytest.raises(TypeError):
        atomic_write_json(path, object())
    assert not os.path.exists(path)                        # no truncated/empty file left behind


def test_no_temp_file_left_behind_after_a_bad_write(tmp_path):
    path = str(tmp_path / "x.json")
    atomic_write_json(path, {"good": "data"})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"bad": object()})
    leftovers = [f for f in os.listdir(tmp_path) if f != "x.json"]
    assert leftovers == []                                 # the .tmp-atomic-*.json scratch file is cleaned up
