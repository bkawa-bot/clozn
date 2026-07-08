"""test_facts_mode -- the on/off gate + per-profile store paths for the FACTS tier (MODEL-FREE).

No model, no GPU. facts_mode is stdlib-only (it wraps memory_mode's settings file + builds store paths),
so this is a pure logic test. The load-bearing invariants: the tier DEFAULTS OFF (the latency rule --
absent/garbage setting => disabled), set/get round-trips through the shared studio_settings.json, and
store_path is per-profile + slug-safe (an unknown/blank/unsafe name collapses to "default").
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn import facts_mode  # noqa: E402
from clozn import memory_mode  # noqa: E402


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    """Isolate the settings file + the profiles dir facts_mode reads/writes."""
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(facts_mode, "PROFILES_DIR", str(tmp_path / "profiles"))
    return tmp_path


# ---- the on/off gate (default OFF -- the latency rule) ----------------------------------------------

def test_defaults_off_on_a_fresh_install():
    assert facts_mode.enabled() is False


def test_set_enabled_round_trips():
    assert facts_mode.set_enabled(True)
    assert facts_mode.enabled() is True
    assert facts_mode.set_enabled(False)
    assert facts_mode.enabled() is False


def test_garbage_setting_reads_as_off(iso):
    # a corrupt / non-bool value must degrade to OFF, never silently on
    memory_mode.set_setting("memory_facts", "banana")
    assert facts_mode.enabled() is False


def test_string_truthy_values_read_as_on(iso):
    for v in ("on", "true", "1", "yes", "On", "TRUE"):
        memory_mode.set_setting("memory_facts", v)
        assert facts_mode.enabled() is True, v
    for v in ("off", "false", "0", "no", ""):
        memory_mode.set_setting("memory_facts", v)
        assert facts_mode.enabled() is False, v


def test_enabling_facts_leaves_other_settings_intact(iso):
    memory_mode.set_mode("prompt")
    memory_mode.set_setting("active_profile", "friend")
    facts_mode.set_enabled(True)
    assert memory_mode.get_mode() == "prompt"
    assert memory_mode.get_setting("active_profile") == "friend"
    assert facts_mode.enabled() is True


# ---- per-profile store paths ------------------------------------------------------------------------

def test_store_path_is_per_profile():
    a = facts_mode.store_path("friend")
    b = facts_mode.store_path("work")
    assert a != b
    assert a.endswith(os.path.join("profiles", "friend.slots.pt"))
    assert b.endswith(os.path.join("profiles", "work.slots.pt"))


def test_store_path_defaults_when_no_profile():
    assert facts_mode.store_path(None).endswith("default.slots.pt")
    assert facts_mode.store_path("").endswith("default.slots.pt")
    assert facts_mode.store_path("   ").endswith("default.slots.pt")


def test_store_path_rejects_unsafe_names_falling_back_to_default():
    # a name that isn't slug-safe (would be a bad filename) must collapse to default, not path-traverse
    for bad in ("../evil", "Not A Slug!", "a/b", "..", "x" * 50):
        assert facts_mode.store_path(bad).endswith("default.slots.pt"), bad


def test_store_path_honors_the_profiles_dir_global(iso):
    p = facts_mode.store_path("friend")
    assert p.startswith(str(iso / "profiles"))
