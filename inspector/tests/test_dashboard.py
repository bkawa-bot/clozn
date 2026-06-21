"""Fast oracle for the unified dashboard (Phase 2.4) — "one inspector, every substrate".

All model-free: it proves the SEAM, not any checkpoint —
  (a) DISPATCH: each substrate selector builds the correct StateSource type (engine-* construct
      WITHOUT a live server — the constructor does no I/O), and
  (b) ONE DRIVE LOOP end-to-end on the `toy` source: the Spine drives it and the dashboard renders
      a self-contained SVG/HTML page under runs/.

The engine HTTP path and the RWKV checkpoint are exercised elsewhere behind `-m model`; nothing
here touches the network or loads transformers, so `pytest -m "not model"` stays green.
"""
import numpy as np
import pytest

from clozn.dashboard import (
    Recorder,
    SUBSTRATES,
    build_source,
    dashboard,
    list_substrates,
    prepare_inputs,
)
from clozn.sources.engine import EngineStateSource
from clozn.sources.toy_recurrent import ToyRecurrentSource
from clozn.spine import Spine, StateStep


# --------------------------------------------------------------------------------------------------
# (a) DISPATCH — selector -> the right StateSource type, no network for engine-*
# --------------------------------------------------------------------------------------------------
def test_registry_lists_all_four_substrates():
    assert set(list_substrates()) == {"toy", "rwkv", "engine-ar", "engine-diffusion"}
    assert set(SUBSTRATES) == set(list_substrates())


def test_toy_selector_builds_toy_source():
    src = build_source("toy", "abc")
    assert isinstance(src, ToyRecurrentSource)


def test_engine_ar_selector_builds_engine_source_tagged_autoregressive():
    # no server required: EngineStateSource.__init__ is pure (no connection until step()).
    src = build_source("engine-ar", "hi", base_url="http://127.0.0.1:9999")
    assert isinstance(src, EngineStateSource)
    assert src.substrate == "autoregressive"
    assert src.base_url == "http://127.0.0.1:9999"


def test_engine_diffusion_selector_builds_engine_source_tagged_diffusion():
    src = build_source("engine-diffusion", "hi")
    assert isinstance(src, EngineStateSource)
    assert src.substrate == "diffusion"


def test_each_selector_dispatches_to_a_distinct_expected_type():
    expected = {
        "toy": ToyRecurrentSource,
        "engine-ar": EngineStateSource,
        "engine-diffusion": EngineStateSource,
    }
    for sel, typ in expected.items():
        assert isinstance(build_source(sel, "abc"), typ), sel


def test_unknown_selector_raises():
    with pytest.raises(ValueError):
        build_source("does-not-exist", "x")
    with pytest.raises(ValueError):
        dashboard("nope", "x")


# --------------------------------------------------------------------------------------------------
# prepare_inputs — source-agnostic prompt -> step inputs (duck-typed off the seam, no substrate switch)
# --------------------------------------------------------------------------------------------------
def test_prepare_inputs_toy_steps_per_symbol():
    src = build_source("toy", "abz!")
    assert prepare_inputs(src, "abz!") == ["a", "b", "z", "!"]


def test_prepare_inputs_engine_is_a_single_whole_prompt_step():
    src = build_source("engine-ar", "hi")
    # no tokenizer + no per-symbol vocab -> one step drives a whole engine run
    assert prepare_inputs(src, "hello world") == ["hello world"]


def test_prepare_inputs_respects_max_steps():
    src = build_source("toy", "abcdefgh")
    assert prepare_inputs(src, "abcdefgh", max_steps=3) == ["a", "b", "c"]


# --------------------------------------------------------------------------------------------------
# Recorder — a pure observer; copies the step so it can't corrupt the source (spine invariant 4)
# --------------------------------------------------------------------------------------------------
def test_recorder_captures_every_step_and_cannot_mutate_source():
    src = ToyRecurrentSource(list("abc"), d=8, seed=0)
    rec = Recorder()
    list(Spine(src, [rec]).run(list("abc")))
    assert len(rec.steps) == 3
    assert [s.token for s in rec.steps] == list("abc")
    assert all(isinstance(s, StateStep) for s in rec.steps)
    # scribbling on a captured step must not reach the live source
    rec.steps[-1].state["S"][:] = 999.0
    assert src.get_state()["S"].max() < 999.0


# --------------------------------------------------------------------------------------------------
# (b) ONE DRIVE LOOP end-to-end on the toy source — Spine drives, dashboard renders, model-free
# --------------------------------------------------------------------------------------------------
def test_dashboard_runs_end_to_end_on_toy_and_writes_html(tmp_path):
    out = tmp_path / "toy.html"
    path = dashboard("toy", "the quick brown fox", out=str(out))
    assert path == str(out)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # self-contained SVG/HTML (the house style), NOT matplotlib
    assert html.startswith("<!doctype html>")
    assert "<svg" in html
    assert "matplotlib" not in html.lower()
    # the Watch panel is present and substrate-tagged
    assert "WATCH" in html
    assert "toy" in html


def test_dashboard_watch_renders_one_panel_per_state_component(tmp_path):
    # the toy state has exactly one component ("S"); the substrate-agnostic Watch must label it.
    out = tmp_path / "toy2.html"
    dashboard("toy", "abcabc", out=str(out))
    html = out.read_text(encoding="utf-8")
    assert "state-write·intensity" in html or "state-write intensity" in html
    # the component name "S" appears as a row label in the heatmap SVG
    assert ">S<" in html


def test_dashboard_default_output_path_is_under_runs():
    # no `out=` -> runs/dashboard_<selector>.html (the convention inspect.py established)
    path = dashboard("toy", "abc")
    assert path.replace("\\", "/").endswith("runs/dashboard_toy.html")


def test_toy_dashboard_is_deterministic(tmp_path):
    a = tmp_path / "a.html"
    b = tmp_path / "b.html"
    dashboard("toy", "deterministic run", out=str(a))
    dashboard("toy", "deterministic run", out=str(b))
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# The ENGINE substrate drives the IDENTICAL loop — with a mocked SSE stream, still no server.
# Proves engine-* is wired through the same Spine/dashboard path the toy uses (the seam's payoff),
# without standing up the C++ engine (that lives behind -m model).
# --------------------------------------------------------------------------------------------------
def _engine_frames():
    """Mock SSE frames for one engine run: two committed-token frames + a terminal board frame."""
    from clozn.sources.engine import encode_tensor
    return [
        {"step": 0, "token": 7592, "state": {"hidden": encode_tensor(np.array([[0.5, -0.25, 1.0]], np.float32))},
         "readouts": [{"name": "logit-lens", "value": [["hello", 0.7]], "confidence": 0.7}],
         "meta": {"substrate": "autoregressive", "token": "hello", "top1": "hello"}},
        {"step": 1, "token": 2088, "state": {"hidden": encode_tensor(np.array([[0.1, 2.0, -1.5]], np.float32))},
         "readouts": [], "meta": {"substrate": "autoregressive", "token": "world", "top1": "world"}},
        {"step": 2, "token": None, "state": {"board": encode_tensor(np.array([101, 7592, 2088, 102], np.int64))},
         "readouts": [], "meta": {"substrate": "autoregressive", "kind": "end"}},
    ]


def _patch_engine_stream(monkeypatch):
    """Make EngineStateSource._stream_steps replay mock frames instead of hitting the network."""
    from clozn.sources.engine import EngineStateSource, parse_state_step
    frames = _engine_frames()

    def fake_stream(self, body):
        self._snapshot = {"board": np.array([101, 7592, 2088, 102], dtype=np.int64)}
        self._final = None
        for fr in frames:
            yield parse_state_step(fr)

    monkeypatch.setattr(EngineStateSource, "_stream_steps", fake_stream)


def test_engine_substrate_drives_the_same_dashboard_loop_without_a_server(tmp_path, monkeypatch):
    _patch_engine_stream(monkeypatch)
    out = tmp_path / "engine.html"
    path = dashboard("engine-ar", "Hello", out=str(out), base_url="http://127.0.0.1:9999")
    assert path == str(out)
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<svg" in html and "WATCH" in html
    assert "autoregressive" in html                 # the substrate label rode through
    assert "matplotlib" not in html.lower()


def test_engine_run_folds_to_one_aggregated_step_through_the_spine(tmp_path, monkeypatch):
    _patch_engine_stream(monkeypatch)
    src = build_source("engine-ar", "Hello")
    rec = Recorder()
    list(Spine(src, [rec]).run(prepare_inputs(src, "Hello")))
    # one engine run = one aggregated StateStep (the source's own contract), driven by the same Spine
    assert len(rec.steps) == 1
    assert rec.steps[0].meta["n_frames"] == 3
    assert "board" in rec.steps[0].state


# --------------------------------------------------------------------------------------------------
# The substrate-agnostic Watch renderer works on ANY state bag (the seam, isolated)
# --------------------------------------------------------------------------------------------------
def test_state_evolution_renders_for_an_arbitrary_state_bag():
    from clozn.viz import render_state_evolution
    # a hand-built stream with two oddly-named components of different shapes — no substrate involved
    steps = [
        StateStep(1, "x", {"foo": np.zeros((3, 3)), "bar": np.ones(4)}, meta={"token": "x"}),
        StateStep(2, "y", {"foo": np.ones((3, 3)), "bar": np.ones(4)}, meta={"token": "y", "top1": "z"}),
    ]
    html = render_state_evolution(steps)
    assert html.startswith("<!doctype html>")
    assert ">foo<" in html and ">bar<" in html      # both components became rows
    assert "matplotlib" not in html.lower()


# --------------------------------------------------------------------------------------------------
# GATED (-m model): the REAL RWKV substrate through the IDENTICAL dashboard entry. Proves the "all
# three substrates, one dashboard" claim on a real checkpoint when one is cached; skipped on CI.
# --------------------------------------------------------------------------------------------------
@pytest.mark.model
def test_dashboard_runs_end_to_end_on_real_rwkv(tmp_path):
    out = tmp_path / "rwkv.html"
    # the same dashboard() entry, just a different selector — that's the whole point.
    path = dashboard("rwkv", "The capital of France is Paris.", out=str(out), do_probe=False)
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<svg" in html and "WATCH" in html
    assert "att_num" in html                        # RWKV's recurrent components became Watch rows
