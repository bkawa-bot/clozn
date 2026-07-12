"""test_model_diff.py -- model-free unit tests for analysis/model_diff.py and the (unregistered)
server/routes/diff.py route module. Synthetic run records only: no engine, no network, no live server,
no real ~/.clozn (the suite-diff tests write ci.save_result files into pytest's tmp_path and point the
route's `dir` override there)."""
from __future__ import annotations

import os
from urllib.parse import urlencode

import pytest

from clozn.analysis.model_diff import SURFACE_SIMILARITY_LABEL, diff_runs
from clozn.server.routes import diff as diff_routes
from clozn.testkit import ci


# ============================================================================================ fixtures
def _run(rid, prompt, tokens, *, confidence=None, alternatives=None, token_ids=None,
         model="qwen2.5-7b", quant=None, model_file=None, response=None):
    """A minimal but store-shaped run record (clozn.runs.store.record's fields; trace per runs/trace.py:
    parallel tokens/confidence/alternatives arrays). tokens=None -> a traceless (light-tier) run."""
    meta = {}
    if quant:
        meta["quant"] = quant
    if model_file:
        meta["model_file"] = model_file
    rec = {
        "id": rid,
        "model": model,
        "substrate": "engine",
        "messages": [{"role": "user", "content": prompt}],
        "response": response if response is not None else "".join(tokens or []),
        "meta": meta,
        "trace": {},
    }
    if tokens is not None:
        rec["trace"] = {"tokens": list(tokens)}
        if confidence is not None:
            rec["trace"]["confidence"] = list(confidence)
        if alternatives is not None:
            rec["trace"]["alternatives"] = list(alternatives)
        if token_ids is not None:
            rec["trace"]["token_ids"] = list(token_ids)
    return rec


_PARIS = ["The", " capital", " is", " Paris", "."]
_CONF_A = [0.99, 0.95, 0.9, 0.8, 0.97]


class FakeHandler:
    """Duck-typed stand-in for app.py's handler: routes only ever touch `.path` and `._json(...)`."""

    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, code, obj, extra_headers=None):
        self.status = code
        self.body = obj


# =========================================================================== (a) identical -> no divergence
def test_identical_runs_have_no_first_divergence():
    a = _run("run_a", "capital?", _PARIS, confidence=_CONF_A, quant="Q8_0")
    b = _run("run_b", "capital?", _PARIS, confidence=_CONF_A, quant="Q4_K_M")
    out = diff_runs(a, b)
    assert out["ok"] is True
    assert out["prompts_match"] is True
    assert "warn" not in out
    assert out["trace_available"] is True
    assert out["first_divergence"] is None
    assert out["common_prefix_len"] == len(_PARIS)
    assert out["summary"]["identical"] is True
    assert out["summary"]["char_similarity"] == pytest.approx(1.0)
    assert all(p["same"] for p in out["positions"])
    # the never-diverged almost-said signal is explicitly not-checked, not a fake "no"
    assert out["summary"]["b_was_alternative_in_a"]["checked"] is False
    assert out["summary"]["b_was_alternative_in_a"]["found"] is None


def test_models_and_quant_tags_are_echoed():
    a = _run("run_a", "q", _PARIS, quant="Q8_0", model_file="m-Q8_0.gguf")
    b = _run("run_b", "q", _PARIS, quant="Q4_K_M", model="other-model")
    out = diff_runs(a, b)
    assert out["a"] == {"run_id": "run_a", "model": "qwen2.5-7b", "quant": "Q8_0",
                        "model_file": "m-Q8_0.gguf", "substrate": "engine"}
    assert out["b"]["quant"] == "Q4_K_M"
    assert out["b"]["model"] == "other-model"


# ======================================================================== (b) divergence at the right index
def test_divergence_detected_at_right_index_with_both_confidences():
    b_toks = ["The", " capital", " is", " Lyon", "."]
    a = _run("run_a", "capital?", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "capital?", b_toks, confidence=[0.99, 0.95, 0.9, 0.4, 0.9])
    out = diff_runs(a, b)
    assert out["common_prefix_len"] == 3
    fd = out["first_divergence"]
    assert fd["index"] == 3
    assert fd["kind"] == "token_mismatch"
    assert fd["a_piece"] == " Paris"
    assert fd["b_piece"] == " Lyon"
    assert fd["a_conf"] == pytest.approx(0.8)
    assert fd["b_conf"] == pytest.approx(0.4)
    assert out["summary"]["identical"] is False
    assert [p["same"] for p in out["positions"]] == [True, True, True, False, True]
    assert out["positions"][3] == {"i": 3, "a_piece": " Paris", "b_piece": " Lyon", "same": False,
                                   "a_conf": pytest.approx(0.8), "b_conf": pytest.approx(0.4)}


def test_completely_different_runs_diverge_at_zero():
    a = _run("run_a", "q", ["Yes", "."], confidence=[0.9, 0.9])
    b = _run("run_b", "q", ["No", " way", "."], confidence=[0.8, 0.7, 0.9])
    out = diff_runs(a, b)
    assert out["common_prefix_len"] == 0
    assert out["first_divergence"]["index"] == 0
    assert out["first_divergence"]["kind"] == "token_mismatch"


def test_strict_prefix_is_a_length_divergence():
    a = _run("run_a", "q", _PARIS[:3], confidence=_CONF_A[:3])
    b = _run("run_b", "q", _PARIS, confidence=_CONF_A)
    out = diff_runs(a, b)
    assert out["common_prefix_len"] == 3
    fd = out["first_divergence"]
    assert fd == {"index": 3, "kind": "length_mismatch", "a_piece": None, "b_piece": " Paris",
                  "a_conf": None, "b_conf": pytest.approx(0.8)}
    assert out["summary"]["a_reply_tokens"] == 3
    assert out["summary"]["b_reply_tokens"] == 5
    # a committed nothing at the split -> the almost-said lookup is inverted/meaningless: not-checked
    assert out["summary"]["b_was_alternative_in_a"]["checked"] is False


def test_positions_list_caps_at_200_but_divergence_math_runs_on_full_traces():
    long_a = ["tok"] * 250
    long_b = ["tok"] * 249 + ["OTHER"]
    out = diff_runs(_run("run_a", "q", long_a), _run("run_b", "q", long_b))
    assert len(out["positions"]) == 200
    assert out["positions_truncated"] is True
    assert out["common_prefix_len"] == 249          # computed beyond the display cap
    assert out["first_divergence"]["index"] == 249


# ======================================================================= (c) the almost-said (rank) signal
def test_b_token_found_in_a_alternatives_with_rank_and_prob():
    alts_a = [[], [], [], [{"piece": " Rome", "prob": 0.22},
                           {"piece": " Lyon", "prob": 0.15, "token_id": 77}], []]
    a = _run("run_a", "capital?", _PARIS, confidence=_CONF_A, alternatives=alts_a)
    b = _run("run_b", "capital?", ["The", " capital", " is", " Lyon", "."])
    almost = diff_runs(a, b)["summary"]["b_was_alternative_in_a"]
    assert almost["checked"] is True
    assert almost["found"] is True
    assert almost["rank"] == 1
    assert almost["prob"] == pytest.approx(0.15)
    assert almost["matched_by"] == "piece"


def test_b_token_matched_by_token_id_when_both_sides_carry_ids():
    alts_a = [[], [], [], [{"piece": " Lyon-render-variant", "prob": 0.3, "token_id": 77}], []]
    a = _run("run_a", "q", _PARIS, alternatives=alts_a)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."],
             token_ids=[1, 2, 3, 77, 5])
    almost = diff_runs(a, b)["summary"]["b_was_alternative_in_a"]
    assert almost["found"] is True
    assert almost["rank"] == 0
    assert almost["matched_by"] == "token_id"


def test_b_token_absent_from_recorded_alternatives_is_found_false():
    alts_a = [[], [], [], [{"piece": " Rome", "prob": 0.22}], []]
    a = _run("run_a", "q", _PARIS, alternatives=alts_a)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    almost = diff_runs(a, b)["summary"]["b_was_alternative_in_a"]
    assert almost["checked"] is True
    assert almost["found"] is False
    assert almost["rank"] is None


def test_no_recorded_alternatives_is_unknown_never_no():
    a = _run("run_a", "q", _PARIS)                   # no alternatives array at all
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    almost = diff_runs(a, b)["summary"]["b_was_alternative_in_a"]
    assert almost["checked"] is False
    assert almost["found"] is None                   # UNKNOWN -- not folded into "no"
    assert "UNKNOWN" in almost["note"]


# ============================================================================ (d) prompt guard + honesty
def test_different_prompts_warn_not_a_controlled_comparison():
    a = _run("run_a", "capital of France?", _PARIS)
    b = _run("run_b", "capital of Italy?", ["The", " capital", " is", " Rome", "."])
    out = diff_runs(a, b)
    assert out["prompts_match"] is False
    assert "NOT a controlled comparison" in out["warn"]
    assert out["ok"] is True                         # warned, never refused


def test_char_similarity_is_labeled_surface_only():
    a = _run("run_a", "q", _PARIS)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    out = diff_runs(a, b)
    assert out["summary"]["char_similarity_label"] == SURFACE_SIMILARITY_LABEL
    assert out["summary"]["char_similarity_label"] == "surface similarity — wording, not meaning"
    assert 0.0 <= out["summary"]["char_similarity"] <= 1.0
    assert "OBSERVATIONAL" in out["caveat"]


# =============================================================================== (e) degrade + error shapes
def test_traceless_run_degrades_to_text_only():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", None, response="The capital is Lyon.")
    out = diff_runs(a, b)
    assert out["ok"] is True
    assert out["trace_available"] is False
    assert out["trace_missing"] == ["b"]
    assert out["first_divergence"] is None
    assert out["positions"] == []
    assert out["common_prefix_len"] is None
    assert out["summary"]["a_reply_tokens"] == 5
    assert out["summary"]["b_reply_tokens"] is None
    assert out["summary"]["b_was_alternative_in_a"] is None
    assert out["summary"]["identical"] is False
    assert 0.0 < out["summary"]["char_similarity"] < 1.0    # the text diff still works
    assert out["summary"]["char_similarity_label"] == SURFACE_SIMILARITY_LABEL


def test_mean_confidences_in_summary():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", _PARIS)                   # tokens but no confidence array
    out = diff_runs(a, b)
    assert out["summary"]["a_mean_confidence"] == pytest.approx(sum(_CONF_A) / len(_CONF_A))
    assert out["summary"]["b_mean_confidence"] is None


def test_missing_run_is_a_clean_error_shape():
    real = _run("run_a", "q", _PARIS)
    out = diff_runs(None, real)
    assert out == {"mode": "model_diff", "ok": False, "missing": ["a"],
                   "error": "run a missing/unreadable -- nothing to diff"}
    both = diff_runs({}, "not-a-dict")
    assert both["ok"] is False
    assert both["missing"] == ["a", "b"]


def test_diff_runs_never_raises_on_garbage_traces():
    a = _run("run_a", "q", _PARIS)
    a["trace"]["confidence"] = "junk-not-a-list"
    a["trace"]["alternatives"] = {"also": "junk"}
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    b["messages"] = "junk"
    out = diff_runs(a, b)
    assert out["ok"] is True
    assert out["first_divergence"]["index"] == 3


# ========================================================================= (f) POST /diff/runs route module
def test_post_diff_runs_route_diffs_two_stored_runs(monkeypatch):
    import clozn.runs.store as store
    runs = {"run_a": _run("run_a", "q", _PARIS),
            "run_b": _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])}
    monkeypatch.setattr(store, "get_run", lambda rid: runs.get(rid))
    h = FakeHandler("/diff/runs")
    assert diff_routes.try_post(h, "/diff/runs", {"a": "run_a", "b": "run_b"}) is True
    assert h.status == 200
    assert h.body["ok"] is True
    assert h.body["first_divergence"]["index"] == 3


def test_post_diff_runs_route_404s_missing_run(monkeypatch):
    import clozn.runs.store as store
    runs = {"run_a": _run("run_a", "q", _PARIS)}
    monkeypatch.setattr(store, "get_run", lambda rid: runs.get(rid))
    h = FakeHandler("/diff/runs")
    assert diff_routes.try_post(h, "/diff/runs", {"a": "run_a", "b": "run_gone"}) is True
    assert h.status == 404
    assert h.body["missing"] == ["run_gone"]


def test_post_diff_runs_route_400s_missing_ids():
    h = FakeHandler("/diff/runs")
    assert diff_routes.try_post(h, "/diff/runs", {"a": "run_a"}) is True
    assert h.status == 400


def test_diff_route_module_claims_only_diff_paths():
    h = FakeHandler("/somewhere/else")
    assert diff_routes.try_get(h, "/somewhere/else") is False
    assert diff_routes.try_post(h, "/somewhere/else", {}) is False
    assert h.status is None                          # never wrote a response it didn't own


# ========================================================================== (g) GET /diff/suites wrapper
def _suite(timestamp, *, c1_status, c2_min_conf):
    return ci.SuiteResult(
        model_note="synthetic", timestamp=timestamp, status=c1_status,
        counts={c1_status: 1, "pass": 1},
        cases=[ci.CaseResult(name="c1", run_id="r1", status=c1_status),
               ci.CaseResult(name="c2", run_id="r2", status="pass", min_confidence=c2_min_conf)])


def test_get_diff_suites_wraps_ci_diff_suites_over_saved_results(tmp_path):
    prev = _suite("2026-07-11T00:00:00", c1_status="pass", c2_min_conf=0.9)
    curr = _suite("2026-07-12T00:00:00", c1_status="fail", c2_min_conf=0.5)
    path_a = ci.save_result(prev, directory=str(tmp_path))
    path_b = ci.save_result(curr, directory=str(tmp_path))
    name_a = os.path.basename(path_a)
    name_b = os.path.basename(path_b)[:-len(".json")]          # extension optional on the wire
    h = FakeHandler("/diff/suites?" + urlencode({"a": name_a, "b": name_b, "dir": str(tmp_path)}))
    assert diff_routes.try_get(h, "/diff/suites") is True
    assert h.status == 200
    body = h.body
    assert body["a"] == name_a and body["b"] == name_b
    assert body["prev_timestamp"] == "2026-07-11T00:00:00"
    assert body["curr_timestamp"] == "2026-07-12T00:00:00"
    # the exact ci.diff_suites verdicts, passed through untouched
    assert len(body["regressions"]) == 1
    assert body["regressions"][0]["case"] == "c1"
    assert body["regressions"][0]["kind"] == "status_regression"
    assert body["fixed"] == []
    assert len(body["drift"]) == 1
    assert body["drift"][0]["case"] == "c2"
    assert body["drift"][0]["delta"] == pytest.approx(-0.4)
    assert body["new_cases"] == [] and body["removed_cases"] == []
    assert "REGRESSIONS (1)" in body["text"]                    # Diff.render(), riding along verbatim


def test_get_diff_suites_404s_unknown_suite(tmp_path):
    path_a = ci.save_result(_suite("t", c1_status="pass", c2_min_conf=0.9), directory=str(tmp_path))
    h = FakeHandler("/diff/suites?" + urlencode({"a": os.path.basename(path_a), "b": "nope",
                                                 "dir": str(tmp_path)}))
    assert diff_routes.try_get(h, "/diff/suites") is True
    assert h.status == 404
    assert h.body["missing"] == ["nope"]


def test_get_diff_suites_400s_missing_params(tmp_path):
    h = FakeHandler("/diff/suites?" + urlencode({"a": "only-one", "dir": str(tmp_path)}))
    assert diff_routes.try_get(h, "/diff/suites") is True
    assert h.status == 400


def test_suite_name_resolution_never_escapes_the_directory(tmp_path):
    # a traversal-shaped name must not resolve to a file outside `dir` -- basename-only resolution
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    inner = tmp_path / "ci"
    inner.mkdir()
    assert diff_routes._resolve_suite(str(inner), "..\\outside.json") is None
    assert diff_routes._resolve_suite(str(inner), "../outside.json") is None
