"""The `forks` panel on the M1 explain object (explain._forks) -- the close-call locator, surfaced into
the Run Inspector + shareable card. Pure over fixture run dicts."""
from __future__ import annotations

from clozn.receipts import explain


def _run(*steps):
    """steps = (emitted, p_emitted, [(alt, prob), ...]) -> a run dict with a realistic engine trace."""
    return {"id": "run_x", "trace": {
        "tokens": [e for (e, _p, _a) in steps],
        "confidence": [p for (_e, p, _a) in steps],
        "alternatives": [[{"piece": a, "prob": pa} for (a, pa) in alts] for (_e, _p, alts) in steps],
    }}


def test_explain_always_includes_a_forks_panel():
    x = explain.explain({"id": "r"})
    assert "forks" in x and x["forks"]["available"] is False        # no trace -> honest absence


def test_forks_panel_reports_a_near_tie_and_flags_meaningful():
    x = explain.explain(_run(("5", 0.54, [("0", 0.45)]),           # digit fork -> meaningful
                             ("imagine", 0.50, [("think", 0.48)]))) # phrasing fork -> not
    f = x["forks"]
    assert f["available"] is True
    assert len(f["forks"]) == 2 and f["meaningful_count"] == 1
    assert f["summary"].startswith("2 close calls")


def test_forks_panel_silent_on_a_confident_run():
    f = explain.explain(_run(("Paris", 0.98, [("Lyon", 0.01)])))["forks"]
    assert f["available"] is True and f["forks"] == [] and f["summary"] == ""


def test_explain_never_raises_on_junk():
    for junk in (None, "str", 42, {"trace": "nope"}):
        x = explain.explain(junk)
        assert x["forks"]["available"] is False
