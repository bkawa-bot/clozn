"""Static contracts for Replay's complaint-to-dial quick-repair surface."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_quick_repair_is_mounted_and_keeps_the_legacy_complaint_mapping():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "<${QuickRepair} rec=${rec}/>" in replay
    assert 'data-testid="quick-repair"' in replay
    for complaint, axis in (("verbose", "concise"), ("vague", "concrete"),
                            ("agreeable", "candid"), ("cold", "warm")):
        assert f'key: "{complaint}"' in replay
        assert f'axis: "{axis}"' in replay


def test_quick_repair_uses_matched_counterfactual_and_records_feedback_without_auto_save():
    client = _read(HEAVN / "api.mjs")
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert 'feedbackRecord: body => post("/feedback"' in client
    assert "api.feedbackRecord({" in replay
    assert "await api.counterfactual(rec.id" in replay
    assert "two matched greedy generations" in replay
    assert "from this run's recorded value" in replay
    assert "Nothing becomes a default unless you explicitly save it" in replay
    # Save persists what was actually TESTED (the server's read-back value), not the client's pre-clamp
    # guess -- see test_quick_repair_reads_back_the_real_post_clamp_value_not_the_guess below for the
    # full honesty contract this wiring exists to satisfy.
    assert "await api.steerSet(picked.axis, value)" in replay


def test_quick_repair_reads_back_the_real_post_clamp_value_not_the_guess():
    """Honesty graft: the panel must display the value the SERVER actually applied (read back from the
    child run), never the client's own pre-clamp arithmetic passed off as the outcome. A hardcoded
    per-axis cap table is exactly the failure mode this guards against -- it silently goes stale the day
    axes.py's own `max` values are re-tuned, while a client-displayed number that still claims to be
    correct is precisely the fabrication this test exists to catch."""
    counterfactual = _read(ROOT / "clozn" / "replay" / "counterfactual.py")
    replay = _read(HEAVN / "modules" / "replay.mjs")
    # server: counterfactual() surfaces the child run's real recorded dial state, not just an echo of
    # the request (`overrides_applied` already echoes the request; `applied_dials` must not).
    assert '"applied_dials"' in counterfactual
    assert 'cf_child["behavior"]["active_dials"]' in counterfactual or "active_dials" in counterfactual
    # client: reads that field back rather than trusting its own arithmetic.
    assert "res.applied_dials" in replay or "result.applied_dials" in replay
    assert "appliedValue" in replay
    # the displayed "repair candidate" number prefers the read-back; `picked.target` survives only as
    # the explicit "asked for" half of the display, never as the implied outcome.
    assert '(appliedNow != null ? appliedNow : picked.target).toFixed(2)' in replay
    assert "asked for ${picked.target.toFixed(2)}" in replay
    # no hand-maintained duplicate of the server's per-axis caps -- caps come from /steer/axes.
    assert "QUICK_AXIS_MAX = {" not in replay
    assert "api.steerAxes()" in replay


def test_quick_repair_requires_a_verified_coherent_change_before_save():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "result.causal_verified === true" in replay
    assert "result.has_effect === true" in replay
    assert "!coherence.degenerate" in replay
    assert "SAVE AS DEFAULT" in replay
    assert "stored sampled reply is context only, not a subtraction term" in replay


def test_quick_repair_styles_fit_the_narrow_replay_rail():
    theme = _read(HEAVN / "theme.css")
    for selector in (".quick-repair-body", ".quick-repair-presets", ".quick-repair-btn",
                     ".quick-repair-compare", ".quick-repair-actions", ".quick-repair-message.error"):
        assert selector in theme
    assert ".quick-repair-compare{display:grid;grid-template-columns:1fr;" in theme
