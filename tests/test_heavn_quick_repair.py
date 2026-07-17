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
    assert "await api.steerSet(picked.axis, picked.target)" in replay


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
