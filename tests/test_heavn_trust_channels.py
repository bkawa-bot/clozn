"""Static contracts for Replay's proxy, truth-calibrated, and support channels."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_api_keeps_nli_support_explicit_and_slow_timeout_separate():
    api = _read(HEAVN / "api.mjs")
    assert "trustSpans: (id, support = false)" in api
    assert "support ? { support: true } : {}" in api
    assert "support ? 300000 : 60000" in api


def test_replay_surfaces_three_honestly_named_channels():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert 'data-testid="trust-channels"' in replay
    for phrase in ("TRUTH-CAL", "labeled-probe estimate", "acceptance", "PROXY",
                   "NOT CHECKED", "CHECK SUPPORT", "Never external evidence"):
        assert phrase in replay


def test_truth_shading_requires_non_small_provenance_matched_result():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "sp.truth_correctness_estimate != null && trust.truth && !trust.truth.small_n" in replay
    assert "temperature-scaled correctness estimate" in replay
    assert "raw confidence does not equal correctness" in replay


def test_support_is_user_triggered_and_never_relabels_entailment_as_fact_check():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "await api.trustSpans(rec.id, true)" in replay
    assert "optional local NLI (~440 MB; may use an accelerator), no generation" in replay
    assert "Uses stored causal receipts only" in replay
    assert "active-influence manifest (presence, not causality)" in replay
    assert "the evidence tier will be disclosed after checking" in replay


def test_trust_channel_layout_collapses_on_narrow_screens():
    theme = _read(HEAVN / "theme.css")
    assert ".trust-channels{display:grid" in theme
    assert ".trust-channels{grid-template-columns:1fr}" in theme
