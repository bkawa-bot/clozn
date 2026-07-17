"""Static contracts for Patch's model-free learned-preference review surface."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_preference_api_exposes_refresh_and_resolution():
    client = _read(HEAVN / "api.mjs")
    assert 'preferences: (threshold = 3) => postE("/preferences"' in client
    assert 'preferenceResolve: (id, action) => postE("/preferences/resolve"' in client


def test_patch_mounts_suggestions_and_refreshes_dials_after_approval():
    patch = _read(HEAVN / "modules" / "patch.mjs")
    assert "<${PreferenceSuggestions} onApplied=${loadAxes}/>" in patch
    assert 'data-testid="preference-suggestions"' in patch
    assert "await api.preferences(3)" in patch
    assert "await api.preferenceResolve(proposal.id, action)" in patch
    assert "await onApplied()" in patch


def test_preference_copy_preserves_review_and_evidence_boundaries():
    patch = _read(HEAVN / "modules" / "patch.mjs")
    for phrase in ("A model-free rollup of your quick-repair clicks",
                   "not an inference about", "no dial changes without APPROVE",
                   "no live steering object was available, so no dial changed",
                   "fresh threshold of evidence"):
        assert phrase in patch
    assert "p.evidence.join" in patch


def test_preference_styles_include_mobile_and_all_outcomes():
    theme = _read(HEAVN / "theme.css")
    for selector in (".preference-body", ".preference-row", ".preference-evidence",
                     ".preference-message.ok", ".preference-message.warn",
                     ".preference-message.error"):
        assert selector in theme
    assert ".preference-actions{grid-column:1;grid-row:auto}" in theme
