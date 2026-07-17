"""Static contracts for Replay's zero-generation Explain surface."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_replay_mounts_explain_for_the_selected_run():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "<${ExplainPanel} rec=${rec}/>" in replay
    for testid in ("explain-panel", "explain-confidence", "explain-influences",
                   "explain-forks", "explain-concepts"):
        assert f'data-testid="{testid}"' in replay


def test_explain_uses_the_free_record_assembly_endpoint():
    client = _read(HEAVN / "api.mjs")
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert 'explain:  id => post("/runs/" + enc(id) + "/explain", {}, 30000)' in client
    assert "await api.explain(rec.id)" in replay
    assert "guardSample(rec)" in replay


def test_explain_copy_preserves_honesty_boundaries():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    for phrase in ("Measured per token - never an overall score",
                   "Present on this turn does not mean causally responsible",
                   "A correlational locator for a useful branch test, never a fragility verdict",
                   "not a verified chain of thought", "zero generation"):
        assert phrase in replay
    assert "ACTIVE · NOT PROVEN" in replay


def test_explain_styles_fit_the_narrow_replay_rail():
    theme = _read(HEAVN / "theme.css")
    for selector in (".explain-body", ".explain-grid", ".explain-card",
                     ".explain-row", ".explain-error", ".explain-foot"):
        assert selector in theme
    assert ".explain-grid{display:grid;grid-template-columns:1fr;" in theme
