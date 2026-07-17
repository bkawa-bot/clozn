"""Static contracts for the model-free journal actuary panel."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_actuary_api_exposes_report_and_past_only_run_assessment():
    api = _read(HEAVN / "api.mjs")
    assert 'journalActuary: () => j("/journal/actuary"' in api
    assert 'runActuary: id => postE("/runs/" + enc(id) + "/actuary"' in api


def test_read_mounts_actuary_and_consumes_server_verdict_without_rescoring():
    read = _read(HEAVN / "modules" / "read.mjs")
    assert '<${ActuaryPanel} rec=${rec}/>' in read
    assert 'data-testid="actuary-panel"' in read
    assert "Promise.all([api.journalActuary(), api.runActuary(rec.id)])" in read
    assert 'verdict === "warning"' in read
    assert "RESEMBLES PAST FAILURES" in read
    assert "failure_score" not in read


def test_actuary_copy_preserves_proxy_small_n_and_no_safety_claims():
    read = _read(HEAVN / "modules" / "read.mjs")
    for phrase in ("accepted-proxy", "bad-proxy", "WEAK EVIDENCE · NO WARNING",
                   "This heuristic did not flag the trace. That is not evidence that the answer is correct.",
                   "not a fact-check", "no risk estimate was substituted", "PROXY, NOT CORRECTNESS"):
        assert phrase in read


def test_actuary_report_surfaces_calibration_drift_and_exact_drivers():
    read = _read(HEAVN / "modules" / "read.mjs")
    assert "a.drivers.map" in read
    assert "cal.bins.filter" in read
    assert "drift.slice(0,4)" in read
    assert "ECE proxy" in read
    assert "cal.note" in read and "a.note" in read


def test_actuary_styles_cover_warning_curve_drivers_and_drift():
    theme = _read(HEAVN / "theme.css")
    for selector in (".actuary-current.warning", ".actuary-current.weak", ".actuary-current.clear",
                     ".actuary-drivers", ".actuary-bins", ".actuary-bin", ".actuary-drift",
                     ".actuary-note"):
        assert selector in theme
