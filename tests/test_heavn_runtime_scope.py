"""Static wiring and honesty contracts for Scope's raw runtime state bench."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_api_exposes_raw_harvest_and_temporary_observe():
    client = _read(HEAVN / "api.mjs")
    assert 'engineHarvest: text => postE("/engine/harvest"' in client
    assert 'engineObserve: (text, position, scale) => postE("/engine/observe"' in client


def test_scope_mounts_the_runtime_bench_on_the_shared_text():
    scope = _read(HEAVN / "modules" / "scope.mjs")
    assert "<${RuntimeStateBench} text=${text} live=${live}/>" in scope
    assert 'data-testid="runtime-bench"' in scope
    assert 'data-testid="runtime-token-grid"' in scope
    assert 'data-testid="runtime-observation"' in scope


def test_runtime_bench_names_the_non_persistent_boundary_and_receipt():
    scope = _read(HEAVN / "modules" / "scope.mjs")
    for phrase in ("single comparison forward only", "no model weights", "TOP-1 FLIPPED",
                   "TOP-1 HELD", "not evidence that the model permanently learned"):
        assert phrase in scope
    for field in ("baseline_top", "edited_top", "moved_l2", "shifted"):
        assert f"observation.{field}" in scope


def test_runtime_bench_styles_cover_selection_distributions_and_mobile():
    theme = _read(HEAVN / "theme.css")
    for selector in (".runtime-token.selected", ".runtime-observe-controls",
                     ".runtime-distributions", ".runtime-error", ".runtime-receipt"):
        assert selector in theme
    assert "@media(max-width:720px)" in theme


def test_architecture_declares_one_canonical_steering_surface():
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    assert "/steer/*` is the canonical" in architecture
    assert "/engine/steer/*" in architecture and "deprecated compatibility" in architecture
