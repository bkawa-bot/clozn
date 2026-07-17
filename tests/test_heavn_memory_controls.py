"""Static contracts for heavn Memory's carrier and strength controls."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_memory_api_exposes_mode_and_strength_reads_and_writes():
    client = _read(HEAVN / "api.mjs")
    assert 'memoryMode: () => j("/memory/mode"' in client
    assert 'memorySetMode: mode => postE("/memory/mode"' in client
    assert 'memoryStrength: value => postE("/memory/strength"' in client


def test_memory_desk_mounts_live_controls_and_uses_server_reported_modes():
    memory = _read(HEAVN / "modules" / "memory.mjs")
    assert "<${MemoryControls}" in memory
    assert 'data-testid="memory-controls"' in memory
    assert 'data-testid="memory-strength"' in memory
    assert "modes.map(" in memory
    assert "await api.memorySetMode(target)" in memory
    assert "await api.memoryStrength(v)" in memory


def test_memory_copy_names_product_lab_and_strength_boundaries():
    memory = _read(HEAVN / "modules" / "memory.mjs")
    for phrase in ("Internalized soft-prefix memory is lab-only",
                   "card changes can retrain for minutes",
                   "Prompt mode is binary: 0 keeps cards out",
                   "Positive values do not scale intensity",
                   "Saving does not retrain", "POST /memory/strength did not answer"):
        assert phrase in memory


def test_memory_control_styles_are_responsive_and_have_result_states():
    theme = _read(HEAVN / "theme.css")
    for selector in (".memory-controls-body", ".memory-mode-control", ".memory-strength-control",
                     ".memory-mode-options", ".memory-control-message.warn",
                     ".memory-control-message.error"):
        assert selector in theme
    assert ".memory-controls-body{grid-template-columns:1fr}" in theme
