"""Static contract checks for the model-free heavn Facts panel.

The server behavior has full fake-slot coverage in test_facts_server.py. These checks protect the no-build
ESM wiring and the honesty-critical UI copy without importing a model, allocating GPU memory, or needing a
browser in the routine test suite.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_memory_desk_mounts_the_facts_panel():
    memory = _read(HEAVN / "modules" / "memory.mjs")
    assert 'import { FactsPanel } from "./facts.mjs"' in memory
    assert "<${FactsPanel} live=${live}/>" in memory


def test_api_exposes_every_facts_operation_with_server_errors():
    client = _read(HEAVN / "api.mjs")
    for path in ("/facts/list", "/facts/mode", "/facts/add", "/facts/delete", "/facts/read"):
        assert path in client
    for method in ("factsList", "factsMode", "factsAdd", "factsDelete", "factsRead"):
        declaration = next(line for line in client.splitlines() if method + ":" in line)
        assert "postE(" in declaration


def test_panel_names_latency_surprise_abstention_and_v1_boundary():
    panel = _read(HEAVN / "modules" / "facts.mjs")
    for phrase in ("OFF BY DEFAULT", "extra model pass", "does not change the chat reply",
                   "surprise-gated", "abstain", "slot_ms", "CONFIRM DELETE"):
        assert phrase in panel
    assert 'data-testid="fact-receipt"' in panel
    assert 'data-testid="fact-list"' in panel


def test_panel_renders_all_read_receipt_outcomes():
    panel = _read(HEAVN / "modules" / "facts.mjs")
    assert "receipt.empty" in panel
    assert 'receipt.abstained === true || receipt.hit == null' in panel
    assert 'abstained ? "ABSTAINED" : "HIT"' in panel
    for field in ("sim", "gate_floor", "slot_ms", "count"):
        assert f"receipt.{field}" in panel


def test_fact_styles_cover_receipts_actions_and_small_screens():
    theme = _read(HEAVN / "theme.css")
    for selector in (".facts-topline", ".fact-row", ".fact-form", ".fact-receipt.hit",
                     ".fact-receipt.abstain", ".fact-message.warn"):
        assert selector in theme
    assert "@media(max-width:720px)" in theme
