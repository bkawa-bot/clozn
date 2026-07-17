"""Static contracts for the document-first permalink landing."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_read_is_a_real_heavn_module_and_permalink_default():
    app = _read(HEAVN / "app.mjs")
    assert 'import { ReadModule } from "./modules/read.mjs"' in app
    assert '{ id: "read",     nm: "Read",     sub: "answer first"' in app
    assert 'if(deep) store.set({ route: "read", readRequest: deep })' in app


def test_missing_permalink_never_substitutes_an_unrelated_run():
    app = _read(HEAVN / "app.mjs")
    assert 'readError: `Run "${deep}" was not found in this local journal.`' in app
    assert "if(!r && list.runs.length)" not in app


def test_read_view_is_record_only_and_labels_confidence_honestly():
    read = _read(HEAVN / "modules" / "read.mjs")
    assert "const LOW_CONF = 0.5" in read
    assert "function sketchySpans(steps)" in read
    assert "await api." not in read
    assert "fetch(" not in read
    for phrase in ("raw token confidence measures commitment, not truth",
                   "A low-confidence span is a locator for inspection",
                   "not that the answer was correct",
                   "No token trace was captured"):
        assert phrase in read


def test_read_view_groups_spans_and_exposes_recorded_alternatives():
    read = _read(HEAVN / "modules" / "read.mjs")
    assert "active.end = i + 1" in read
    assert "step.alts.slice(0,4)" in read
    assert "selected span · tokens" in read
    assert 'store.set({ route: "replay", P: steps.length })' in read


def test_read_styles_cover_document_zoom_and_mobile_layouts():
    theme = _read(HEAVN / "theme.css")
    for selector in (".read-grid", ".read-document", ".read-answer", ".read-token.sketchy",
                     ".read-legend", ".read-zoom", ".read-span-list", ".read-detail"):
        assert selector in theme
    assert ".read-grid{grid-template-columns:1fr}" in theme
    assert ".read-head{flex-direction:column" in theme
