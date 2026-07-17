"""Asset-level guard that heavn remains the sole product frontend."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "studio"


def test_legacy_product_frontend_is_removed():
    assert not (STUDIO / "engine.html").exists()
    pages = STUDIO / "pages"
    assert not pages.exists() or not any(pages.iterdir())


def test_heavn_product_entrypoint_remains_packaged():
    assert (STUDIO / "heavn" / "index.html").is_file()
    assert (STUDIO / "heavn" / "app.mjs").is_file()


def test_lab_denoise_page_has_no_links_to_cut_legacy_assets():
    denoise = (STUDIO / "denoise.html").read_text(encoding="utf-8")
    for dead in ("app.html", "brain.html", "engine.html", "pages/"):
        assert dead not in denoise
