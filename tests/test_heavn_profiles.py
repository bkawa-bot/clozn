"""Static contract checks for the model-free heavn Profiles Settings surface.

The interactive behavior is browser-verified; these checks keep the no-build ESM wiring, API paths, and
honesty copy from silently disappearing during refactors without requiring Node, a model, or a GPU.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_settings_module_replaces_the_stub_in_the_heavn_router():
    app = _read(HEAVN / "app.mjs")
    assert 'import { SettingsModule } from "./modules/settings.mjs"' in app
    assert 'id: "settings"' in app and "view: SettingsModule" in app
    assert "SettingsStub" not in app


def test_profile_api_exposes_every_ui_operation():
    client = _read(HEAVN / "api.mjs")
    for path in ("/profiles/list", "/profiles/save", "/profiles/switch", "/profiles/export",
                 "/profiles/import", "/profiles/delete"):
        assert path in client
    # Mutations need postE so server validation/rejection text reaches the UI.
    for method in ("profilesSave", "profilesSwitch", "profilesExport", "profilesImport", "profilesDelete"):
        declaration = next(line for line in client.splitlines() if method + ":" in line)
        assert "postE(" in declaration


def test_settings_profile_surface_names_replacement_and_portability_boundaries():
    module = _read(HEAVN / "modules" / "settings.mjs")
    assert "Switching replaces" in module
    assert "personas never blend" in module
    assert "vectors" in module and "never exported" in module
    assert "preserved fact source" in module
    assert "UPDATE FROM LIVE" in module and "CONFIRM DELETE" in module
    assert 'data-testid="profile-list"' in module


def test_profile_styles_include_responsive_and_action_states():
    theme = _read(HEAVN / "theme.css")
    for selector in (".settings-grid", ".profile-row.active", ".profile-actions",
                     ".profile-message.error", ".spd.danger.armed"):
        assert selector in theme
    assert "@media(max-width:1050px)" in theme
