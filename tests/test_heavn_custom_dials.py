"""Static contracts for Patch's user-created pole-pair dial surface."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_custom_dial_api_keeps_errors_and_allows_slow_model_work():
    client = _read(HEAVN / "api.mjs")
    assert 'steerCustom: (name, pos, neg) => postE("/steer/custom"' in client
    assert "{ name, pos, neg }, 300000" in client
    assert 'steerCustomDelete: name => postE("/steer/custom_delete"' in client


def test_patch_mounts_maker_and_refreshes_axes_after_create_and_delete():
    patch = _read(HEAVN / "modules" / "patch.mjs")
    assert '<${CustomDialMaker} axesState=${axesState} onCreated=${loadAxes}/>' in patch
    assert '<${DialsPanel} axesState=${axesState} onChanged=${loadAxes}/>' in patch
    assert 'data-testid="custom-dial-maker"' in patch
    assert "await api.steerCustom(n, p, q)" in patch
    assert "await api.steerCustomDelete(axis.name)" in patch
    assert "await onCreated()" in patch
    assert "await onChanged()" in patch


def test_maker_bounds_and_validates_its_inputs_before_model_work():
    patch = _read(HEAVN / "modules" / "patch.mjs")
    assert 'maxlength="24"' in patch
    assert patch.count('maxlength="320"') == 2
    assert "if(p === q)" in patch
    assert ".toLowerCase() === n.toLowerCase()" in patch
    assert "Need a name and both pole descriptions." in patch


def test_custom_dial_copy_is_honest_about_cost_and_evidence():
    patch = _read(HEAVN / "modules" / "patch.mjs")
    for phrase in ("Model work.", "use the GPU", "not</b> run calibration",
                   "prove that the dial changes behavior", "recipe is saved; calibration has not been run"):
        assert phrase in patch


def test_custom_dial_styles_cover_mobile_and_all_outcomes():
    theme = _read(HEAVN / "theme.css")
    for selector in (".custom-dial-form", ".custom-dial-rule", ".custom-dial-actions",
                     ".custom-dial-message.info", ".custom-dial-message.ok",
                     ".custom-dial-message.error", ".steer-delete.armed"):
        assert selector in theme
    assert ".custom-dial-form{grid-template-columns:1fr}" in theme
