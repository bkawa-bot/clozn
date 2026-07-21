"""Static product contracts for the developer-home experiment matrix."""
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "studio" / "heavn" / "modules" / "experiment.mjs"


def _source() -> str:
    return MODULE.read_text(encoding="utf-8")


def test_matrix_precedes_existing_form_and_keeps_execution_path():
    source = _source()
    assert source.index("<${ExperimentMatrix}") < source.index("<${FormPanel}")
    assert "api.experimentTypes()" in source
    assert "api.runExperiment(rec.id, change" in source
    assert "prefillFromRun(nextType, rec)" in source
    assert "pendingRef.current = { ctype: nextType, fields: prefill" in source


def test_matrix_reads_catalog_fields_verbatim_and_has_honest_fallbacks():
    source = _source()
    for field in ("catalog.label", "catalog.substrate", "catalog.op",
                  "catalog.control", "catalog.cost_hint"):
        assert field in source
    assert 'return "not reported"' in source
    assert "required capability is verified on submit" in source
    assert 'if(!rec) return { ready: false' in source
    assert 'if(!live || rec._sample) return { ready: false' in source


def test_matrix_has_accessible_table_selection_and_root_style_hooks():
    source = _source()
    for markup in ('<table class="experiment-matrix-table"', '<caption class="experiment-matrix-caption"',
                   '<th scope="col">', 'scope="row"', 'aria-pressed=', 'aria-selected='):
        assert markup in source
    for hook in ("experiment-matrix-head", "experiment-matrix-row", "experiment-matrix-type",
                 "experiment-matrix-change", "experiment-matrix-capability",
                 "experiment-matrix-method-control", "experiment-matrix-cost",
                 "experiment-matrix-state", "experiment-matrix-select"):
        assert hook in source
    assert "onClick=${() => selectType(type)}" in source
    assert "event.stopPropagation(); selectType(type);" in source


def test_experiment_module_javascript_syntax():
    node = shutil.which("node")
    if node:
        subprocess.run([node, "--check", str(MODULE)], check=True, capture_output=True, text=True)

