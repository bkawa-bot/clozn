"""Static contracts for the read-only Studio Evidence home."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "studio" / "heavn" / "modules" / "evidence.mjs"
APP = ROOT / "studio" / "heavn" / "app.mjs"


def _source() -> str:
    return EVIDENCE.read_text(encoding="utf-8")


def test_evidence_home_uses_the_shared_run_evidence_experiment_objects():
    source = _source()
    assert "normalizeRun as runObject" in source
    assert "normalizeEvidence as evidenceObjects" in source
    assert "normalizeExperiment as experimentObject" in source
    assert "callObjectModel(runObject, rec)" in source
    assert "callObjectModel(evidenceObjects, run.raw || rec, exact.runId)" in source
    assert "callObjectModel(experimentObject, storedExperiment)" in source
    assert "export function EvidenceModule()" in source


def test_evidence_is_a_first_class_home_beside_replay_and_experiment():
    source = APP.read_text(encoding="utf-8")
    assert 'import { EvidenceModule } from "./modules/evidence.mjs"' in source
    positions = [source.index(f'id: "{route}"') for route in ("replay", "experiment", "evidence")]
    assert positions == sorted(positions)
    assert 'id: "evidence", nm: "Evidence", sub: "method & controls", view: EvidenceModule' in source


def test_evidence_cards_have_fixed_product_order_and_honest_absence():
    source = _source()
    titles = (
        'title: "Context ↔ answer map"',
        'title: "Causal receipts"',
        'title: "Calibration and trust"',
        'title: "Lens and workspace readouts"',
    )
    positions = [source.index(title) for title in titles]
    assert positions == sorted(positions)
    assert 'class="evidence-home-absent" role="status"' in source
    assert "Evidence Home never computes missing evidence" in source
    assert "Nothing on this page triggers model work" in source


def test_each_stored_object_surfaces_evidence_contract_fields():
    source = _source()
    for field in ("status", "method", "controls", "latency", "artifact", "provenance"):
        assert field in source
    assert '["run", exact.runId]' in source
    assert '["model", exact.model]' in source
    assert 'display(value, "not recorded")' in source


def test_evidence_home_navigation_is_navigation_only():
    source = _source()
    assert 'store.set({ route: "replay" })' in source
    assert 'route: "experiment"' in source
    assert "api.cardUrl(runId)" in source
    for computation in ("api.influenceMap(", "api.receipts(", "api.receipt(",
                        "api.trustSpans(", "api.jlens(", "api.runExperiment("):
        assert computation not in source


def test_evidence_home_is_semantic_accessible_and_avoids_unsupported_claims():
    source = _source()
    for element in ("<main", "<article", "<header", "<nav", "<dl", "<dt>", "<dd>"):
        assert element in source
    assert 'aria-label="Stored evidence, ordered by decision relevance"' in source
    assert 'aria-label="Evidence destinations"' in source
    assert "evidence-home-root" in source
    assert "evidence-home-card" in source
    assert "%" not in source
    assert "percentage" not in source.lower()
    assert "circuit" not in source.lower()
