"""Static product contracts for anchored and internalized memory surfacing."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_memory_desk_names_the_two_learned_carriers_without_hiding_prompt_cards():
    memory = _read(HEAVN / "modules" / "memory.mjs")
    assert 'data-testid="memory-carrier-map"' in memory
    for phrase in ("ANCHORED · PRODUCT", "sparse α lookup", "attempts an equal-magnitude random control", "INTERNALIZED · LAB",
                   "opaque trained carrier", "Prompt cards are a separate readable-context route"):
        assert phrase in memory


def test_each_anchored_bag_can_run_the_existing_null_controlled_experiment():
    memory = _read(HEAVN / "modules" / "memory.mjs")
    assert '<${AnchoredShelf} cards=${cards} live=${live} rec=${rec}/>' in memory
    assert 'api.runExperiment(rec.id, { type: "anchored_recall", card_id: bag.card_id })' in memory
    assert "PROVE ON THIS RUN" in memory
    assert "cost: 2–3 fresh model generations · no KV reuse" in memory
    assert 'data-testid="anchored-proof"' in memory


def test_proof_preserves_tri_state_verdict_and_random_control_evidence():
    memory = _read(HEAVN / "modules" / "memory.mjs")
    for phrase in ("result.causal_verified === true", "result.has_effect === true",
                   "EFFECT BEYOND NULL", "NO EFFECT BEYOND NULL", "not computed",
                   "EFFECT VS BASELINE · NULL MISSING", "baseline-only, weaker evidence",
                   "lexicon hits", "target logprob", "random control"):
        assert phrase in memory


def test_memory_tier_and_receipt_layouts_collapse_on_narrow_screens():
    theme = _read(HEAVN / "theme.css")
    for selector in (".memory-carrier-map", ".anchored-proof", ".anchored-proof-metrics",
                     ".anchored-proof-arms"):
        assert selector in theme
    assert ".memory-carrier-map{grid-template-columns:1fr}" in theme
    assert ".anchored-proof-metrics,.anchored-proof-arms{grid-template-columns:1fr}" in theme
