"""Static contracts for Studio's provenance surfacing (POST /runs/<id>/provenance, the
attention-knockout context-vs-parametric receipt) -- BK's two-piece ask: a plain-language answer-source
chip wherever a run shows its answer, and an on-demand "trace this token" action in the click-a-token
popover that honestly falls back to the wired provenance route (no causal-trace server route exists
yet -- it stays CLI-only) instead of inventing a heavy live path.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_api_exposes_the_provenance_route():
    api = _read(HEAVN / "api.mjs")
    assert '"/runs/" + enc(id) + "/provenance"' in api
    assert "provenance:" in api


def test_all_four_verdicts_map_to_bks_plain_language_labels():
    module = _read(HEAVN / "provenance.mjs")
    # verbatim, per the task spec -- no strength/confidence word added to any of them
    mapping = {
        "CONTEXT_CARRIED": "Answered from your context",
        "MIXED": "From context + the model",
        "PARAMETRIC": "From the model's own knowledge",
        "INCONCLUSIVE": "Couldn't determine the source",
    }
    for verdict, label in mapping.items():
        assert verdict in module
        assert label in module


def test_blocked_and_unavailable_states_render_honestly_never_a_fake_verdict():
    module = _read(HEAVN / "provenance.mjs")
    assert "Provenance unavailable" in module
    assert "provenance unavailable" in module
    assert "needs a cloze-server started with --no-flash-attn" in module
    # the loose (falsy) `ok` check -- a 404/400 wire shape with no `ok` key must also read unavailable,
    # never fall through to a fabricated verdict lookup
    assert "!receipt.ok" in module or "!receipt || !receipt.ok" in module


def test_chip_is_computed_on_demand_never_pre_attached():
    module = _read(HEAVN / "provenance.mjs")
    assert 'status: "idle"' in module
    assert 'status: "busy"' in module
    assert 'status: "done"' in module
    # never auto-fires: only a click drives the fetch
    assert "onClick=${check}" in module


def test_chip_shares_one_cache_with_the_pop_action():
    module = _read(HEAVN / "provenance.mjs")
    assert "export function getProvenance(rec)" in module
    assert "provCache" in module


def test_chip_is_wired_into_the_monitor_next_to_the_policy_chip():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert 'import { ProvenanceChip' in replay
    assert "<${ProvenanceChip} rec=${rec}/>" in replay
    # never shown mid-stream against a stale previous run's receipt
    assert "!liveView && html`<${ProvenanceChip}" in replay


def test_click_a_token_popover_has_a_trace_this_token_action():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "trace this token" in replay
    assert "const traceToken = async ()" in replay
    assert "await getProvenance(rec)" in replay


def test_causal_trace_route_now_exists_and_the_cli_affordance_still_stands():
    # A causal-trace route was added AFTER this Studio work (POST /runs/<id>/causal-trace, per
    # position, contrastive) so the panel CAN be wired to it; until that wiring lands, the panel
    # keeps the honest terminal affordance. This test tracks BOTH facts.
    routes_dir = ROOT / "clozn" / "server" / "routes"
    route_files = {p.stem for p in routes_dir.glob("*.py")}
    assert "causal_trace" in route_files
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "clozn causal-trace --from-run" in replay   # the CLI affordance still present
    assert "--contrast auto" in replay


def test_trace_this_token_never_claims_precision_the_wired_route_lacks():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "not necessarily token" in replay
    assert "no per-position scoring yet" in replay


def test_trace_this_token_names_the_full_causal_receipts_honest_shape():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    for phrase in ("PASS", "NO_CAUSAL_NODES", "FAILED_CONTROLS", "control-ratio", "legibility",
                   "individual sites rarely carry the answer", "causal skeleton, not a full explanation"):
        assert phrase in replay


def test_provenance_rows_render_the_scored_text_verbatim_not_a_paraphrase():
    module = _read(HEAVN / "provenance.mjs")
    assert '"scored text"' in module
    assert '"verdict"' in module
    assert '"dependence"' in module
    assert '"best control ratio"' in module
    assert '"carrying span"' in module
    assert "focus_null" in module


def test_provenance_chip_css_is_teal_accented_distinct_from_the_policy_chip():
    theme = _read(HEAVN / "theme.css")
    assert "details.prov-chip summary" in theme
    assert ".prov-chip-idle" in theme
    assert ".prov-chip-unavailable" in theme
    assert ".prov-chip-body" in theme
