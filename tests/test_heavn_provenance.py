"""Static contracts for Studio's provenance/causal-trace surfacing -- BK's two-piece ask:

1. A plain-language answer-source CHIP wherever a run shows its answer (Monitor, replay.mjs),
   built on POST /runs/<id>/provenance (attention-knockout context-vs-parametric receipt).
2. An on-demand "trace this token" ACTION in the click-a-token popover (Pop, replay.mjs), built on
   POST /runs/<id>/causal-trace (clozn/server/routes/causal_trace.py, added after piece 1 shipped)
   -- a real per-position ablation trace of the clicked continuation token, contrastive by default.

Both pieces are computed on demand (idle -> busy -> done), never pre-attached, and degrade honestly
on a blocked/failed receipt -- never a crash, never a fabricated verdict.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------- piece 1: the chip

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
    assert "export function getProvenance(rec)" in module
    assert "provCache" in module


def test_chip_is_wired_into_the_monitor_next_to_the_policy_chip():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert 'import { ProvenanceChip' in replay
    assert "<${ProvenanceChip} rec=${rec}/>" in replay
    # never shown mid-stream against a stale previous run's receipt
    assert "!liveView && html`<${ProvenanceChip}" in replay


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


def test_provenance_scope_caveat_matches_the_backends_corrected_two_family_note():
    # clozn/analysis/provenance.py's SCOPE_NOTE was corrected (8fac4ff): TWO model families, not one,
    # and the 41/41 figure only reproduces under the current grading code. Studio's own restatement
    # (the SCOPE_CAVEAT constant actually shown in the chip) must carry the corrected claim -- the
    # module comment nearby is allowed to mention the old "one model family" phrasing as history.
    module = _read(HEAVN / "provenance.mjs")
    assert 'const SCOPE_CAVEAT = "attention-knockout measurement' in module
    assert "two-family" in module or "two model families" in module
    assert "Qwen2.5-7B" in module and "Llama-3.1-8B" in module
    assert "under current grading" in module or "under the current grading" in module


# --------------------------------------------------------------- piece 2: click-a-token causal trace

def test_causal_trace_route_exists_and_api_exposes_it():
    # POST /runs/<id>/causal-trace (300c8e5).
    routes_dir = ROOT / "clozn" / "server" / "routes"
    route_files = {p.stem for p in routes_dir.glob("*.py")}
    assert "causal_trace" in route_files
    api = _read(HEAVN / "api.mjs")
    assert '"/runs/" + enc(id) + "/causal-trace"' in api
    assert "causalTrace" in api
    # answer-selective + any-engine defaults, per the route's own contract
    assert 'contrast = "auto"' in api
    assert 'screen_mode = "ablate"' in api


def test_click_a_token_popover_actually_traces_the_clicked_position():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "trace this token" in replay
    assert "const traceToken = async ()" in replay
    # the clicked token index `i` rides as `position` -- this is a REAL per-position trace now, not
    # a provenance-route fallback
    assert "await api.causalTrace(rec.id, { position: i" in replay
    assert 'contrast: "auto"' in replay
    assert 'screen_mode: "ablate"' in replay
    # the old provenance-fallback precision caveat must be gone -- the route it described no longer
    # applies (the new route DOES support per-position scoring), so keeping that text would now
    # be a false claim of a limitation that's been fixed
    assert "no per-position scoring yet" not in replay


def test_trace_action_is_on_demand_idle_busy_done():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert 'useState({ status: "idle", receipt: null })' in replay
    assert '{ status: "busy", receipt: null }' in replay
    assert '{ status: "done"' in replay


def test_trace_renders_all_three_honest_verdicts_and_failed_controls_is_a_warning():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    for phrase in ("PASS", "NO_CAUSAL_NODES", "FAILED_CONTROLS"):
        assert phrase in replay
    # FAILED_CONTROLS must read as an explicit warning, never silently blended in with PASS
    assert 'verdict === "FAILED_CONTROLS"' in replay
    assert "do not trust this trace" in replay
    assert "random interventions moved the target" in replay


def test_trace_renders_node_fields_sorted_by_delta_full():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    for field in ("n.layer", "n.pos", "n.control_ratio", "n.strength", "n.legibility", "n.name"):
        assert field in replay
    assert "control_ratio" in replay and 'toFixed(1) + "x"' in replay
    assert "Math.round(n.legibility * 100)" in replay
    assert 'Math.abs(b.delta_full || 0) - Math.abs(a.delta_full || 0)' in replay


def test_trace_no_nodes_reads_as_an_honest_finding_not_an_error():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "no nodes beat the noise floor" in replay


def test_trace_always_carries_the_distributed_function_caveat():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "individual sites rarely carry the answer" in replay
    assert "causal skeleton, not a full explanation" in replay


def test_trace_blocked_state_is_quiet_and_honest_never_a_crash():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "trace unavailable" in replay
    assert "needs the engine's ablation screen" in replay


def test_terminal_affordance_kept_as_a_secondary_path():
    replay = _read(HEAVN / "modules" / "replay.mjs")
    assert "clozn causal-trace --from-run" in replay
    assert "--contrast auto" in replay
    assert "full JSON receipt" in replay   # framed as secondary/export, not the primary path anymore
