"""Static product contracts for Studio's context-to-answer influence explorer."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVN = ROOT / "studio" / "heavn"


def _read(name: str) -> str:
    return (HEAVN / name).read_text(encoding="utf-8")


def test_influence_map_is_mounted_and_calls_the_bounded_post_route():
    api = _read("api.mjs")
    replay = _read("modules/replay.mjs")
    assert '<${InfluenceMap} rec=${rec}/>' in replay
    assert 'data-testid="influence-map"' in replay
    assert 'postE("/runs/" + enc(id) + "/influence-map"' in api
    assert "await api.influenceMap(rec.id, refresh ? { refresh: true } : {})" in replay
    assert "600000" in api


def test_influence_map_links_both_directions_for_pointer_and_keyboard():
    replay = _read("modules/replay.mjs")
    for contract in (
        'active.kind === "context"',
        'active.kind === "answer"',
        "onMouseEnter=",
        "onFocus=",
        "onBlur=",
        "togglePin",
        "const active = pinned || hovered || focused;",
        "onClick=",
        "aria-pressed=",
        'aria-describedby="influence-map-status"',
        'aria-live="polite"',
        'data-strongest=',
    ):
        assert contract in replay


def test_influence_map_preserves_evidence_floor_and_claim_boundaries():
    replay = _read("modules/replay.mjs")
    assert "clears_floor" in replay
    assert "below-floor measurements remain subdued" in replay
    assert "no clear source" in replay
    assert "controlled behavioral dependence" in replay
    assert "not an attention path or circuit explanation" in replay
    component = replay[replay.index("function InfluenceMap"):replay.index("function SpanForensics")]
    assert "percentage" not in component.lower()
    assert "percent" not in component.lower()


def test_influence_map_styles_encode_strength_and_absence_accessibly():
    theme = _read("theme.css")
    for selector in (
        ".influence-map-grid",
        ".influence-context-span",
        ".influence-answer-span",
        '[data-clears-floor="false"]',
        '[data-no-clear-source="true"]',
        '[data-strongest="true"]',
    ):
        assert selector in theme
    assert "@media(prefers-reduced-motion:reduce)" in theme
    assert "@media(max-width:900px){.influence-map-grid{grid-template-columns:1fr}" in theme
