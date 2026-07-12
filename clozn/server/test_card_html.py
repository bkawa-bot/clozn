"""Tests for the shareable receipt card renderer (card_html.render_card) — pure fixture bundles in,
assertions on the HTML string out. No server, no substrate, no fs."""
from __future__ import annotations

import re

import clozn.receipts.bundle as receipt_bundle
from clozn.server.card_html import render_card


def _run(**over) -> dict:
    run = {
        "id": "run_test0000001_abc123",
        "created_at": "2026-07-12T04:11:19",
        "created_ts": 1783854679.9,
        "source": "openai_api",
        "client": "studio",
        "model": "clozn-qwen",
        "substrate": "engine",
        "prompt_summary": "What country is shaped like a boot?",
        "response_summary": "Italy is shaped like a boot.",
        "messages": [{"role": "user", "content": "What country is shaped like a boot?"}],
        "response": "Italy is shaped like a boot.",
        "memory": {"cards_applied": ["enjoys geography trivia"], "applied_ids": ["mem_1"],
                   "mode": "prompt", "gate": 0.62},
        "behavior": {"active_dials": {}},
        "trace": {
            "tokens": ["Italy", " is", " shaped", " like", " a", " boot", "."],
            "confidence": [0.91, 0.98, 0.95, 0.98, 0.99, 0.44, 0.99],
            "alternatives": [],
        },
        "timing": {"started_at": 1.0, "ended_at": 1.4, "duration_ms": 400},
        "parent_run_id": None,
        "finish_reason": "stop",
        "error": None,
        "meta": {},
    }
    run.update(over)
    return run


_RECEIPTS = {
    "run_id": "run_test0000001_abc123",
    "receipts": [{
        "influence": {"card_id": "mem_1", "text": "enjoys <geography> trivia"},
        "has_effect": True,
        "causal_verified": True,
        "baseline_reply": "Italy is shaped like a boot.",
        "ablated_reply": "The country shaped like a boot is Italy.",
        "delta": {"words": [6, 8], "wps": [6.0, 8.0], "changed": 38},
    }],
    "forced_receipts": [{
        "influence": {"card_id": "mem_1", "text": "enjoys <geography> trivia"},
        "mode": "forced",
        "causal_verified": True,
        "has_effect": True,
        "sum_nats": 2.437,
        "mean_nats_per_token": 0.348,
        "top_dependent": [
            {"index": 0, "piece": " Italy", "delta": 1.2},
            {"index": 5, "piece": " boot", "delta": -0.31},
        ],
    }],
    "skipped": [],
    "redundant_pairs": [],
}


def _bundle(run=None, receipts=None) -> dict:
    return receipt_bundle.build(run if run is not None else _run(), explain=None, receipts=receipts)


# --------------------------------------------------------------- receipts present: numbers, escaped
def test_receipts_render_with_numbers_and_escaping():
    out = render_card(_bundle(receipts=_RECEIPTS))
    # forced-receipt numbers appear
    assert "+2.437" in out               # sum_nats
    assert "+1.200" in out               # top-dependent delta
    assert "-0.310" in out               # negative delta
    assert " Italy" in out               # dependent-token piece
    # influence text appears ESCAPED, never raw
    assert "enjoys &lt;geography&gt; trivia" in out
    assert "enjoys <geography>" not in out
    # verdict chips + derivation tag
    assert "changed the answer" in out
    assert "leave-one-out + forced scoring" in out
    # the honest-absence sentence must NOT appear when receipts exist
    assert "no receipts computed for this run" not in out


# --------------------------------------------------------------------- no receipts: honest absence
def test_no_receipts_renders_honest_absence():
    out = render_card(_bundle())
    assert ("no receipts computed for this run — receipts are measured on demand, "
            "never assumed") in out


# ------------------------------------------------------------------------------- injection-proofing
def test_script_in_reply_is_escaped_inert():
    payload = "<script>alert(1)</script>"
    run = _run(response=payload,
               messages=[{"role": "user", "content": "say " + payload}],
               trace={"tokens": ["<script>", "alert(1)", "</script>"],
                      "confidence": [0.9, 0.9, 0.9], "alternatives": []})
    out = render_card(_bundle(run=run))
    assert "<script" not in out                      # the card ships zero <script> of its own, so: none at all
    assert "&lt;script&gt;" in out                   # the payload is present, inert


# ------------------------------------------------------------------------------- self-containment
def test_no_external_urls_in_src_or_href_attributes():
    out = render_card(_bundle(receipts=_RECEIPTS))
    attrs = re.findall(r"""(?:src|href)\s*=\s*["'][^"']*["']""", out, flags=re.IGNORECASE)
    for a in attrs:
        assert "http://" not in a and "https://" not in a, a
    assert "@import" not in out and "url(" not in out    # CSS can't fetch either


# ------------------------------------------------------------------------------- structure smoke
def test_structure_smoke():
    out = render_card(_bundle(receipts=_RECEIPTS))
    assert out.startswith("<!doctype html>")
    assert out.count("<html") == 1 and out.count("</html>") == 1
    # sections, in order
    for section in ("run receipt", "the exchange", "influences &amp; receipts",
                    "lens readouts", "lineage", "measured, not asserted"):
        assert section in out, section
    # masthead + footer both carry the run id
    assert out.count("run_test0000001_abc123") >= 2
    # the CRT (twilight indigo, never black) and phosphor text are present
    assert "#2B3160" in out and "#1E2447" in out and "#B8F5E4" in out
    # per-token confidence shading: opacity varies with conf, low-conf gets the dotted class
    assert 'class="tk lo"' in out
    # captured/derived legend
    assert "recorded at generation time" in out
    assert "computed afterwards from the record" in out


# ------------------------------------------------------------------------------- lens readouts
def test_lens_readouts_render_with_provenance():
    run = _run()
    run["trace"]["workspace_readouts"] = [
        {"type": "workspace_readout", "run_id": run["id"], "token_index": 0, "token_text": "Italy",
         "layer": 25, "position": 0, "provider": "jacobian_lens_l25",
         "provider_type": "jacobian_lens", "readout_kind": "token",
         "top_readouts": [{"label": " peninsula", "score": 0.8}, {"label": " coast", "score": 0.5}]},
    ]
    out = render_card(_bundle(run=run))
    assert "layer 25" in out
    assert "peninsula" in out and "coast" in out
    assert "jacobian_lens_l25" in out
    assert "NOT the model&#x27;s literal thought" in out     # the honesty caption, escaped
    assert "no lens readout recorded" not in out


def test_lens_absent_renders_honest_one_liner():
    out = render_card(_bundle())
    assert "no lens readout recorded on this run" in out


# ------------------------------------------------------------------------------------ lineage
def test_lineage_tree_renders_parent_and_children():
    bundle = _bundle(run=_run(parent_run_id="run_parent00001_aaaaaa"))
    bundle["lineage"] = {
        "run_id": "run_test0000001_abc123",
        "tree": {"id": "run_parent00001_aaaaaa", "change_label": None, "children": [
            {"id": "run_test0000001_abc123", "change_label": "re-roll", "is_current": True,
             "children": [{"id": "run_child000001_bbbbbb", "change_label": "memory off",
                           "children": []}]},
        ]},
    }
    out = render_card(bundle)
    assert "run_parent00001_aaaaaa" in out
    assert "run_child000001_bbbbbb" in out
    assert "this run" in out


def test_lineage_absent_is_honest():
    out = render_card(_bundle())
    assert "no lineage — an original run" in out


# ------------------------------------------------------------------------- degrades, never raises
def test_degenerate_bundles_never_raise():
    for bundle in ({}, {"run": None}, {"run": {}, "trace": None},
                   receipt_bundle.build(None)):
        out = render_card(bundle)
        assert out.startswith("<!doctype html>")
        assert "measured, not asserted" in out
