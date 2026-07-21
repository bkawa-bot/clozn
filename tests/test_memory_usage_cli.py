"""Focused coverage for the evidence-only memory usage receipt."""
from __future__ import annotations

import json

import pytest

from clozn.cli import main as cli
from clozn.cli.commands.memory import format_memory_usage
from clozn.runs.memory_usage import memory_usage
import clozn.runs.store as runlog


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))


def test_prompt_receipt_reports_exact_evidence_without_estimating_memory_tokens():
    block = "Memory: café"
    run = {
        "id": "run_exact",
        "memory": {
            "mode": "prompt",
            "cards_applied": ["likes concise answers"],
            "applied_ids": ["mem_1"],
            "applied_scope_kinds": ["project"],
            "relevance": [0.812345],
            "gate": 0.91,
            "strength": 1.0,
            "prompt_block": block,
            "candidate_cards": [
                {"id": "mem_1", "text": "likes concise answers"},
                {"id": "mem_3", "text": "likes long answers"},
            ],
            "omitted_cards": [{"id": "mem_3", "text": "likes long answers"}],
            "selection_stage": "active_prompt_cards_considered_by_turn_gate",
            "omission_reason": "topic_gate_below_threshold",
            "baseline_prompt_tokens": 18,
            "prompt_token_cost": 7,
            "anchored": [{"card_id": "mem_2", "coef": 0.4}],
            "anchored_layer": 8,
            "anchored_s_total": 0.4,
            "anchored_scope_excluded_count": 2,
            "facts": {"read": {"ids": ["fact_1"], "slot_ms": 2.5}},
        },
        "changes_applied": {"disabled_memory_ids": ["mem_3"]},
        "context_receipt": {"limits": {"prompt_tokens": 25}},
        "meta": {"prompt_tokens": 99},
    }

    receipt = memory_usage(run)

    injected = receipt["prompt_cards"]["injected"]
    assert injected["cards"] == [{"text": "likes concise answers", "id": "mem_1",
                                   "relevance": 0.8123, "scope_kind": "project"}]
    assert receipt["prompt_cards"]["selected"]["status"] == "observed"
    assert [card["id"] for card in receipt["prompt_cards"]["selected"]["cards"]] == ["mem_1", "mem_3"]
    assert receipt["prompt_cards"]["omitted"]["ids"] == ["mem_3"]
    assert receipt["prompt_cards"]["omitted"]["reason"] == "topic_gate_below_threshold"
    assert receipt["token_cost"] == {
        "status": "observed",
        "memory_prompt_tokens": 7,
        "prompt_block_utf8_bytes": len(block.encode("utf-8")),
        "baseline_prompt_tokens": 18,
        "total_prompt_tokens": 25,
        "total_prompt_tokens_source": "context_receipt.limits.prompt_tokens",
        "evidence": ["memory.prompt_token_cost", "memory.baseline_prompt_tokens"],
        "note": "Exact matched prompt-token delta captured for this run.",
    }
    assert receipt["anchored"]["bags"][0]["card_id"] == "mem_2"
    assert receipt["anchored"]["scope_excluded_count"] == 2
    assert receipt["facts"]["evidence"]["read"]["ids"] == ["fact_1"]
    rendered = format_memory_usage(receipt)
    assert "selected - 2 card(s), capture-time" in rendered
    assert "omitted - mem_3" in rendered
    assert "token cost - 7 prompt-memory tokens (exact delta)" in rendered
    assert "mem_1: likes concise answers (project, relevance 0.8123)" in rendered
    assert "excluded by scope: 2" in rendered


def test_no_prompt_block_has_zero_cost_but_unknown_omissions():
    receipt = memory_usage({
        "id": "run_empty",
        "memory": {"mode": "prompt", "cards_applied": [], "applied_ids": []},
    })

    assert receipt["prompt_cards"]["injected"]["count"] == 0
    assert receipt["prompt_cards"]["selected"]["cards"] == []
    assert receipt["prompt_cards"]["omitted"]["status"] == "unavailable"
    assert receipt["token_cost"]["memory_prompt_tokens"] == 0
    assert receipt["token_cost"]["prompt_block_utf8_bytes"] == 0


def test_historical_prompt_block_keeps_derived_selection_and_unavailable_delta():
    receipt = memory_usage({
        "id": "run_old",
        "memory": {"mode": "prompt", "cards_applied": ["be brief"],
                   "applied_ids": ["mem_1"], "prompt_block": "Memory: be brief"},
        "meta": {"prompt_tokens": 12},
    })

    assert receipt["prompt_cards"]["selected"]["status"] == "derived"
    assert "no later per-card stage" in receipt["prompt_cards"]["selected"]["note"]
    assert receipt["prompt_cards"]["omitted"]["status"] == "unavailable"
    assert receipt["token_cost"]["status"] == "unavailable"
    assert receipt["token_cost"]["memory_prompt_tokens"] is None
    assert receipt["token_cost"]["total_prompt_tokens"] == 12


def test_cli_uses_latest_organic_run(isolated, capsys):
    wanted = runlog.record(
        source="openai_api", model="m", messages=[{"role": "user", "content": "hi"}],
        response="hello", memory={"mode": "prompt", "cards_applied": ["be brief"],
                                  "applied_ids": ["mem_1"]}, started=1000.0,
    )
    runlog.record(source="replay", model="m", messages=[], response="derived",
                  parent_run_id=wanted, memory={"mode": "prompt", "cards_applied": []},
                  started=1001.0)

    assert cli.main(["memory", "used", "last", "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["run_id"] == wanted
    assert receipt["prompt_cards"]["injected"]["cards"][0]["id"] == "mem_1"
