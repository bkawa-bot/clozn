"""Model-free tests for clozn.receipts.intervention_manifest (roadmap Phase 4.2): the vendored
clozn-client validator, canonical-json/sha256, and the replay executor against a fake engine."""
from __future__ import annotations

import os
import sys

import pytest

from clozn.receipts.intervention_manifest import (
    SCHEMA,
    ManifestError,
    canonical_json_object,
    manifest_sha256,
    replay_manifest,
    validate_manifest,
)


def _manifest(**over) -> dict:
    base = {
        "schema": SCHEMA,
        "name": "demo experiment",
        "request": {"prompt": "The sky is", "continuation_ids": [101, 102]},
        "arms": [
            {"name": "cut layer2", "attention_knockout": [
                {"layer": 2, "queries": [3, 4], "keys": [0, 1]},
            ]},
        ],
    }
    base.update(over)
    return base


class FakeEngine:
    """A duck-typed engine: .score(**kwargs) -> dict, .health() -> dict. Records every score() call
    (in order) so tests can assert exact arm-execution sequencing."""

    def __init__(self, *, health=None, scores=None):
        self.calls: list[dict] = []
        self._health = health if health is not None else {"capabilities": {"attn_knockout": True}}
        self._scores = list(scores) if scores is not None else None
        self._next_logprob = -1.0

    def health(self) -> dict:
        return self._health

    def score(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self._scores is not None:
            return self._scores.pop(0)
        value = self._next_logprob
        self._next_logprob -= 0.5
        return {"n_prompt": 2, "n_cont": 1, "tokens": [{"id": 9, "piece": "x", "logprob": value}],
                "sum_logprob": value}


# ------------------------------------------------------------------------------------------ validation

def test_validate_manifest_accepts_the_minimal_valid_shape():
    validated = validate_manifest(_manifest())
    assert validated["schema"] == SCHEMA
    assert validated["name"] == "demo experiment"
    assert len(validated["arms"]) == 1


def test_validate_manifest_rejects_wrong_schema():
    with pytest.raises(ManifestError, match="unsupported manifest schema"):
        validate_manifest(_manifest(schema="something.else.v1"))


def test_validate_manifest_rejects_non_object():
    with pytest.raises(ManifestError, match="must be an object"):
        validate_manifest("not a dict")


def test_validate_manifest_requires_exactly_one_of_prompt_or_prompt_ids():
    with pytest.raises(ManifestError, match="exactly one of prompt or prompt_ids"):
        validate_manifest(_manifest(request={
            "prompt": "hi", "prompt_ids": [1], "continuation_ids": [2],
        }))
    with pytest.raises(ManifestError, match="exactly one of prompt or prompt_ids"):
        validate_manifest(_manifest(request={"continuation_ids": [2]}))


def test_validate_manifest_requires_exactly_one_of_continuation_or_continuation_ids():
    with pytest.raises(ManifestError, match="exactly one of continuation or continuation_ids"):
        validate_manifest(_manifest(request={"prompt": "hi"}))


def test_validate_manifest_rejects_empty_arms():
    with pytest.raises(ManifestError, match="non-empty array"):
        validate_manifest(_manifest(arms=[]))


def test_validate_manifest_rejects_duplicate_arm_names():
    with pytest.raises(ManifestError, match="unique"):
        validate_manifest(_manifest(arms=[
            {"name": "same", "steer_vec": [0.1]},
            {"name": "same", "steer_vec": [0.2]},
        ]))


def test_validate_manifest_arm_requires_knockout_steer_or_steer_vec():
    with pytest.raises(ManifestError, match="must define a knockout, steer, or steer_vec"):
        validate_manifest(_manifest(arms=[{"name": "empty"}]))


def test_validate_manifest_knockout_requires_non_negative_layer():
    with pytest.raises(ManifestError, match="non-negative integer"):
        validate_manifest(_manifest(arms=[
            {"name": "bad", "attention_knockout": [{"layer": -1, "queries": [0], "keys": [0]}]},
        ]))


def test_validate_manifest_knockout_requires_non_empty_queries_and_keys():
    with pytest.raises(ManifestError, match="queries"):
        validate_manifest(_manifest(arms=[
            {"name": "bad", "attention_knockout": [{"layer": 1, "queries": [], "keys": [0]}]},
        ]))


def test_validate_manifest_topk_must_be_a_non_negative_int():
    with pytest.raises(ManifestError, match="topk"):
        validate_manifest(_manifest(request={
            "prompt": "hi", "continuation_ids": [1], "topk": -1,
        }))


def test_validate_manifest_steer_vec_rejects_non_finite():
    with pytest.raises(ManifestError, match="finite"):
        validate_manifest(_manifest(arms=[{"name": "bad", "steer_vec": [float("inf")]}]))


def test_validate_manifest_name_rejects_control_characters():
    with pytest.raises(ManifestError, match="printable"):
        validate_manifest(_manifest(name="bad\x00name"))


def test_validate_manifest_never_mutates_the_input():
    original = _manifest()
    import copy
    before = copy.deepcopy(original)
    validate_manifest(original)
    assert original == before


# ---------------------------------------------------------------------------------- canonical / sha256

def test_manifest_sha256_is_deterministic():
    validated = validate_manifest(_manifest())
    assert manifest_sha256(validated) == manifest_sha256(validate_manifest(_manifest()))
    assert len(manifest_sha256(validated)) == 64


def test_manifest_sha256_changes_with_arm_order():
    two_arms = _manifest(arms=[
        {"name": "a", "steer_vec": [0.1]},
        {"name": "b", "steer_vec": [0.2]},
    ])
    reordered = _manifest(arms=[
        {"name": "b", "steer_vec": [0.2]},
        {"name": "a", "steer_vec": [0.1]},
    ])
    assert manifest_sha256(validate_manifest(two_arms)) != manifest_sha256(validate_manifest(reordered))


def test_manifest_sha256_ignores_key_order_in_the_source_json():
    a = validate_manifest({"schema": SCHEMA, "name": "n", "request": {"prompt": "p", "continuation_ids": [1]},
                           "arms": [{"name": "arm", "steer_vec": [0.1]}]})
    b = validate_manifest({"arms": [{"steer_vec": [0.1], "name": "arm"}],
                           "request": {"continuation_ids": [1], "prompt": "p"},
                           "name": "n", "schema": SCHEMA})
    assert manifest_sha256(a) == manifest_sha256(b)


def test_canonical_json_object_omits_empty_expected_health_and_metadata():
    validated = validate_manifest(_manifest())
    obj = canonical_json_object(validated)
    assert "expected_health" not in obj and "metadata" not in obj


def test_canonical_json_object_keeps_non_empty_expected_health_and_metadata():
    validated = validate_manifest(_manifest(expected_health={"n_layer": 24}, metadata={"purpose": "x"}))
    obj = canonical_json_object(validated)
    assert obj["expected_health"] == {"n_layer": 24}
    assert obj["metadata"] == {"purpose": "x"}


def test_manifest_sha256_matches_clozn_client_reference_implementation():
    """Cross-check against the REAL clozn-client library (imported only in this test, never in the
    server module) -- proves a manifest built with the pip package and replayed here compares exactly
    across sessions, per this roadmap item's requirement."""
    client_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "clozn-client", "src")
    if client_src not in sys.path:
        sys.path.insert(0, client_src)
    clozn_client_manifests = pytest.importorskip("clozn_client.manifests")
    clozn_client_models = pytest.importorskip("clozn_client.models")

    manifest = clozn_client_manifests.InterventionManifest(
        name="cross check",
        request=clozn_client_manifests.ScoreRequest(prompt="The sky is", continuation_ids=(101, 102)),
        arms=(
            clozn_client_manifests.InterventionArm(
                name="cut", attention_knockout=(
                    clozn_client_models.AttentionKnockout(layer=2, queries=(3, 4), keys=(0, 1)),
                ),
            ),
            clozn_client_manifests.InterventionArm(name="steer", steer={"concept": "calm", "coef": 1.5}),
            clozn_client_manifests.InterventionArm(name="raw", steer_vec=(0.1, 0.2, 0.3)),
        ),
        expected_health={"n_layer": 24},
        metadata={"purpose": "demo"},
    )
    validated = validate_manifest(manifest.to_json_object())
    assert manifest_sha256(validated) == manifest.sha256
    assert canonical_json_object(validated) == manifest.to_json_object()


# --------------------------------------------------------------------------------------- replay: happy path

def test_replay_manifest_executes_baseline_then_arms_in_manifest_order():
    engine = FakeEngine()
    manifest = _manifest(arms=[
        {"name": "first", "steer_vec": [0.1]},
        {"name": "second", "steer_vec": [0.2]},
        {"name": "third", "attention_knockout": [{"layer": 0, "queries": [0], "keys": [0]}]},
    ])
    out = replay_manifest(manifest, engine, health=engine.health())
    assert out["performed"] is True
    assert [arm["name"] for arm in out["arms"]] == ["first", "second", "third"]
    # baseline call has neither steer nor attn_knockout; exactly 1 + len(arms) calls total.
    assert len(engine.calls) == 4
    assert "steer_vec" not in engine.calls[0] and "attn_knockout" not in engine.calls[0]
    assert engine.calls[1]["steer_vec"] == [0.1]
    assert engine.calls[2]["steer_vec"] == [0.2]
    assert engine.calls[3]["attn_knockout"] == [{"layer": 0, "queries": [0], "keys": [0], "renormalize": True}]


def test_replay_manifest_reports_manifest_sha256_and_name():
    engine = FakeEngine()
    manifest = _manifest()
    out = replay_manifest(manifest, engine, health=engine.health())
    assert out["manifest_sha256"] == manifest_sha256(validate_manifest(manifest))
    assert out["manifest_name"] == "demo experiment"


def test_replay_manifest_support_drop_is_signed_baseline_minus_arm():
    engine = FakeEngine(scores=[
        {"n_prompt": 2, "n_cont": 1, "tokens": [], "sum_logprob": -1.0},   # baseline
        {"n_prompt": 2, "n_cont": 1, "tokens": [], "sum_logprob": -3.0},   # arm: hurt support
    ])
    manifest = _manifest(arms=[{"name": "cut", "attention_knockout": [
        {"layer": 0, "queries": [0], "keys": [0]},
    ]}])
    out = replay_manifest(manifest, engine, health=engine.health())
    # baseline(-1.0) - arm(-3.0) = +2.0: positive support_drop means the intervention hurt the reply.
    assert out["arms"][0]["support_drop"] == pytest.approx(2.0)


def test_replay_manifest_labels_re_prefilled_replay_class():
    engine = FakeEngine()
    out = replay_manifest(_manifest(), engine, health=engine.health())
    assert out["replay_class"] == "re_prefilled"
    assert all(arm["replay_class"] == "re_prefilled" for arm in out["arms"])


def test_replay_manifest_identity_block_reads_from_health():
    health = {
        "capabilities": {"attn_knockout": True}, "model": "qwen-test", "model_sha256": "abc123",
        "architecture": "qwen2", "n_layer": 24, "n_embd": 896, "protocol_version": "1.0",
        "mode": "autoregressive",
    }
    engine = FakeEngine(health=health)
    out = replay_manifest(_manifest(), engine, health=health)
    assert out["identity"] == {
        "model": "qwen-test", "model_sha256": "abc123", "architecture": "qwen2",
        "n_layer": 24, "n_embd": 896, "protocol_version": "1.0", "mode": "autoregressive",
    }


def test_replay_manifest_steer_only_arm_needs_no_capability():
    """A manifest with ONLY steer/steer_vec arms (no attention_knockout) must run even when health
    reports attn_knockout=false -- the capability gate is per-requirement, not blanket."""
    engine = FakeEngine(health={"capabilities": {"attn_knockout": False}})
    manifest = _manifest(arms=[{"name": "steer only", "steer_vec": [0.1, 0.2]}])
    out = replay_manifest(manifest, engine, health=engine.health())
    assert out["performed"] is True
    assert out["capabilities"] == []


def test_replay_manifest_timing_is_measured_and_score_calls_counted():
    engine = FakeEngine()
    out = replay_manifest(_manifest(), engine, health=engine.health())
    assert out["timing"]["score_calls"] == 2  # baseline + 1 arm
    assert out["timing"]["total_ms"] >= 0.0
    assert out["timing"]["baseline_ms"] >= 0.0


# --------------------------------------------------------------------------------- replay: capability refusal

def test_replay_manifest_refuses_when_attn_knockout_capability_missing():
    engine = FakeEngine(health={"capabilities": {"attn_knockout": False}})
    manifest = _manifest()  # has an attention_knockout arm
    out = replay_manifest(manifest, engine, health=engine.health())
    assert out["performed"] is False
    assert out["error"]["code"] == "capability_unavailable"
    assert "attn_knockout" in out["error"]["message"]
    assert out["capabilities"] == [
        {"capability": "attn_knockout", "available": False,
         "reason": "capabilities.attn_knockout=true is required (engine health: False)"},
    ]
    # the engine must NEVER be called when a required capability is missing.
    assert engine.calls == []


def test_replay_manifest_refuses_when_health_has_no_capabilities_block_at_all():
    engine = FakeEngine(health={})
    out = replay_manifest(_manifest(), engine, health={})
    assert out["performed"] is False
    assert engine.calls == []


def test_replay_manifest_still_reports_manifest_sha256_on_refusal():
    engine = FakeEngine(health={"capabilities": {"attn_knockout": False}})
    manifest = _manifest()
    out = replay_manifest(manifest, engine, health=engine.health())
    assert out["manifest_sha256"] == manifest_sha256(validate_manifest(manifest))


# ------------------------------------------------------------------------------------------ raises through

def test_replay_manifest_raises_manifest_error_for_invalid_manifest_before_touching_the_engine():
    engine = FakeEngine()
    with pytest.raises(ManifestError):
        replay_manifest(_manifest(schema="bogus"), engine, health=engine.health())
    assert engine.calls == []
