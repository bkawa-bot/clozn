"""Pure run-to-regression-suite promotion tests."""
from __future__ import annotations

import copy
import json

import pytest

from clozn.testkit import promotion


def _run(run_id="run_one", *, identity=None, messages=None, response="Captured answer."):
    return {
        "id": run_id,
        "messages": messages or [
            {"role": "system", "content": "Be exact."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "response": response,
        "model": "local-model",
        "meta": {"max_tokens": 96, "temperature": 0.0},
        "identity": identity or {
            "model_sha256": "a" * 64,
            "template_fingerprint": "0123456789abcdef",
        },
        "client_key": "client_secret-opaque",
        "session_key": "session_secret-opaque",
        "project_key": "project_secret-opaque",
        "error": None,
    }


def test_create_suite_draft_is_deterministic_runnable_and_does_not_mutate_runs():
    run = _run()
    original = copy.deepcopy(run)

    first = promotion.create_suite_draft("captured", [run])
    second = promotion.create_suite_draft("captured", [copy.deepcopy(run)])

    assert first == second
    assert run == original
    assert first["schema_version"] == promotion.REGRESSION_SUITE_SCHEMA
    assert first["state"] == "draft"
    assert first["cases"] == [{
        "name": "run_one",
        "messages": run["messages"],
        "model": "local-model",
        "max_tokens": 96,
        "sampling": {"temperature": 0.0},
        "expect": {"equals": "Captured answer."},
        "source": {"run_id": "run_one", "sha256": first["cases"][0]["source"]["sha256"]},
    }]
    encoded = json.dumps(first)
    assert "client_secret" not in encoded
    assert "session_secret" not in encoded
    assert "project_secret" not in encoded
    assert "model_sha256" not in encoded


def test_source_digest_is_canonical_across_identity_key_order():
    identity_a = {"model_sha256": "a" * 64, "template_fingerprint": "fingerprint"}
    identity_b = {"template_fingerprint": "fingerprint", "model_sha256": "a" * 64}
    a = promotion.create_suite_draft("s", [_run(identity=identity_a)])
    b = promotion.create_suite_draft("s", [_run(identity=identity_b)])
    assert a["cases"][0]["source"]["sha256"] == b["cases"][0]["source"]["sha256"]


@pytest.mark.parametrize("change", [
    lambda run: run.update(identity={}),
    lambda run: run.update(model=""),
    lambda run: run.update(error="generation failed"),
    lambda run: run["meta"].update(max_tokens=0),
    lambda run: run.update(messages=[{"role": "tool", "content": "result"}]),
    lambda run: run.update(messages=[{"role": "user", "content": "x", "name": "extra"}]),
    lambda run: run.update(messages=[{"role": "user", "content": "x"},
                                     {"role": "assistant", "content": "answer"}]),
])
def test_malformed_or_nonreproducible_source_runs_are_rejected(change):
    run = _run()
    change(run)
    with pytest.raises(promotion.PromotionError):
        promotion.create_suite_draft("s", [run])


def test_duplicate_selected_run_ids_are_rejected():
    with pytest.raises(promotion.PromotionError, match="duplicate selected run id"):
        promotion.create_suite_draft("s", [_run(), _run()])


def test_edit_and_redact_are_copy_on_write_and_keep_source_provenance():
    case = promotion.create_suite_draft("s", [
        _run(messages=[{"role": "user", "content": "Email Ada at ada@example.test"}],
             response="Ada owns ada@example.test")
    ])["cases"][0]
    original = copy.deepcopy(case)

    redacted = promotion.redact_case(
        case, {"ada@example.test": "[EMAIL]", "Ada": "[NAME]"})
    edited = promotion.edit_case(redacted, max_tokens=48, expect={"contains": ["[NAME]"]})

    assert case == original
    assert redacted["messages"][0]["content"] == "Email [NAME] at [EMAIL]"
    assert redacted["expect"]["equals"] == "[NAME] owns [EMAIL]"
    assert edited["max_tokens"] == 48
    assert edited["expect"] == {"contains": ["[NAME]"]}
    assert edited["source"] == case["source"]


def test_redaction_is_order_independent_and_does_not_cascade_replacement_text():
    case = promotion.create_suite_draft("s", [
        _run(messages=[{"role": "user", "content": "Alice Smith / Alice"}])
    ])["cases"][0]
    first = promotion.redact_case(case, {"Alice": "Bob", "Alice Smith": "PERSON"})
    second = promotion.redact_case(case, {"Alice Smith": "PERSON", "Alice": "Bob"})
    assert first == second
    assert first["messages"][0]["content"] == "PERSON / Bob"


def test_whole_suite_redaction_and_sampling_reproducibility_metadata():
    fixed = _run()
    fixed["meta"].update(sampler_mode="sample", seed=7, top_p=0.9)
    unseeded = _run("run_two")
    unseeded["meta"].update(sampler_mode="sample")
    draft = promotion.create_suite_draft("s", [fixed, unseeded])

    assert draft["cases"][0]["sampling"] == {
        "temperature": 0.0, "top_p": 0.9, "seed": 7,
    }
    assert "warnings" not in draft["cases"][0]
    assert "no fixed seed" in draft["cases"][1]["warnings"][0]

    redacted = promotion.redact_suite(draft, {"2 + 2": "[REDACTED]"})
    assert all("[REDACTED]" in case["messages"][1]["content"]
               for case in redacted["cases"])


def test_freeze_and_validation_cover_every_runnable_field_and_suite_membership():
    draft = promotion.create_suite_draft("s", [_run(), _run("run_two")])
    frozen = promotion.freeze_suite(draft)
    assert frozen["state"] == "frozen"
    assert promotion.validate_suite(frozen) == frozen

    mutations = [
        lambda value: value["cases"][0]["messages"][1].update(content="changed"),
        lambda value: value["cases"][0].update(model="changed"),
        lambda value: value["cases"][0].update(max_tokens=1),
        lambda value: value["cases"][0]["expect"].update(equals="changed"),
        lambda value: value["cases"][0]["source"].update(run_id="run_other"),
        lambda value: value["cases"].pop(),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(frozen)
        mutate(changed)
        with pytest.raises(promotion.PromotionError, match="changed after it was frozen"):
            promotion.validate_suite(changed)


def test_frozen_cases_cannot_be_edited_redacted_or_refrozen():
    frozen = promotion.freeze_suite(promotion.create_suite_draft("s", [_run()]))
    case = frozen["cases"][0]
    with pytest.raises(promotion.PromotionError, match="frozen"):
        promotion.edit_case(case, max_tokens=1)
    with pytest.raises(promotion.PromotionError, match="frozen"):
        promotion.redact_case(case, {"answer": "[REDACTED]"})
    with pytest.raises(promotion.PromotionError, match="already frozen"):
        promotion.freeze_suite(frozen)


def test_verify_source_detects_id_identity_input_and_response_drift():
    source = _run()
    case = promotion.freeze_suite(
        promotion.create_suite_draft("s", [source]))["cases"][0]
    assert promotion.verify_source(case, source)
    for change in (
        lambda run: run.update(id="run_other"),
        lambda run: run["identity"].update(template_fingerprint="changed"),
        lambda run: run["messages"][1].update(content="changed"),
        lambda run: run.update(response="changed"),
    ):
        drifted = copy.deepcopy(source)
        change(drifted)
        assert not promotion.verify_source(case, drifted)
