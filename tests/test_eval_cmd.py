"""`clozn eval` argparse wiring -- build the real parser and assert the command is registered with the
right defaults + dispatch fn. No live endpoint (that's cmd_eval's job, exercised manually)."""
from __future__ import annotations

import builtins

from clozn.eval import bench, store as eval_store
from clozn.cli.main import build_parser
from clozn.cli.commands.eval import cmd_eval


def _subparser_choices(p):
    for a in p._actions:
        if getattr(a, "choices", None) and "eval" in a.choices:
            return a.choices
    return {}


def test_eval_is_registered():
    assert "eval" in _subparser_choices(build_parser())


def test_eval_defaults_and_dispatch():
    ns = build_parser().parse_args(["eval"])
    assert ns.which == "arith" and ns.score == "min" and ns.target_error == 0.05
    assert ns.url.endswith(":8080") and ns.fn is cmd_eval


def test_eval_accepts_set_score_and_target_error():
    ns = build_parser().parse_args(["eval", "--set", "all", "--score", "mean",
                                    "--target-error", "0.1", "--json"])
    assert ns.which == "all" and ns.score == "mean" and ns.target_error == 0.1 and ns.json is True


def test_eval_accepts_task_and_wizard():
    ns = build_parser().parse_args(["eval", "--wizard", "--task", "customer-support"])
    assert ns.wizard is True and ns.task == "customer-support"


def test_list_profiles_is_model_free_and_prints_task_provenance(monkeypatch, capsys):
    monkeypatch.setattr(
        eval_store,
        "list_profiles",
        lambda: [{"model": "model-a", "task": "retrieval qa", "set": "extended", "n": 42,
                  "score": "min", "policy": {"answer_at": 0.8, "ask_at": 0.5}}],
    )
    monkeypatch.setattr(
        bench,
        "bench",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("listing must not run probes")),
    )

    ns = build_parser().parse_args(["eval", "--list-profiles"])
    assert cmd_eval(ns) == 0
    text = capsys.readouterr().out
    assert "model-a" in text and "task=retrieval qa" in text
    assert "not a live fact-check" in text


def test_invalid_task_and_target_error_stop_before_model_work(monkeypatch, capsys):
    def must_not_bench(*_args, **_kwargs):
        raise AssertionError("validation must happen before bench/model work")

    monkeypatch.setattr(bench, "bench", must_not_bench)
    for argv in (["eval", "--task", "bad\ntask"],
                 ["eval", "--task", "x" * 81],
                 ["eval", "--target-error", "-0.01"],
                 ["eval", "--target-error", "nan"]):
        ns = build_parser().parse_args(argv)
        assert cmd_eval(ns) == 2
    text = capsys.readouterr().out
    assert "task must be" in text
    assert "target error must be" in text


def _fake_out(model="model-sha-123"):
    return {
        "n": 2,
        "unmatched": 1,
        "model": model,
        "pairs": [(0.92, True), (0.31, False)],
        "rows": [],
        "report": {"available": True},
    }


def test_wizard_guides_choices_saves_task_profile_and_prints_claim_limits(monkeypatch, capsys):
    answers = iter(["coding", "extended", "mean", "0.1", "yes"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))
    called = {}
    monkeypatch.setattr(bench, "bench", lambda url, which, score:
                        called.update(url=url, which=which, score=score) or _fake_out())
    monkeypatch.setattr(bench, "_print", lambda *_args: None)

    def save_profile(payload, *, task):
        called.update(payload=payload, task=task)
        return {"profile_path": "/profiles/model/coding.json", "active_path": "/eval_report.json"}

    monkeypatch.setattr(eval_store, "save_profile", save_profile, raising=False)
    ns = build_parser().parse_args(["eval", "--wizard"])
    assert cmd_eval(ns) == 0

    assert called["which"] == "extended" and called["score"] == "mean"
    assert called["task"] == "coding"
    assert called["payload"]["model"] == "model-sha-123"
    assert called["payload"]["target_error"] == 0.1
    text = capsys.readouterr().out
    assert "Calibration plan: task=coding" in text
    assert "2 gradeable sample(s), 1 unmatched" in text
    assert "model=model-sha-123  task=coding" in text
    assert "Policy tradeoff" in text and "correct_withheld=" in text
    assert "Distribution limit" in text
    assert "recorded token probabilities" in text
    assert "not a live fact-check" in text
    assert "active report -> /eval_report.json" in text


def test_wizard_eof_uses_cli_defaults_and_does_not_save_unasked(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    called = {}
    monkeypatch.setattr(bench, "bench", lambda url, which, score:
                        called.update(which=which, score=score) or _fake_out())
    monkeypatch.setattr(bench, "_print", lambda *_args: None)
    monkeypatch.setattr(eval_store, "save_profile",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected save")),
                        raising=False)

    ns = build_parser().parse_args(["eval", "--wizard", "--task", "support", "--set", "hard",
                                    "--score", "mean", "--target-error", "0.2"])
    assert cmd_eval(ns) == 0
    assert called == {"which": "hard", "score": "mean"}
    assert "task=support  set=hard  score=mean  target_error=0.2  save=no" in capsys.readouterr().out


def test_nonwizard_save_uses_task_aware_store_and_keeps_report_shape(monkeypatch):
    monkeypatch.setattr(bench, "bench", lambda *_args: _fake_out())
    monkeypatch.setattr(bench, "_print", lambda *_args: None)
    saved = {}
    monkeypatch.setattr(eval_store, "save_profile",
                        lambda payload, *, task: saved.update(payload=payload, task=task) or "profile.json",
                        raising=False)
    ns = build_parser().parse_args(["eval", "--save", "--task", "retrieval"])
    assert cmd_eval(ns) == 0
    assert saved["task"] == "retrieval" and saved["payload"]["task"] == "retrieval"
    for key in ("set", "score", "target_error", "model", "n", "unmatched", "report", "policy", "rows"):
        assert key in saved["payload"]


def test_save_refuses_to_create_a_cross_model_profile_without_active_model(monkeypatch, capsys):
    monkeypatch.setattr(bench, "bench", lambda *_args: _fake_out(model=None))
    monkeypatch.setattr(bench, "_print", lambda *_args: None)
    monkeypatch.setattr(eval_store, "save_profile",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected save")),
                        raising=False)
    ns = build_parser().parse_args(["eval", "--save", "--task", "Customer Support"])
    assert cmd_eval(ns) == 1
    assert "no active model identity" in capsys.readouterr().out
