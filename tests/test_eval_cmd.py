"""`clozn eval` argparse wiring -- build the real parser and assert the command is registered with the
right defaults + dispatch fn. No live endpoint (that's cmd_eval's job, exercised manually)."""
from __future__ import annotations

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
