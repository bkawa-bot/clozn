"""test_diff_model -- clozn/cli/commands/diff_model.py (`clozn diff-model`, Phase-1 §4.1): base-vs-
fine-tune/merge per-token behavior receipts, generalizing `clozn quant-check`.

Model-free / GPU-free throughout, mirroring tests/test_quant_check.py's own discipline: no real engine,
no C++ server, no GPU. `check_tokenizer_compat`/`check_template_match` are exercised against small fake
engines exposing only `.score`/`.apply_template`; `classify_verdict` is exercised against FIXTURE
receipts built with the real, already-tested `quant_receipts.diff_quant_scores` (no new fixture shape
invented -- mirrors test_quant_check.py's own `_receipt`/`_skip` helpers); `run_direction`/
`run_diff_model` are exercised against a fake engine exposing `.apply_template`/`.score`/`.complete`
(mirrors quant_check's own `FakeEngine`); `add_subparser`'s argparse wiring is exercised on a throwaway
parser, never touching clozn/cli/main.py.

DEFERRED (not covered here, by design): `cmd_diff_model` itself -- the real two-engine boot -- needs a
live engine/GPU (see diff_model.py's module docstring). Everything upstream of "actually boot a process
and open a socket" is covered.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.cli.commands.diff_model as dm  # noqa: E402
import clozn.cli.commands.quant_check as qc  # noqa: E402
import clozn.receipts.quant_receipts as qr  # noqa: E402
from clozn.cli.main import CloznError  # noqa: E402


def _tok(id_, piece, logprob, topk=None):
    t = {"id": id_, "piece": piece, "logprob": logprob}
    if topk is not None:
        t["topk"] = topk
    return t


# ==================================================================================== fake engines

class TokenizeEngine:
    """Minimal fake for tokenizer-preflight tests: `.score()` tokenizes `continuation` deterministically
    via a caller-supplied function, ignoring `prompt` entirely (the preflight's whole point is comparing
    tokenization, not templating)."""

    def __init__(self, tokenize_fn):
        self.tokenize_fn = tokenize_fn
        self.score_calls = []

    def score(self, prompt=None, **kw):
        self.score_calls.append({"prompt": prompt, **kw})
        return {"tokens": self.tokenize_fn(kw.get("continuation", ""))}


def _word_tokens(text):
    return [{"id": 1000 + i, "piece": w} for i, w in enumerate(text.split())]


class SimpleFakeEngine:
    """A minimal fake EngineClient covering apply_template/score/complete -- enough to drive the whole
    diff-model pipeline (tokenizer preflight, template check, fresh-run generation, teacher-forced
    scoring) without a real engine. `score()` always echoes back whatever ids it's asked to
    teacher-force (continuation_ids primary, else a deterministic per-word tokenization of continuation
    text -- the SAME scheme `_word_tokens` uses, so two instances of this class always tokenize
    identically unless one overrides `_tokenize_text`), with a per-instance logprob and an optional
    per-position argmax override so tests can dial in flips/dependence-shifts precisely."""

    def __init__(self, *, template="TPL", completion_text="one two three four five", base_logprob=-0.1,
                argmax_offset=0):
        self.template = template
        self.completion_text = completion_text
        self.base_logprob = base_logprob
        self.argmax_offset = argmax_offset   # 0 -> argmax always matches the forced id (no flips)
        self.apply_template_calls = []
        self.score_calls = []
        self.complete_calls = []

    def apply_template(self, messages):
        self.apply_template_calls.append(list(messages))
        return self.template

    def complete(self, prompt, **params):
        self.complete_calls.append({"prompt": prompt, **params})
        return {"choices": [{"text": self.completion_text}]}

    def _tokenize_text(self, text):
        return [(1000 + i, w) for i, w in enumerate(text.split())]

    def score(self, prompt=None, **kw):
        self.score_calls.append({"prompt": prompt, **kw})
        topk = int(kw.get("topk", 0) or 0)
        if kw.get("continuation_ids") is not None:
            ids = [int(t) for t in kw["continuation_ids"]]
            pieces = [f"t{i}" for i in ids]
        elif kw.get("continuation") is not None:
            pairs = self._tokenize_text(kw["continuation"])
            ids = [p[0] for p in pairs]
            pieces = [p[1] for p in pairs]
        else:
            ids, pieces = [], []
        tokens = []
        for tid, piece in zip(ids, pieces):
            entry = {"id": tid, "piece": piece, "logprob": self.base_logprob}
            if topk > 0:
                argmax_id = tid + self.argmax_offset
                if argmax_id == tid:
                    entry["topk"] = [{"id": tid, "piece": piece, "logprob": self.base_logprob}]
                else:
                    entry["topk"] = [{"id": argmax_id, "piece": f"{piece}_ALT", "logprob": self.base_logprob},
                                     {"id": tid, "piece": piece, "logprob": self.base_logprob - 1.0}]
            tokens.append(entry)
        return {"tokens": tokens}


def _args(**overrides):
    base = dict(runs=2, from_log=False, topk=8, max_tokens=50, both=False, own_templates=False)
    base.update(overrides)
    return argparse.Namespace(**base)


# ==================================================================================== check_tokenizer_compat

def test_check_tokenizer_compat_identical_is_compatible():
    sub_a = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    sub_b = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    out = dm.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is True
    assert len(out["probes"]) == len(dm._TOKENIZER_PROBES)
    for p in out["probes"]:
        assert p["ids_match"] is True
        assert p["pieces_match"] is True
        assert p["n_a"] > 0 and p["n_b"] > 0


def test_check_tokenizer_compat_different_ids_is_incompatible():
    def _shifted(text):
        return [{"id": 2000 + i, "piece": w} for i, w in enumerate(text.split())]

    sub_a = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    sub_b = qc._EngineScoreSub(TokenizeEngine(_shifted))
    out = dm.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is False
    assert any(not p["ids_match"] for p in out["probes"])


def test_check_tokenizer_compat_same_ids_different_pieces_is_incompatible():
    def _relabelled(text):
        return [{"id": 1000 + i, "piece": w + "_X"} for i, w in enumerate(text.split())]

    sub_a = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    sub_b = qc._EngineScoreSub(TokenizeEngine(_relabelled))
    out = dm.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is False
    assert all(p["ids_match"] for p in out["probes"])          # ids agree...
    assert all(not p["pieces_match"] for p in out["probes"])   # ...but pieces don't


def test_check_tokenizer_compat_never_raises_when_engine_blows_up():
    class BoomEngine:
        def score(self, prompt=None, **kw):
            raise RuntimeError("boom")

    sub_a = qc._EngineScoreSub(BoomEngine())
    sub_b = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    out = dm.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is False
    assert all(not p["ids_match"] for p in out["probes"])


def test_tokenizer_refusal_message_names_the_failed_probes_and_suggests_same_family():
    out = {"compatible": False, "probes": [{"probe": "code_snippet", "ids_match": False, "pieces_match": False}]}
    msg = dm._tokenizer_refusal_message(out)
    assert "code_snippet" in msg
    assert "meaningless" in msg
    assert "same-tokenizer-family" in msg or "same tokenizer" in msg.lower()


# ==================================================================================== check_template_match

def test_check_template_match_identical():
    sub_a = qc._EngineScoreSub(SimpleFakeEngine(template="SAME"))
    sub_b = qc._EngineScoreSub(SimpleFakeEngine(template="SAME"))
    out = dm.check_template_match(sub_a, sub_b)
    assert out["match"] is True
    assert out["rendering_a"] == out["rendering_b"] == "SAME"


def test_check_template_match_different():
    sub_a = qc._EngineScoreSub(SimpleFakeEngine(template="A_TEMPLATE"))
    sub_b = qc._EngineScoreSub(SimpleFakeEngine(template="B_TEMPLATE"))
    out = dm.check_template_match(sub_a, sub_b)
    assert out["match"] is False
    assert out["rendering_a"] == "A_TEMPLATE"
    assert out["rendering_b"] == "B_TEMPLATE"


def test_check_template_match_never_raises_when_apply_template_blows_up():
    class BoomEngine:
        def apply_template(self, messages):
            raise RuntimeError("no embedded template")

    sub_a = qc._EngineScoreSub(BoomEngine())
    sub_b = qc._EngineScoreSub(SimpleFakeEngine(template="X"))
    out = dm.check_template_match(sub_a, sub_b)
    assert out["match"] is False
    assert out["rendering_a"] is None


# ==================================================================================== _EngineScoreSub.template_engine

def test_engine_score_sub_defaults_template_engine_to_self():
    eng = SimpleFakeEngine(template="OWN")
    sub = qc._EngineScoreSub(eng)
    assert sub.template_engine is eng


def test_engine_score_sub_template_engine_renders_for_a_different_engines_score_call():
    eng_a = SimpleFakeEngine(template="A_RENDER")
    eng_b = SimpleFakeEngine(template="B_RENDER")
    sub_b = qc._EngineScoreSub(eng_b, template_engine=eng_a)

    sub_b.score_tokens([{"role": "user", "content": "hi"}], [1], topk=0)

    assert eng_a.apply_template_calls    # A rendered the prompt
    assert not eng_b.apply_template_calls   # B never rendered
    assert eng_b.score_calls[-1]["prompt"] == "A_RENDER"   # B's engine.score received A's rendering


# ==================================================================================== classify_verdict

def _receipt_n_tokens(n, *, delta=0.0, flip_at=None, run_id="run_x", category="cat"):
    """A verified receipt over `n` tokens, each with a constant per-position |delta_nats| == `delta`
    (so mean_abs_delta_nats_all == delta exactly), with an argmax flip injected at index `flip_at`
    (None -> no flips at all). Built via the real, already-tested `quant_receipts.diff_quant_scores` on
    fixture arrays -- mirrors test_quant_check.py's own `_receipt` helper."""
    answer = list(range(n))
    tokens_a = [_tok(i, f"a{i}", -0.1, [_tok(i, f"a{i}", -0.1)]) for i in range(n)]
    tokens_b = []
    for i in range(n):
        lp_b = -0.1 - delta
        if flip_at is not None and i == flip_at:
            topk_b = [_tok(90000 + i, f"z{i}", lp_b + 0.05), _tok(i, f"a{i}", lp_b)]
        else:
            topk_b = [_tok(i, f"a{i}", lp_b)]
        tokens_b.append(_tok(i, f"a{i}", lp_b, topk_b))
    r = qr.diff_quant_scores(answer, tokens_a, tokens_b, label_a="A", label_b="B")
    r["run_id"] = run_id
    r["category"] = category
    return r


def test_classify_verdict_no_detectable_diff():
    r = _receipt_n_tokens(100, delta=0.001, category="factual_qa")
    agg = qc.aggregate_receipts([r], label_a="A", label_b="B")
    v = dm.classify_verdict([r], agg)
    assert v["verdict"] == "NO_DETECTABLE_DIFF"
    assert v["total_tokens"] == 100
    assert v["total_flipped"] == 0
    assert v["mean_abs_delta_nats_all_mean"] == pytest.approx(0.001)
    assert "silent no-op" in v["message"]
    assert v["is_heuristic"] is True


def test_classify_verdict_changed_with_flip_and_per_category_counts():
    r1 = _receipt_n_tokens(60, delta=0.001, flip_at=0, run_id="r1", category="code")
    r2 = _receipt_n_tokens(60, delta=0.001, flip_at=5, run_id="r2", category="reasoning")
    agg = qc.aggregate_receipts([r1, r2], label_a="A", label_b="B")
    v = dm.classify_verdict([r1, r2], agg)
    assert v["verdict"] == "CHANGED"
    assert v["total_tokens"] == 120
    assert v["total_flipped"] == 2
    assert v["per_category_flips"] == {"code": 1, "reasoning": 1}


def test_classify_verdict_changed_zero_flips_but_delta_at_threshold_boundary():
    # exactly AT the threshold: "< 0.02" must fail (not "<="), so this falls through to CHANGED even
    # with zero flips -- a real, flip-free confidence shift big enough not to wave through.
    r = _receipt_n_tokens(100, delta=dm._VERDICT_MAX_MEAN_ABS_DELTA_NATS, category="x")
    agg = qc.aggregate_receipts([r], label_a="A", label_b="B")
    v = dm.classify_verdict([r], agg)
    assert v["verdict"] == "CHANGED"
    assert v["total_flipped"] == 0


def test_classify_verdict_insufficient_sample_below_token_floor():
    r = _receipt_n_tokens(50, delta=0.001, category="x")
    agg = qc.aggregate_receipts([r], label_a="A", label_b="B")
    v = dm.classify_verdict([r], agg)
    assert v["verdict"] == "INSUFFICIENT_SAMPLE"
    assert "too small" in v["message"]


def test_classify_verdict_token_floor_boundary_99_vs_100():
    r99 = _receipt_n_tokens(99, delta=0.001, category="x")
    agg99 = qc.aggregate_receipts([r99], label_a="A", label_b="B")
    v99 = dm.classify_verdict([r99], agg99)
    assert v99["verdict"] == "INSUFFICIENT_SAMPLE"

    r100 = _receipt_n_tokens(100, delta=0.001, category="x")
    agg100 = qc.aggregate_receipts([r100], label_a="A", label_b="B")
    v100 = dm.classify_verdict([r100], agg100)
    assert v100["verdict"] == "NO_DETECTABLE_DIFF"


def test_classify_verdict_insufficient_sample_takes_priority_even_with_flips():
    r = _receipt_n_tokens(40, delta=1.0, flip_at=0, category="x")
    agg = qc.aggregate_receipts([r], label_a="A", label_b="B")
    v = dm.classify_verdict([r], agg)
    assert v["verdict"] == "INSUFFICIENT_SAMPLE"


def test_classify_verdict_ignores_unverified_receipts():
    good = _receipt_n_tokens(100, delta=0.001, category="x")
    skipped = {"causal_verified": False, "run_id": "bad", "category": "x", "note": "no ids"}
    agg = qc.aggregate_receipts([good, skipped], label_a="A", label_b="B")
    v = dm.classify_verdict([good, skipped], agg)
    assert v["verdict"] == "NO_DETECTABLE_DIFF"
    assert v["total_tokens"] == 100


def test_format_verdict_renders_thresholds_and_message():
    r = _receipt_n_tokens(100, delta=0.001, category="x")
    agg = qc.aggregate_receipts([r], label_a="A", label_b="B")
    v = dm.classify_verdict([r], agg)
    out = dm.format_verdict(v)
    assert "verdict: NO_DETECTABLE_DIFF" in out
    assert "heuristic screen" in out
    assert "thresholds:" in out
    assert "silent no-op" in out


def test_format_verdict_renders_per_category_flips_for_changed():
    r1 = _receipt_n_tokens(60, delta=0.001, flip_at=0, run_id="r1", category="code")
    r2 = _receipt_n_tokens(60, delta=0.001, flip_at=5, run_id="r2", category="reasoning")
    agg = qc.aggregate_receipts([r1, r2], label_a="A", label_b="B")
    v = dm.classify_verdict([r1, r2], agg)
    out = dm.format_verdict(v)
    assert "per-category argmax flips" in out
    assert "code: 1" in out
    assert "reasoning: 1" in out


# ==================================================================================== run_diff_model wiring

def test_run_diff_model_refuses_on_incompatible_tokenizer():
    def _shifted(text):
        return [{"id": 2000 + i, "piece": w} for i, w in enumerate(text.split())]

    eng_a = TokenizeEngine(_word_tokens)
    eng_b = TokenizeEngine(_shifted)
    with pytest.raises(CloznError) as exc:
        dm.run_diff_model(eng_a, eng_b, _args(), label_a="ref", label_b="cand")
    msg = str(exc.value)
    assert "tokeniz" in msg.lower()
    assert "meaningless" in msg.lower()


def test_run_diff_model_happy_path_reference_anchored_only():
    eng_a = SimpleFakeEngine(template="T")
    eng_b = SimpleFakeEngine(template="T")   # identical templates -- no override needed
    result = dm.run_diff_model(eng_a, eng_b, _args(runs=2), label_a="ref", label_b="cand")
    assert result["tokenizer_compat"]["compatible"] is True
    assert result["template_match"] is True
    assert result["template_policy"] == "reference"
    assert "reference_anchored" in result
    assert "candidate_anchored" not in result
    assert result["reference_anchored"]["agg"]["total_tokens"] > 0
    assert result["reference_anchored"]["verdict"]["verdict"] in ("NO_DETECTABLE_DIFF", "CHANGED",
                                                                  "INSUFFICIENT_SAMPLE")


def test_run_diff_model_default_policy_renders_candidate_under_reference_template_when_they_differ():
    eng_a = SimpleFakeEngine(template="A_RENDER")
    eng_b = SimpleFakeEngine(template="B_RENDER")
    result = dm.run_diff_model(eng_a, eng_b, _args(runs=1), label_a="ref", label_b="cand")
    assert result["template_match"] is False
    assert result["template_policy"] == "reference"
    assert result["template_caveat"] is not None
    assert "WEIGHTS" in result["template_caveat"]
    # B's engine.score calls (teacher-forcing) should have received A's rendering as the prompt
    b_score_prompts = [c.get("prompt") for c in eng_b.score_calls if c.get("continuation_ids") is not None]
    assert b_score_prompts and all(p == "A_RENDER" for p in b_score_prompts)
    # B's apply_template was called exactly once -- by check_template_match's own preflight probe, never
    # by the ladder itself (which rendered under A via template_engine instead).
    assert len(eng_b.apply_template_calls) == 1


def test_run_diff_model_own_templates_flag_keeps_each_engines_own_rendering():
    eng_a = SimpleFakeEngine(template="A_RENDER")
    eng_b = SimpleFakeEngine(template="B_RENDER")
    result = dm.run_diff_model(eng_a, eng_b, _args(runs=1, own_templates=True), label_a="ref", label_b="cand")
    assert result["template_policy"] == "own"
    assert result["template_caveat"] is not None
    assert "DEPLOYED" in result["template_caveat"]
    b_score_prompts = [c.get("prompt") for c in eng_b.score_calls if c.get("continuation_ids") is not None]
    assert b_score_prompts and all(p == "B_RENDER" for p in b_score_prompts)
    assert eng_b.apply_template_calls   # B rendered its own template this time


def test_run_diff_model_own_templates_no_caveat_when_templates_already_match():
    eng_a = SimpleFakeEngine(template="SAME")
    eng_b = SimpleFakeEngine(template="SAME")
    result = dm.run_diff_model(eng_a, eng_b, _args(runs=1, own_templates=True), label_a="ref", label_b="cand")
    assert result["template_policy"] == "own"
    assert result["template_match"] is True
    assert result["template_caveat"] is None


# ==================================================================================== --both wiring

def test_both_flag_calls_run_direction_twice_with_swapped_generation_role(monkeypatch):
    calls = []

    def fake_run_direction(sub_gen, sub_a, sub_b, *, label_a, label_b, args):
        calls.append(sub_gen)
        agg = qc.aggregate_receipts([], label_a=label_a, label_b=label_b)
        return [], agg

    monkeypatch.setattr(dm, "run_direction", fake_run_direction)

    eng_a = SimpleFakeEngine(template="T")
    eng_b = SimpleFakeEngine(template="T")
    result = dm.run_diff_model(eng_a, eng_b, _args(both=True), label_a="ref", label_b="cand")

    assert len(calls) == 2
    assert calls[0].engine is eng_a   # reference-anchored: generate under the reference
    assert calls[1].engine is eng_b   # candidate-anchored: generate under the candidate
    assert "reference_anchored" in result and "candidate_anchored" in result


def test_both_flag_false_calls_run_direction_once(monkeypatch):
    calls = []

    def fake_run_direction(sub_gen, sub_a, sub_b, *, label_a, label_b, args):
        calls.append(sub_gen)
        agg = qc.aggregate_receipts([], label_a=label_a, label_b=label_b)
        return [], agg

    monkeypatch.setattr(dm, "run_direction", fake_run_direction)

    eng_a = SimpleFakeEngine(template="T")
    eng_b = SimpleFakeEngine(template="T")
    result = dm.run_diff_model(eng_a, eng_b, _args(both=False), label_a="ref", label_b="cand")

    assert len(calls) == 1
    assert "candidate_anchored" not in result


# ==================================================================================== run_direction

def test_run_direction_generates_under_sub_gen_and_diffs_under_sub_a_sub_b():
    eng_a = SimpleFakeEngine(template="T", completion_text="alpha beta gamma")
    eng_b = SimpleFakeEngine(template="T")
    sub_a = qc._EngineScoreSub(eng_a)
    sub_b = qc._EngineScoreSub(eng_b)
    receipts, agg = dm.run_direction(sub_a, sub_a, sub_b, label_a="A", label_b="B", args=_args(runs=1))
    assert len(receipts) == 1
    assert eng_a.complete_calls   # generation happened under sub_gen == sub_a
    assert not eng_b.complete_calls
    assert agg["label_a"] == "A" and agg["label_b"] == "B"


# ==================================================================================== format_diff_model_report

def test_format_diff_model_report_headline_and_direction_labels():
    eng_a = SimpleFakeEngine(template="T")
    eng_b = SimpleFakeEngine(template="T")
    result = dm.run_diff_model(eng_a, eng_b, _args(runs=1, both=True), label_a="Base", label_b="Tuned")
    out = dm.format_diff_model_report(result)
    assert "diff-model: Base vs Tuned" in out
    assert "quant-check:" not in out
    assert "reference-anchored" in out
    assert "candidate-anchored" in out
    assert "forgetting/no-op view" in out
    assert "target-gain view" in out
    assert "verdict:" in out


def test_format_diff_model_report_no_both_omits_candidate_anchored():
    eng_a = SimpleFakeEngine(template="T")
    eng_b = SimpleFakeEngine(template="T")
    result = dm.run_diff_model(eng_a, eng_b, _args(runs=1, both=False), label_a="Base", label_b="Tuned")
    out = dm.format_diff_model_report(result)
    assert "candidate-anchored" not in out


# ==================================================================================== add_subparser / argparse

def _build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    dm.add_subparser(sub)
    return p


def test_add_subparser_defaults():
    p = _build_parser()
    args = p.parse_args(["diff-model", "base.gguf", "tuned.gguf"])
    assert args.reference == "base.gguf"
    assert args.candidate == "tuned.gguf"
    assert args.runs == 8
    assert args.from_log is False
    assert args.topk == 8
    assert args.max_tokens == 200
    assert args.port_a == 0
    assert args.port_b == 0
    assert args.cpu is False
    assert args.json is False
    assert args.both is False
    assert args.own_templates is False
    assert args.fn is dm.cmd_diff_model


def test_add_subparser_parses_overrides():
    p = _build_parser()
    args = p.parse_args(["diff-model", "base.gguf", "tuned.gguf", "--runs", "20", "--from-log",
                        "--topk", "16", "--max-tokens", "64", "--port-a", "9001", "--port-b", "9002",
                        "--cpu", "--json", "--both", "--own-templates"])
    assert args.runs == 20
    assert args.from_log is True
    assert args.topk == 16
    assert args.max_tokens == 64
    assert args.port_a == 9001
    assert args.port_b == 9002
    assert args.cpu is True
    assert args.json is True
    assert args.both is True
    assert args.own_templates is True


def test_add_subparser_requires_both_models():
    import contextlib
    import io
    p = _build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            p.parse_args(["diff-model", "only_one.gguf"])
            raised = False
        except SystemExit:
            raised = True
    assert raised
