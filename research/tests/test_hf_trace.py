"""No-model tests for the HF chat per-token trace capture (issue B3, HF path).

The studio's MAIN chat (the HF Qwen path) now records, per generated response token, the model's
confidence and the alternatives it weighed -- WITHOUT altering the generated text. It does so with a
pure pass-through transformers.LogitsProcessor that only OBSERVES the score row, plus a helper that
aligns the recorded rows to the tokens actually emitted. Both live in self_teach_server and are model-
free, so we exercise them here with hand-made logits tensors and a stub tokenizer -- no HF, no GGUF,
no CUDA, no generate() call.

We assert the three load-bearing guarantees:
  (a) the processor returns `scores` UNCHANGED (torch.equal) -- this is what makes the reply byte-identical
  (b) it records the model distribution such that the committed token's confidence is its own probability
      (the argmax's prob under greedy) and the alternatives are the other top-k, chosen excluded
  (c) steps_from_records + runlog.steps_to_trace yield parallel, aligned tokens/confidence/alternatives
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))               # research/ (self_teach_server, runlog live here)

import runlog                                               # noqa: E402
from self_teach_server import RecordingLogitsProcessor, steps_from_records   # noqa: E402


# --------------------------------------------------------------------------- a stub tokenizer
# steps_from_records only ever calls tok.decode([id]); map an id to a readable piece so we can assert.
class StubTok:
    def __init__(self, vocab):
        self.vocab = vocab                                  # {id: piece}

    def decode(self, ids, **kw):
        return "".join(self.vocab.get(int(i), f"<{int(i)}>") for i in ids)


VOCAB = {0: "The", 1: " cat", 2: " dog", 3: " fox", 4: " sat", 5: " ran", 6: "<eos>"}


def logits_row(scores_by_id):
    """A [1, V] logits tensor with the given raw scores at those ids (others very negative)."""
    row = torch.full((1, len(VOCAB)), -30.0)
    for i, s in scores_by_id.items():
        row[0, i] = float(s)
    return row


# --------------------------------------------------------------------------- (a) pure pass-through

def test_processor_returns_scores_unchanged():
    proc = RecordingLogitsProcessor(topk=4)
    scores = torch.randn(1, len(VOCAB))
    before = scores.clone()
    out = proc(torch.tensor([[0, 1]]), scores)             # input_ids shape is irrelevant to a pass-through
    assert out is scores                                   # same object handed straight back
    assert torch.equal(out, before)                        # and byte-identical contents -> reply is unchanged
    assert torch.equal(scores, before)                     # it did not mutate in place either


def test_processor_never_mutates_across_many_steps():
    proc = RecordingLogitsProcessor()
    for _ in range(5):
        s = torch.randn(1, len(VOCAB))
        snap = s.clone()
        r = proc(torch.tensor([[0]]), s)
        assert torch.equal(r, snap)
    assert len(proc.records) == 5                           # one record per call/step


def test_processor_tolerates_bad_scores_without_raising():
    """A malformed score row must not throw (observation can't be allowed to break generation)."""
    proc = RecordingLogitsProcessor()
    out = proc(torch.tensor([[0]]), "not-a-tensor")        # type: ignore[arg-type]
    assert out == "not-a-tensor"                            # returned untouched
    assert proc.records[-1] == {"ids": [], "probs": []}    # degraded to an empty row, no crash


# --------------------------------------------------------------------------- (b) records the distribution

def test_records_topk_ids_and_probs():
    proc = RecordingLogitsProcessor(topk=3)
    # scores chosen so softmax ranks: id2 > id1 > id4 > ... (id2 the argmax)
    proc(torch.tensor([[0]]), logits_row({2: 5.0, 1: 4.0, 4: 3.0, 0: 0.0}))
    rec = proc.records[0]
    assert rec["ids"][0] == 2                               # argmax first
    assert rec["ids"] == [2, 1, 4]                          # top-3 by prob, descending
    # probs are a real softmax over the row: descending, in (0,1], carrying most of the mass (the ~thousands
    # of suppressed tail ids hold a small remainder, so the top-3 sum is just under 1).
    assert rec["probs"] == sorted(rec["probs"], reverse=True)
    assert all(0.0 < p <= 1.0 for p in rec["probs"])
    assert 0.9 < sum(rec["probs"]) <= 1.0                 # top-3 carry ~all mass; tail is a small remainder


def test_greedy_confidence_is_argmax_prob_and_alts_exclude_chosen():
    """Under greedy the emitted token IS the argmax, so its confidence must equal the argmax's probability
    and the alternatives are the remaining top-k -- the chosen token excluded."""
    proc = RecordingLogitsProcessor(topk=4)
    proc(torch.tensor([[0]]), logits_row({2: 5.0, 1: 4.0, 4: 3.0, 3: 2.0}))
    rec = proc.records[0]
    argmax_id = rec["ids"][0]                               # == 2
    argmax_prob = rec["probs"][0]
    tok = StubTok(VOCAB)
    steps = steps_from_records(proc.records, gen_ids=[argmax_id], tok=tok)   # greedy: emitted == argmax
    assert len(steps) == 1
    step = steps[0]
    assert step["piece"] == " dog"                         # id 2
    assert step["conf"] == round(argmax_prob, 4)           # confidence is the committed (argmax) token's prob
    assert all(a["piece"] != " dog" for a in step["alts"])  # chosen excluded from alternatives
    assert [a["piece"] for a in step["alts"]] == [" cat", " sat", " fox"]   # the other top-k, ranked
    assert len(step["alts"]) <= 3


def test_sampled_confidence_is_the_emitted_tokens_own_prob():
    """Under sampling the emitted token often isn't the argmax. Its confidence must be ITS probability,
    not the argmax's, and it must be excluded from its own alternatives."""
    proc = RecordingLogitsProcessor(topk=4)
    proc(torch.tensor([[0]]), logits_row({2: 5.0, 1: 4.0, 4: 3.0, 3: 2.0}))
    rec = proc.records[0]
    prob_by_id = dict(zip(rec["ids"], rec["probs"]))
    emitted = 1                                             # pretend the sampler drew id 1 (" cat"), not argmax 2
    tok = StubTok(VOCAB)
    steps = steps_from_records(proc.records, gen_ids=[emitted], tok=tok)
    assert steps[0]["piece"] == " cat"
    assert steps[0]["conf"] == round(prob_by_id[1], 4)     # the emitted token's own probability
    assert all(a["piece"] != " cat" for a in steps[0]["alts"])   # the chosen (emitted) token is excluded
    assert " dog" in [a["piece"] for a in steps[0]["alts"]]      # the argmax (id 2) is now just an alternative


def test_emitted_token_outside_topk_gets_zero_confidence():
    """A deep sampling-tail pick that fell outside the recorded top-k -> confidence 0.0 (honest: we did
    not observe its probability), and it's still reported as the committed piece."""
    proc = RecordingLogitsProcessor(topk=2)                # record only top-2
    proc(torch.tensor([[0]]), logits_row({2: 5.0, 1: 4.0, 4: 3.0}))   # top-2 == {2,1}; id 4 is outside
    tok = StubTok(VOCAB)
    steps = steps_from_records(proc.records, gen_ids=[4], tok=tok)
    assert steps[0]["piece"] == " sat"                     # id 4 still shown as committed
    assert steps[0]["conf"] == 0.0                         # its prob wasn't in the recorded top-2


# --------------------------------------------------------------------------- (c) steps -> aligned trace

def test_steps_align_one_per_generated_token():
    """A multi-step generation: steps_from_records yields exactly one step per emitted id, in order."""
    proc = RecordingLogitsProcessor(topk=3)
    proc(torch.tensor([[0]]), logits_row({0: 6.0, 1: 1.0, 2: 0.5}))    # step 0: argmax id 0 "The"
    proc(torch.tensor([[0, 0]]), logits_row({1: 5.0, 2: 4.0, 3: 1.0}))  # step 1: argmax id 1 " cat"
    proc(torch.tensor([[0, 0, 1]]), logits_row({4: 5.0, 5: 4.0, 0: 1.0}))  # step 2: argmax id 4 " sat"
    tok = StubTok(VOCAB)
    gen_ids = [0, 1, 4]                                     # greedy emitted sequence "The cat sat"
    steps = steps_from_records(proc.records, gen_ids, tok)
    assert len(steps) == 3
    assert [s["piece"] for s in steps] == ["The", " cat", " sat"]


def test_ragged_lengths_align_to_the_shorter():
    """If the recorder and the emitted-id list disagree in length (e.g. a trailing EOS trimmed away),
    alignment uses the shorter -> never an index error, never a phantom token."""
    proc = RecordingLogitsProcessor(topk=2)
    proc(torch.tensor([[0]]), logits_row({0: 5.0, 1: 1.0}))
    proc(torch.tensor([[0, 0]]), logits_row({1: 5.0, 2: 1.0}))
    proc(torch.tensor([[0, 0, 1]]), logits_row({4: 5.0, 5: 1.0}))   # 3 records
    tok = StubTok(VOCAB)
    steps = steps_from_records(proc.records, gen_ids=[0, 1], tok=tok)   # only 2 emitted ids
    assert [s["piece"] for s in steps] == ["The", " cat"]


def test_steps_feed_runlog_steps_to_trace_aligned():
    """The steps this helper produces flow straight into runlog.steps_to_trace -> parallel arrays with
    one entry per generated token (the on-disk trace contract read by the Run Inspector)."""
    proc = RecordingLogitsProcessor(topk=3)
    proc(torch.tensor([[0]]), logits_row({0: 6.0, 1: 2.0, 2: 1.0}))
    proc(torch.tensor([[0, 0]]), logits_row({1: 5.0, 2: 4.0, 3: 3.0}))
    tok = StubTok(VOCAB)
    steps = steps_from_records(proc.records, gen_ids=[0, 1], tok=tok)
    trace = runlog.steps_to_trace(steps)
    assert trace["tokens"] == ["The", " cat"]
    assert len(trace["confidence"]) == 2
    assert len(trace["alternatives"]) == 2                 # parallel, one per token
    # step 1 committed id 1 -> its alternatives exclude " cat" and are the other top-k
    assert all(a["piece"] != " cat" for a in trace["alternatives"][1])
    # confidences are real probabilities in (0, 1]
    assert all(0.0 < c <= 1.0 for c in trace["confidence"])


def test_empty_records_give_empty_steps_and_empty_trace():
    """No records / no ids -> [] steps -> runlog stores a clean empty {} (the documented degrade path)."""
    tok = StubTok(VOCAB)
    assert steps_from_records([], [], tok) == []
    assert steps_from_records([{"ids": [0], "probs": [0.9]}], [], tok) == []   # nothing emitted
    assert runlog.steps_to_trace(steps_from_records([], [], tok)) == {}
