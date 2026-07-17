"""Pure-logic tests for the CAA (contrastive-activation-addition) exemplar-bank derivation path added to
scripts/calibration/torch_autocalibrate.py -- derive_caa_directions / apply_caa_directions / _pooled_reply_
resid, plus the --exemplars CLI wiring. See torch_autocalibrate.py's own module docstring ("--exemplars
MODE" section) for the why: on Qwen3.5-9B, the pre-existing one-instruction-pair recipe carved a near-zero
axis for 11 of 12 dials, while the identical harness steered Qwen2.5-14B strongly -- this feature lets the
harness derive directions from dial_exemplars.json's matched-pair bank instead, so the two recipes can be
A/B'd head to head on the SAME sweep. This file tests the DERIVATION math and the fallback/bookkeeping
logic only, model-free throughout: NO real model or tokenizer is ever loaded, and no subprocess/GPU work is
launched (a live GPU A/B run is a separate, manual experiment -- see the module docstring's "Run" section).

_pooled_reply_resid's own read of a reply's residual was REWORKED (see its docstring in
torch_autocalibrate.py) from prefix-LENGTH subtraction (`full_ids[0, len(template([user],
add_generation_prompt=True)):]`, assuming the generation prompt is a literal token-prefix of the full
render) to CHARACTER-span location: render the full chat turn as TEXT, `rfind` the reply's own characters
in it, and map that char span to tokens via a fast tokenizer's offset mapping. This was forced by a
MEASURED failure on Qwen3.5-9B, a reasoning model: its generation-prompt render ends inside an OPEN
`<think>` block, while the full render shows that block already CLOSED and empty before the reply -- so the
two renders do not nest, and prefix subtraction started the pool on `</think>` and ran through the trailing
end-of-turn token, i.e. it pooled scaffold+reply, not the reply. That scaffold does not cancel out in a
pos-vs-neg CAA diff (see the regression test below for the full argument). The fake substrate in this file
was reworked in lockstep to exercise the NEW contract.

FAKE SUBSTRATE (no torch.nn.Module, no transformers): FakeTok.apply_chat_template(msgs, tokenize=False)
renders a chat turn as literal TEXT -- "<user>\\n{content}\\n<asst>\\n{content}\\n", one line per turn --
built from each message's content VERBATIM (never tokenized at this step; tokenize=False is asserted, since
that is the only mode the new _pooled_reply_resid ever invokes it in). Tokenization happens in a SEPARATE
step, via FakeTok.__call__(text, return_tensors="pt", return_offsets_mapping=True,
add_special_tokens=False) -- exactly the second call _pooled_reply_resid actually makes -- which splits
`text` on whitespace into "words" (a toy stand-in for a real fast tokenizer's subword pieces; the mapping
logic under test only cares that spans are non-empty and correctly bounded, not about subword granularity)
via the pure, STATELESS `_word_id` function (a fixed formula of the word's own characters, never a shared/
mutable vocab dict -- so two independent calls, or two fakes in the same test, never disagree about what a
word tokenizes to, and a test can recompute the SAME id independently to hand-derive an expected pooled
vector) and reports each word's real CHARACTER offsets into `text`, mirroring `return_offsets_mapping=True`.
FakeModel's forward pass returns a CANNED hidden_states[layer+1] tensor where token id `t`'s own hidden
vector is `t` broadcast across every hidden dim (_token_vec) -- so a mean-pool over any token span is
exactly the mean of THOSE tokens' ids, trivially hand-verifiable without any real linear algebra.
"""
from __future__ import annotations

import os
import re
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)                                          # repo root, for `from clozn import ...`
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "calibration"))  # torch_autocalibrate.py + dial_exemplars.py
import torch_autocalibrate as dac  # noqa: E402
import dial_exemplars  # noqa: E402


# ================================================================================================
# fake tokenizer / model substrate -- no torch.nn.Module, no transformers, no CUDA
# ================================================================================================
HIDDEN = 4                                     # a tiny hidden size -- just enough to distinguish tensors
_TOKEN_RE = re.compile(r"\S+")                 # a fast tokenizer's subword pieces, toy-ified to whitespace runs


def _word_id(word: str) -> int:
    """Deterministic, STATELESS word -> token-id mapping: a pure function of the word's own characters, so
    a test can recompute the identical id independently (no shared vocab dict to get out of sync, no
    encounter-order dependence across fakes or across calls within one test)."""
    return 100 + sum(ord(c) for c in word) % 400


def _token_vec(token_id: int) -> torch.Tensor:
    """Every token's canned hidden state is its own id, broadcast across HIDDEN dims -- a mean-pool over a
    span of tokens is then exactly the mean of those tokens' ids, so a test can tell at a glance (and
    without any real matrix math) whether a token outside the intended span leaked into a pooled result."""
    return torch.full((HIDDEN,), float(token_id))


def _tokenize_with_offsets(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """A toy WHITESPACE tokenizer standing in for a real HuggingFace *fast* tokenizer's
    `return_offsets_mapping=True`: split `text` into maximal non-whitespace runs ("words"), map each to a
    stable id via _word_id, and record its own (start, end) CHARACTER span in `text`. This is the one
    capability _pooled_reply_resid's char-span-to-token-span mapping now depends on -- a real fast
    tokenizer's subword pieces would report finer-grained spans, but the overlap logic under test
    (`b > a and b > char_start and a < char_end`) only cares that spans are non-empty and correctly
    bounded, which word-level granularity exercises identically to subword-level."""
    ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        ids.append(_word_id(m.group()))
        offsets.append((m.start(), m.end()))
    return ids, offsets


class _FakeBatchEncoding(dict):
    """Minimal stand-in for a real fast tokenizer's BatchEncoding: a dict-like object exposing both
    ["input_ids"] and ["offset_mapping"] -- exactly what `sc.tok(text, ..., return_offsets_mapping=True,
    add_special_tokens=False)` returns for a real fast tokenizer, and what _pooled_reply_resid reads both
    keys off of directly. (Unlike the apply_chat_template idiom used elsewhere in torch_autocalibrate.py --
    _last_resid/Rig.gen -- there is no bare-tensor fallback to model here: return_offsets_mapping is a
    fast-tokenizer-only feature and always comes back structured.)"""


class FakeTok:
    """apply_chat_template(msgs, tokenize=False) renders a chat turn as literal, hand-decodable TEXT --
    "<user>\\n{content}\\n" then "<asst>\\n{content}\\n" -- with each message's content copied in VERBATIM.
    `tokenize=False` is asserted: the new _pooled_reply_resid only ever calls apply_chat_template this one
    way (to get rendered text for `rfind`), never to get token ids directly -- tokenization is a SEPARATE,
    later step via __call__, matching the real function's own two-step read exactly."""

    ASST_MARK = "<asst>"   # stands in for a real template's opened generation-prompt scaffold

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False, "the new _pooled_reply_resid path only ever renders TEXT here"
        parts = []
        for m in messages:
            marker = "<user>" if m["role"] == "user" else self.ASST_MARK
            parts.append(f"{marker}\n{m['content']}\n")
        return "".join(parts)

    def __call__(self, text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False):
        assert return_tensors == "pt" and return_offsets_mapping and not add_special_tokens, (
            "this fake only models the ONE call shape _pooled_reply_resid actually makes")
        ids, offsets = _tokenize_with_offsets(text)
        return _FakeBatchEncoding(
            input_ids=torch.tensor([ids], dtype=torch.long),
            offset_mapping=torch.tensor([offsets], dtype=torch.long),
        )


class _ReasoningTok(FakeTok):
    """Stands in for a REASONING model's chat template -- Qwen3.5-9B is the REAL one that forced this
    rework, see torch_autocalibrate.py's _pooled_reply_resid docstring and this file's module docstring.
    The full (user+assistant) render puts an opened-then-CLOSED, EMPTY `<think>` block between the
    assistant marker and the reply's own content -- "<think>\\n\\n</think>\\n\\n" -- exactly the shape that
    broke prefix-length subtraction: a real reasoning template's generation-prompt render ends INSIDE the
    still-open think block ("...assistant\\n<think>\\n"), so it is not a literal token-prefix of the full
    render, which shows the block already closed. This fake only needs to model the FULL render (the new
    code never renders a separate generation-prompt-only pass), so add_generation_prompt support is dropped
    entirely -- it would be dead code under the new contract."""

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        user_msg = messages[0]["content"]
        text = f"<user>\n{user_msg}\n{self.ASST_MARK}\n<think>\n\n</think>\n\n"
        if len(messages) > 1:
            text += f"{messages[1]['content']}\n<endasst>\n"   # <endasst> stands in for a real "<|im_end|>"
        return text


class _MangleReplyTok(FakeTok):
    """Simulates a template that NORMALIZES the assistant's rendered text away from the reply's own exact
    characters -- HTML-entity escaping, NFKC normalization, and case-folding are all real examples; this
    fake uses upper-casing as a trivially hand-verifiable stand-in for all of them. Whatever the mechanism,
    the effect is the same: `full_text.rfind(reply)` can never find the reply's original-case text, so
    `char_start < 0` and _pooled_reply_resid must return None -- SKIP, never pool a garbage/nearby span."""

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        mangled = list(messages)
        if mangled and mangled[-1]["role"] == "assistant":
            mangled[-1] = {"role": "assistant", "content": mangled[-1]["content"].upper()}
        return super().apply_chat_template(mangled)


class _FakeOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeModel:
    """model(ids, output_hidden_states=True).hidden_states[layer+1] is a CANNED tensor keyed off the input
    ids (_token_vec) -- no real forward pass. Every OTHER hidden_states index is left None, so a bug that
    read the wrong layer would crash loudly (None has no .float()) rather than silently pass."""

    def __init__(self, layer: int):
        self.layer = layer

    def __call__(self, ids, output_hidden_states=True):
        assert output_hidden_states is True
        seq = ids[0]
        vecs = torch.stack([_token_vec(int(t)) for t in seq]).unsqueeze(0)   # [1, seq_len, HIDDEN]
        hs = [None] * (self.layer + 2)
        hs[self.layer + 1] = vecs
        return _FakeOutput(hs)


class FakeSC:
    """Stands in for SteeringControl/SingleTurnSteer: only .tok/.model/.layer are read by
    _pooled_reply_resid/derive_caa_directions."""

    def __init__(self, layer: int = 1):
        self.tok = FakeTok()
        self.model = FakeModel(layer)
        self.layer = layer


# ================================================================================================
# _pooled_reply_resid -- the CAA primitive: chat-context reply-span pooling
# ================================================================================================
def test_pooled_reply_resid_matches_hand_derived_mean_of_reply_tokens():
    sc = FakeSC()
    reply = "gamma delta epsilon"
    pooled = dac._pooled_reply_resid(sc, "alpha beta", reply)
    expected_mean = sum(_word_id(w) for w in reply.split()) / 3
    assert pooled is not None
    assert torch.allclose(pooled, torch.full((HIDDEN,), expected_mean), atol=1e-5)


def test_pooled_reply_resid_prompt_tokens_dont_affect_the_result():
    """THE core guarantee this recipe depends on: changing the PROMPT (even to a totally different length
    and vocabulary) while holding the reply fixed must not move the pooled result at all. Under the new
    char-span code the prompt's own tokens are still rendered and encoded (they're part of `full_text`), but
    the reply's char span is located by `rfind`-ing the reply's OWN text, independent of what precedes it --
    so a regression that pooled from token 0, or that let a long prompt shift the span, would fail this
    immediately. (Neither prompt below contains "gamma delta epsilon" as a substring, so there is exactly
    one place in each rendered text `rfind` can match.)"""
    sc = FakeSC()
    reply = "gamma delta epsilon"
    pooled_short_prompt = dac._pooled_reply_resid(sc, "alpha beta", reply)
    pooled_long_prompt = dac._pooled_reply_resid(sc, "totally different and much longer prompt text here", reply)
    assert torch.allclose(pooled_short_prompt, pooled_long_prompt, atol=1e-6)


def test_pooled_reply_resid_empty_reply_returns_none():
    """An empty reply, or a reply that is only whitespace, can never overlap any real token's char span (no
    fast-tokenizer token is ever whitespace-only) -- the offset-overlap filter comes back empty, and
    _pooled_reply_resid must return None: SKIP, never pool an empty/garbage mean. (This guard used to be
    framed as "prefix_len >= len(full_ids)" under prefix subtraction; under the new char-span code the
    equivalent failure mode is simply an empty overlap set -- for "" specifically, `str.rfind("")` returns
    `len(full_text)`, which is >= 0, so this exercises the SECOND guard -- empty overlap -- not the first.)"""
    sc = FakeSC()
    assert dac._pooled_reply_resid(sc, "alpha beta", "") is None
    assert dac._pooled_reply_resid(sc, "alpha beta", "   ") is None


def test_pooled_reply_resid_uses_the_sc_layer_plus_one_convention():
    """hidden_states[sc.layer + 1] is the SAME indexing convention _last_resid/directional_alignment
    already use -- a different sc.layer must read a different hidden_states slot, and FakeModel only
    populates that one slot (every other index is None), so a wrong-layer bug would raise, not silently
    return a stale value."""
    sc = FakeSC(layer=3)
    pooled = dac._pooled_reply_resid(sc, "alpha beta", "gamma delta")
    expected_mean = sum(_word_id(w) for w in ("gamma", "delta")) / 2
    assert torch.allclose(pooled, torch.full((HIDDEN,), expected_mean), atol=1e-5)


def test_pooled_reply_resid_plain_non_reasoning_template_still_pools_correctly():
    """No-regression companion to the reasoning-template test below: a plain, Qwen2.5-style template (the
    assistant turn is just a role marker + content, no think block at all) must keep pooling EXACTLY the
    reply's own tokens under the new char-span code -- the SAME rfind + offset-mapping logic has to handle
    both template shapes correctly, with no per-template special-casing anywhere in _pooled_reply_resid.
    Uses the SAME prompt/reply text as the reasoning-template regression test on purpose, so the two tests
    read as a matched pair."""
    sc = FakeSC()   # sc.tok is the plain FakeTok
    prompt = "Describe the city at dusk."
    reply = "The evening folded over the city."
    pooled = dac._pooled_reply_resid(sc, prompt, reply)
    reply_words = reply.split()
    expected = torch.full((HIDDEN,), sum(_word_id(w) for w in reply_words) / len(reply_words))
    assert pooled is not None
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_pooled_reply_resid_reasoning_template_excludes_think_scaffold_and_end_token():
    """THE regression this whole rework exists to fix (see torch_autocalibrate.py's module docstring and
    _pooled_reply_resid's own docstring, and this file's module docstring): on a reasoning model, the
    generation-prompt render ends INSIDE an open `<think>` block, but the full (user+assistant) render shows
    that block already opened-and-closed, EMPTY, before the reply. The OLD prefix-length-subtraction code
    assumed `template([user], add_generation_prompt=True)` is a literal token PREFIX of
    `template([user, assistant])` -- false here -- so it started pooling on the FIRST token where the two
    renders diverge (`</think>`) and ran through to the end, ALSO swallowing the trailing end-of-turn token.

    That scaffold does NOT cancel out in a pos-vs-neg CAA diff. `</think>` and the end-of-turn token's
    hidden states are IDENTICAL across a matched pair -- same prompt, same scaffold, and causal attention
    only looks backward, so nothing about a LATER, different reply can change an EARLIER token's residual.
    But the two replies in a pair differ in LENGTH (N_pos != N_neg tokens), so a mean-pool that includes the
    shared scaffold averages that SAME scaffold vector in with a DIFFERENT WEIGHT on each side (1/N_pos vs.
    1/N_neg). That is a silent, ASYMMETRIC contamination of exactly the vector this recipe measures: the
    resulting "direction" would partly encode "which reply was longer", not "which reply was positive vs.
    negative" for the trait being calibrated -- a subtle bug no amount of staring at either reply alone would
    catch, because each pole's OWN pooled vector still looks like a plausible average.

    The new char-span code sidesteps all of this by finding the reply's own CHARACTERS in the rendered text
    (`rfind`) and mapping THOSE to tokens via the offset mapping -- correct whether or not a think block is
    present, with no per-template special-casing. This test does not just assert the pooled result is
    correct; it also hand-computes what the OLD bug would have produced instead and asserts the two differ,
    so a future "simplification" back toward prefix arithmetic would fail this loudly, not silently."""
    sc = FakeSC()
    sc.tok = _ReasoningTok()
    prompt = "Describe the city at dusk."
    reply = "The evening folded over the city."
    pooled = dac._pooled_reply_resid(sc, prompt, reply)
    assert pooled is not None

    reply_words = reply.split()
    clean_mean = sum(_word_id(w) for w in reply_words) / len(reply_words)
    assert torch.allclose(pooled, torch.full((HIDDEN,), clean_mean), atol=1e-5), (
        "pooled result must be EXACTLY the mean of the reply's own tokens -- no scaffold, no end token")

    # Hand-compute what the OLD prefix-subtraction bug would have pooled instead: starting from '</think>'
    # (the first token where the prefix render and full render diverge) through the reply, PLUS the
    # trailing '<endasst>' end-of-turn marker (this fake's stand-in for a real template's '<|im_end|>').
    contaminated_words = ["</think>"] + reply_words + ["<endasst>"]
    contaminated_mean = sum(_word_id(w) for w in contaminated_words) / len(contaminated_words)
    assert not torch.isclose(torch.tensor(clean_mean), torch.tensor(contaminated_mean)), (
        "test-construction sanity check: the decoy (old-bug) mean must actually differ from the clean mean, "
        "or the assertion below would pass for the wrong reason")
    assert not torch.allclose(pooled, torch.full((HIDDEN,), contaminated_mean), atol=1e-5), (
        "pooled result must NOT match what prefix-length subtraction would have produced -- if it does, "
        "'</think>' and/or the trailing end-of-turn token leaked back into the pooled span")


def test_pooled_reply_resid_rfind_prefers_the_assistants_own_last_occurrence():
    """When the reply's text ALSO appears earlier in the render -- e.g. embedded inside a longer word in the
    PROMPT -- `full_text.rfind(reply)` must resolve to the assistant's OWN, LATER occurrence, not the
    earlier one. Constructed so the two occurrences are NOT interchangeable if the wrong one is picked: the
    prompt contains "echo" only as a substring of the single longer word "echoback" (never as its own
    token), while the assistant's reply is the standalone word "echo" itself, later in the render. If
    _pooled_reply_resid used `find()` (first match) instead of `rfind()` (last match), char_start would land
    mid-word inside "echoback"; the offset-overlap check (`b > a and b > char_start and a < char_end`) would
    then match the WHOLE "echoback" token (its span overlaps the mis-located char range even though it
    isn't a clean match), and the pooled result would be word-id("echoback") -- a different, wrong value --
    instead of the correct word-id("echo")."""
    sc = FakeSC()
    prompt = "please echoback the phrase then continue"
    reply = "echo"
    pooled = dac._pooled_reply_resid(sc, prompt, reply)
    assert pooled is not None

    correct_mean = float(_word_id("echo"))
    decoy_mean = float(_word_id("echoback"))
    assert decoy_mean != correct_mean, "test-construction sanity check: decoy must differ from the real word"
    assert torch.allclose(pooled, torch.full((HIDDEN,), correct_mean), atol=1e-5), (
        "must pool the assistant's OWN standalone 'echo', not the 'echo' embedded inside the prompt's "
        "'echoback'")
    assert not torch.allclose(pooled, torch.full((HIDDEN,), decoy_mean), atol=1e-5)


def test_pooled_reply_resid_reply_absent_from_rendered_text_returns_none():
    """A template that normalizes/escapes the assistant's text away from the reply's own exact characters
    (_MangleReplyTok's stand-in: upper-casing) means `full_text.rfind(reply)` can never find it: char_start
    < 0, and the function must return None -- SKIP, never pool a garbage/nearby span."""
    sc = FakeSC()
    sc.tok = _MangleReplyTok()
    assert dac._pooled_reply_resid(sc, "alpha beta", "gamma delta") is None


# ================================================================================================
# derive_caa_directions -- the full per-dial mean-diff-of-pairs, unit-normalized
# ================================================================================================
def _make_pairs(n: int, prompt: str = "ask", pos_word: str = "posw", neg_word: str = "negw") -> list[dict]:
    """n MATCHED pairs (same prompt content, only the reply's single word varies by index so every pair's
    diff is distinguishable and non-degenerate) -- shaped like a real dial_exemplars.json pairs list."""
    return [{"prompt": prompt, "pos": f"{pos_word}{i}", "neg": f"{neg_word}{i}"} for i in range(n)]


def test_derive_caa_directions_diff_is_pos_minus_neg_averaged_and_unit_norm():
    sc = FakeSC()
    n = dial_exemplars.MIN_RECOMMENDED_PAIRS
    bank = {"dials": {"warm": {"pairs": _make_pairs(n)}}}

    directions, stats = dac.derive_caa_directions(sc, bank, ["warm"])

    assert "warm" in directions
    vec = directions["warm"]
    assert torch.isclose(vec.norm(), torch.tensor(1.0), atol=1e-5), "must be UNIT-normalized"

    # Hand-derive the expected mean-diff (pre-normalization) from the SAME deterministic word-id rule the
    # fake tokenizer uses, then confirm the stored vector is exactly THAT vector's own unit form.
    per_pair_diff = [_word_id(f"posw{i}") - _word_id(f"negw{i}") for i in range(n)]
    mean_diff_scalar = sum(per_pair_diff) / n
    expected_raw = torch.full((HIDDEN,), float(mean_diff_scalar))
    expected_unit = expected_raw / expected_raw.norm()
    assert torch.allclose(vec, expected_unit, atol=1e-5)

    assert stats["warm"] == {"pairs_used": n, "pairs_skipped": 0}


def test_derive_caa_directions_skips_and_counts_pairs_with_empty_reply_span():
    sc = FakeSC()
    n = dial_exemplars.MIN_RECOMMENDED_PAIRS - 1
    pairs = _make_pairs(n)
    pairs.append({"prompt": "ask", "pos": "", "neg": "negwX"})   # empty POS reply -> whole pair skipped
    bank = {"dials": {"warm": {"pairs": pairs}}}                 # n+1 pairs total -- still clears the gate

    directions, stats = dac.derive_caa_directions(sc, bank, ["warm"])

    assert stats["warm"]["pairs_used"] == n
    assert stats["warm"]["pairs_skipped"] == 1
    assert "warm" in directions, "the n usable pairs still produce a direction"


def test_derive_caa_directions_pair_missing_prompt_is_skipped_and_counted():
    """schema v2 pairs always carry `prompt` (bare-text pairs still LOAD per dial_exemplars.validate, but
    are the honest-but-weaker mode) -- a pair with no prompt at all can't be read in chat context, so this
    recipe skips it outright rather than silently degrading to bare-text for just that one pair."""
    sc = FakeSC()
    n = dial_exemplars.MIN_RECOMMENDED_PAIRS - 1
    pairs = _make_pairs(n)
    pairs.append({"pos": "poswX", "neg": "negwX"})   # no "prompt" key at all
    bank = {"dials": {"warm": {"pairs": pairs}}}

    directions, stats = dac.derive_caa_directions(sc, bank, ["warm"])

    assert stats["warm"]["pairs_used"] == n
    assert stats["warm"]["pairs_skipped"] == 1


def test_derive_caa_directions_dial_under_min_pairs_is_not_attempted():
    """A dial below dial_exemplars.MIN_RECOMMENDED_PAIRS is not even ATTEMPTED (no direction, no stats
    entry) -- the caller (apply_caa_directions) is expected to fall it back to the instruction recipe."""
    sc = FakeSC()
    bank = {"dials": {"warm": {"pairs": _make_pairs(dial_exemplars.MIN_RECOMMENDED_PAIRS - 1)}}}

    directions, stats = dac.derive_caa_directions(sc, bank, ["warm"])

    assert "warm" not in directions
    assert "warm" not in stats


def test_derive_caa_directions_only_requested_dials_and_bank_coverage_intersect():
    """A dial the bank has plenty of pairs for, but that wasn't in `dial_names`, must never be derived --
    the function only ever touches the requested subset."""
    sc = FakeSC()
    n = dial_exemplars.MIN_RECOMMENDED_PAIRS
    bank = {"dials": {
        "warm": {"pairs": _make_pairs(n, prompt="ask1", pos_word="pw", neg_word="nw")},
        "candid": {"pairs": _make_pairs(n, prompt="ask2", pos_word="cw", neg_word="dw")},
    }}
    directions, stats = dac.derive_caa_directions(sc, bank, ["warm"])
    assert set(directions) == {"warm"}
    assert set(stats) == {"warm"}


def test_derive_caa_directions_zero_usable_pairs_yields_no_direction_but_records_stats():
    """Every pair in an otherwise-eligible dial fails the reply-span guard -- here, _MangleReplyTok makes
    EVERY reply's exact text unfindable in the rendered chat (see test_pooled_reply_resid_reply_absent_
    from_rendered_text_returns_none above). This replaces the old regression's mechanism (a template that
    doesn't nest prefix-in-full), which no longer exists as a failure mode under the new char-span code --
    a non-nesting template is now handled CORRECTLY, not skipped (that's the whole point of the rework; see
    the reasoning-template regression test above, which is exactly such a template). The dial still clears
    MIN_RECOMMENDED_PAIRS so IS attempted (appears in stats), but produces no direction at all -- it never
    silently averages zero vectors into a fake 'direction'."""
    sc = FakeSC()
    sc.tok = _MangleReplyTok()
    n = dial_exemplars.MIN_RECOMMENDED_PAIRS
    bank = {"dials": {"warm": {"pairs": _make_pairs(n)}}}

    directions, stats = dac.derive_caa_directions(sc, bank, ["warm"])

    assert "warm" not in directions
    assert stats["warm"] == {"pairs_used": 0, "pairs_skipped": n}


# ================================================================================================
# apply_caa_directions -- the sc.vecs swap + dial_source bookkeeping (model-free: a bare object with .vecs)
# ================================================================================================
class _BareSC:
    def __init__(self, vecs):
        self.vecs = vecs


def test_apply_caa_directions_swaps_covered_dials_only():
    original_candid = torch.tensor([0.0, 1.0])
    sc = _BareSC({"warm": torch.tensor([1.0, 0.0]), "candid": original_candid})
    new_warm = torch.tensor([0.6, 0.8])

    source = dac.apply_caa_directions(sc, ["warm", "candid"], {"warm": new_warm})

    assert torch.equal(sc.vecs["warm"], new_warm)              # swapped to the CAA direction
    assert torch.equal(sc.vecs["candid"], original_candid)     # untouched -- falls back to instructions
    assert source == {"warm": "exemplars", "candid": "instructions"}


def test_apply_caa_directions_no_directions_means_every_dial_falls_back():
    sc = _BareSC({"warm": torch.tensor([1.0, 0.0])})
    source = dac.apply_caa_directions(sc, ["warm"], {})
    assert source == {"warm": "instructions"}
    assert torch.equal(sc.vecs["warm"], torch.tensor([1.0, 0.0]))


def test_apply_caa_directions_dial_source_covers_every_requested_name_in_order():
    sc = _BareSC({"warm": torch.tensor([1.0]), "candid": torch.tensor([1.0]), "poetic": torch.tensor([1.0])})
    source = dac.apply_caa_directions(sc, ["warm", "candid", "poetic"],
                                      {"warm": torch.tensor([0.5]), "poetic": torch.tensor([0.5])})
    assert list(source) == ["warm", "candid", "poetic"]
    assert source == {"warm": "exemplars", "candid": "instructions", "poetic": "exemplars"}


# ================================================================================================
# end-to-end (still model-free): derive_caa_directions -> apply_caa_directions, mirroring how run() wires
# them -- a dial under MIN_RECOMMENDED_PAIRS never gets swapped and dial_source says "instructions".
# ================================================================================================
def test_end_to_end_under_min_pairs_dial_falls_back_through_the_real_wiring():
    sc_vecs = {"warm": torch.tensor([1.0, 0.0]), "candid": torch.tensor([0.0, 1.0])}
    sc = FakeSC()
    sc.vecs = sc_vecs   # FakeSC doesn't carry .vecs itself -- bolt it on for this apply_caa_directions call

    bank = {"dials": {
        "warm": {"pairs": _make_pairs(dial_exemplars.MIN_RECOMMENDED_PAIRS, prompt="ask", pos_word="pw", neg_word="nw")},
        "candid": {"pairs": _make_pairs(dial_exemplars.MIN_RECOMMENDED_PAIRS - 1, prompt="ask", pos_word="cw", neg_word="dw")},
    }}
    directions, stats = dac.derive_caa_directions(sc, bank, ["warm", "candid"])
    source = dac.apply_caa_directions(sc, ["warm", "candid"], directions)

    assert source["warm"] == "exemplars"
    assert source["candid"] == "instructions"
    assert not torch.equal(sc.vecs["warm"], torch.tensor([1.0, 0.0]))   # swapped
    assert torch.equal(sc.vecs["candid"], torch.tensor([0.0, 1.0]))     # unchanged -- fell back
    assert "candid" not in stats                                        # never attempted (under the gate)
    assert stats["warm"]["pairs_used"] == dial_exemplars.MIN_RECOMMENDED_PAIRS


# ================================================================================================
# CLI wiring -- --exemplars [PATH], nargs="?" with a const default
# ================================================================================================
def test_arg_parser_exemplars_default_none_when_flag_omitted():
    a = dac.build_arg_parser().parse_args([])
    assert a.exemplars is None


def test_arg_parser_exemplars_bare_flag_defaults_to_the_bank_path():
    a = dac.build_arg_parser().parse_args(["--exemplars"])
    assert a.exemplars == dial_exemplars.DEFAULT_PATH
    assert os.path.basename(a.exemplars) == "dial_exemplars.json"


def test_arg_parser_exemplars_explicit_path_overrides_the_default():
    a = dac.build_arg_parser().parse_args(["--exemplars", "some/other/bank.json"])
    assert a.exemplars == "some/other/bank.json"


def test_arg_parser_exemplars_combines_with_other_flags():
    a = dac.build_arg_parser().parse_args(["--exemplars", "--dials", "warm", "candid", "--smoke"])
    assert a.exemplars == dial_exemplars.DEFAULT_PATH
    assert a.dials == ["warm", "candid"]
    assert a.smoke is True


# ================================================================================================
# sanity: this feature must not have broken loading the real, shipped exemplar bank through the SAME
# dial_exemplars.load()/.ready() calls derive_caa_directions makes internally (still no model -- .ready()
# and bank["dials"][name]["pairs"] are pure dict/JSON operations).
# ================================================================================================
def test_real_shipped_bank_loads_and_every_listed_dial_is_ready():
    bank = dial_exemplars.load(dial_exemplars.DEFAULT_PATH)
    errors, _warnings = dial_exemplars.validate(bank)
    assert errors == []
    ready = dial_exemplars.ready(bank)
    assert set(ready) == set(dial_exemplars.dial_names(bank)), (
        "every dial in the shipped bank is documented as >= MIN_RECOMMENDED_PAIRS -- if this ever fails "
        "after an edit to dial_exemplars.json, some dial silently dropped below the CAA-readiness bar")
