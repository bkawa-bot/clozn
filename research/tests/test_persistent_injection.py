"""test_persistent_injection -- MODEL-FREE tests for Exp 2 (the persistence phase diagram).

No model, no GPU: covers the pure geometry/metric/cell-sweep math (kv_geometry, warmth_rate,
shuffled_like, _kv_add/_edit_kv_span, _span_from_lengths, turns_to_noise, ALL_CELLS/_resolve_cells,
_scripted_conversation, _assemble_cell_result, _score_reply against a fake tokenizer, wants_four_bit, and
argparse wiring). torch is imported (shuffled_like/_kv_add/_edit_kv_span are tensor ops) but no model is
ever loaded -- the same "model-free" bar test_steering_headroom.py / test_dial_suggestion.py already use
for pure tensor-math helpers in a module that otherwise needs a GPU to run for real.

The GPU-side receipts (does the KV edit actually make a Qwen/Gemma reply warmer, does the curve decay the
way Law #3 predicts) are NOT here -- those need the real smoke/full runs (see persistent_injection.py's
module docstring for the run commands), which this suite deliberately cannot and does not attempt.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import persistent_injection as pinj  # noqa: E402


# ===================================================================================================
# kv_geometry -- read from config, never hardcoded (the pre-reg's exact gotcha: Gemma-2's head_dim=256
# does NOT equal hidden_size // num_attention_heads)
# ===================================================================================================
class FakeCfgQwen7B:
    num_hidden_layers = 28
    num_key_value_heads = 4
    num_attention_heads = 28
    hidden_size = 3584
    # no head_dim attribute at all -> must fall back to the derived value (3584 // 28 == 128)


class FakeCfgGemma9B:
    num_hidden_layers = 42
    num_key_value_heads = 8
    num_attention_heads = 16
    hidden_size = 3584
    head_dim = 256                    # EXPLICIT: 3584 // 16 == 224 would be WRONG -- must read this
    sliding_window = 4096
    sliding_window_pattern = 2


def test_kv_geometry_qwen7b_derives_head_dim_when_absent():
    geo = pinj.kv_geometry(FakeCfgQwen7B())
    assert geo == {"n_layers": 28, "n_kv_heads": 4, "n_attn_heads": 28, "head_dim": 128, "hidden_size": 3584}


def test_kv_geometry_gemma9b_reads_explicit_head_dim_not_derived():
    geo = pinj.kv_geometry(FakeCfgGemma9B())
    assert geo["head_dim"] == 256                 # NOT 3584 // 16 == 224
    assert geo["n_layers"] == 42 and geo["n_kv_heads"] == 8 and geo["n_attn_heads"] == 16
    assert geo["sliding_window"] == 4096
    assert geo["sliding_window_pattern"] == 2


def test_kv_geometry_omits_sliding_window_keys_when_absent():
    geo = pinj.kv_geometry(FakeCfgQwen7B())
    assert "sliding_window" not in geo
    assert "sliding_window_pattern" not in geo


def test_kv_geometry_head_dim_zero_falls_back_to_derived():
    class Cfg(FakeCfgQwen7B):
        head_dim = 0                  # falsy -> treated as absent, not literally zero-dim
    assert pinj.kv_geometry(Cfg())["head_dim"] == 128


# ===================================================================================================
# warmth_rate -- a RATE, not a raw count
# ===================================================================================================
def test_warmth_rate_basic_ratio():
    assert pinj.warmth_rate(score=4, n_tokens=20) == 0.2


def test_warmth_rate_zero_tokens_safe():
    assert pinj.warmth_rate(score=0, n_tokens=0) == 0.0
    assert pinj.warmth_rate(score=3, n_tokens=0) == 3.0        # max(1, 0) == 1, never divides by zero


def test_warmth_rate_longer_reply_cannot_win_on_count_alone():
    # same raw count, but the longer reply's RATE is lower -- the point of normalizing by length
    short = pinj.warmth_rate(score=2, n_tokens=10)
    long_ = pinj.warmth_rate(score=2, n_tokens=100)
    assert short > long_


# ===================================================================================================
# shuffled_like -- norm-preserving, direction-destroying, deterministic null
# ===================================================================================================
def test_shuffled_like_preserves_l2_norm():
    v = torch.randn(64)
    s = pinj.shuffled_like(v, seed=1)
    assert torch.allclose(v.norm(), s.norm(), atol=1e-5)


def test_shuffled_like_is_a_valid_permutation():
    v = torch.arange(16).float()
    s = pinj.shuffled_like(v, seed=7)
    assert torch.equal(torch.sort(s).values, torch.sort(v).values)


def test_shuffled_like_is_deterministic_given_seed():
    v = torch.randn(32)
    a = pinj.shuffled_like(v, seed=42)
    b = pinj.shuffled_like(v, seed=42)
    assert torch.equal(a, b)


def test_shuffled_like_different_seeds_differ():
    v = torch.randn(64)
    a = pinj.shuffled_like(v, seed=1)
    b = pinj.shuffled_like(v, seed=2)
    assert not torch.equal(a, b)


# ===================================================================================================
# _kv_add / _edit_kv_span -- the injection tensor + the in-place cache edit (nf4-safe dtype cast)
# ===================================================================================================
def test_kv_add_shape_and_scale():
    unit = torch.ones(8)                          # n_kv_heads=2, head_dim=4 -> 8 elems
    add = pinj._kv_add(unit, val_norm=10.0, dose=2.0, n_kv_heads=2, head_dim=4)
    assert add.shape == (2, 4)
    assert torch.allclose(add, torch.full((2, 4), 20.0))


class _FakeLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


def _make_fake_cache(n_layers=2, n_kv=2, seq=6, hd=4, dtype=torch.float32):
    layers = [_FakeLayer(torch.zeros(1, n_kv, seq, hd, dtype=dtype),
                         torch.zeros(1, n_kv, seq, hd, dtype=dtype)) for _ in range(n_layers)]
    return _FakeCache(layers)


def test_edit_kv_span_only_touches_the_span_on_values():
    cache = _make_fake_cache()
    add = torch.ones(2, 4)
    pinj._edit_kv_span(cache, layer=0, side="v", span=(2, 4), add_bh=add)
    v = cache.layers[0].values
    assert torch.all(v[:, :, 2:4, :] == 1.0)
    assert torch.all(v[:, :, :2, :] == 0.0)
    assert torch.all(v[:, :, 4:, :] == 0.0)
    # keys and the OTHER layer are untouched
    assert torch.all(cache.layers[0].keys == 0.0)
    assert torch.all(cache.layers[1].values == 0.0)


def test_edit_kv_span_targets_keys_when_side_is_k():
    cache = _make_fake_cache()
    add = torch.full((2, 4), 3.0)
    pinj._edit_kv_span(cache, layer=1, side="k", span=(0, 1), add_bh=add)
    assert torch.all(cache.layers[1].keys[:, :, 0, :] == 3.0)
    assert torch.all(cache.layers[1].keys[:, :, 1:, :] == 0.0)
    assert torch.all(cache.layers[1].values == 0.0)


def test_edit_kv_span_casts_to_the_live_tensor_dtype():
    # the nf4-safety fix: add_bh's dtype must NOT dictate the result -- the cache tensor's own dtype does
    cache = _make_fake_cache(dtype=torch.float64)
    add = torch.ones(2, 4, dtype=torch.float32)
    pinj._edit_kv_span(cache, layer=0, side="v", span=(0, 1), add_bh=add)
    assert cache.layers[0].values.dtype == torch.float64
    assert torch.all(cache.layers[0].values[:, :, 0, :] == 1.0)


def test_edit_kv_span_is_additive_not_overwrite():
    cache = _make_fake_cache()
    cache.layers[0].values[:, :, 0, :] = 5.0
    pinj._edit_kv_span(cache, layer=0, side="v", span=(0, 1), add_bh=torch.full((2, 4), 2.0))
    assert torch.all(cache.layers[0].values[:, :, 0, :] == 7.0)


# ===================================================================================================
# _span_from_lengths -- pure position-index math ('1' vs 'N')
# ===================================================================================================
def test_span_from_lengths_mode_N_is_the_whole_turn():
    assert pinj._span_from_lengths(before_len=10, incl_len=25, mode="N") == (10, 25)


def test_span_from_lengths_mode_1_is_the_last_position_only():
    assert pinj._span_from_lengths(before_len=10, incl_len=25, mode="1") == (24, 25)


def test_span_from_lengths_mode_1_span_width_is_always_one():
    lo, hi = pinj._span_from_lengths(before_len=0, incl_len=5, mode="1")
    assert hi - lo == 1


# ===================================================================================================
# turns_to_noise -- the operationalized half-life + the turn-0 gate
# ===================================================================================================
def test_turns_to_noise_gate_fails_when_turn0_does_not_clear_null():
    d_true = [0.01, 0.01, 0.01]
    d_null = [0.5, 0.5, 0.5]                       # null noise is LARGER than the true effect at turn 0
    r = pinj.turns_to_noise(d_true, d_null)
    assert r["gate_passed"] is False
    assert "GATED" in r["decay_note"]


def test_turns_to_noise_persists_when_never_decays():
    d_true = [1.0, 0.9, 0.95, 1.1]
    d_null = [0.05, 0.05, 0.05, 0.05]
    r = pinj.turns_to_noise(d_true, d_null)
    assert r["gate_passed"] is True
    assert r["turns_to_noise"] is None
    assert "PERSISTS" in r["decay_note"]


def test_turns_to_noise_decays_at_the_expected_turn():
    d_true = [1.0, 0.6, 0.02, 0.01]
    d_null = [0.1, 0.1, 0.1, 0.1]                  # noise_floor == 0.1
    r = pinj.turns_to_noise(d_true, d_null)
    assert r["gate_passed"] is True                # |1.0| > 0.1
    assert r["turns_to_noise"] == 2                # first |d_true| <= 0.1 is index 2 (0.02)
    assert r["noise_floor"] == 0.1
    assert r["turn0_effect"] == 1.0


def test_turns_to_noise_empty_input():
    r = pinj.turns_to_noise([], [])
    assert r["turns_to_noise"] is None
    assert r["gate_passed"] is False
    assert r["decay_note"] == "no data"


def test_turns_to_noise_noise_floor_is_mean_abs_not_per_turn():
    d_true = [1.0, 1.0]
    d_null = [0.0, 0.4]                            # mean(|.|) == 0.2, not the max (0.4) or min (0.0)
    r = pinj.turns_to_noise(d_true, d_null)
    assert r["noise_floor"] == 0.2


# ===================================================================================================
# the sweep cells: ALL_CELLS / SMOKE_CELL_IDS / _resolve_cells
# ===================================================================================================
def test_all_cells_has_ten_unique_ids():
    ids = [c["id"] for c in pinj.ALL_CELLS]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_grid_cells_is_the_full_2x2x2_factorial():
    assert len(pinj.GRID_CELLS) == 8
    combos = {(c["pos"], c["side"], c["cadence"]) for c in pinj.GRID_CELLS}
    assert combos == {(p, s, c) for p in ("1", "N") for s in ("k", "v") for c in ("once", "every_turn")}
    assert all(c["mechanism"] == "raw" for c in pinj.GRID_CELLS)


def test_extra_cells_are_kv_combined_and_phantom():
    ids = {c["id"] for c in pinj.EXTRA_CELLS}
    assert ids == {"Npos_KV_once", "phantom"}
    kv_cell = next(c for c in pinj.EXTRA_CELLS if c["id"] == "Npos_KV_once")
    assert kv_cell["side"] == "kv" and kv_cell["mechanism"] == "raw"
    ph_cell = next(c for c in pinj.EXTRA_CELLS if c["id"] == "phantom")
    assert ph_cell["mechanism"] == "phantom"


def test_smoke_cell_ids_are_valid_and_cover_both_mechanisms():
    by_id = {c["id"]: c for c in pinj.ALL_CELLS}
    assert set(pinj.SMOKE_CELL_IDS) <= set(by_id)
    mechanisms = {by_id[i]["mechanism"] for i in pinj.SMOKE_CELL_IDS}
    assert mechanisms == {"raw", "phantom"}


def test_resolve_cells_smoke_ignores_cells_arg():
    cells = pinj._resolve_cells("Npos_K_once", smoke=True)
    assert [c["id"] for c in cells] == pinj.SMOKE_CELL_IDS


def test_resolve_cells_all_returns_every_cell():
    cells = pinj._resolve_cells("all", smoke=False)
    assert len(cells) == 10
    cells_none = pinj._resolve_cells(None, smoke=False)
    assert len(cells_none) == 10


def test_resolve_cells_explicit_subset():
    cells = pinj._resolve_cells("1pos_K_once, phantom", smoke=False)
    assert [c["id"] for c in cells] == ["1pos_K_once", "phantom"]


def test_resolve_cells_unknown_id_raises():
    with pytest.raises(SystemExit):
        pinj._resolve_cells("not_a_real_cell", smoke=False)


# ===================================================================================================
# _scripted_conversation -- clamps to the antecedent's own follow-up list
# ===================================================================================================
def test_scripted_conversation_clamps_to_followups_length():
    setup, followups = pinj._scripted_conversation(999)
    assert setup == pinj.SETUP_USERS
    assert followups == pinj.FOLLOWUPS


def test_scripted_conversation_minimum_one_turn():
    _, followups = pinj._scripted_conversation(0)
    assert len(followups) == 1
    _, followups = pinj._scripted_conversation(-5)
    assert len(followups) == 1


def test_scripted_conversation_respects_requested_count():
    _, followups = pinj._scripted_conversation(3)
    assert followups == pinj.FOLLOWUPS[:3]


# ===================================================================================================
# wants_four_bit -- mirror_bench.py's convention, mirrored locally
# ===================================================================================================
@pytest.mark.parametrize("name, expected", [
    ("Qwen/Qwen2.5-7B-Instruct", True),
    ("google/gemma-2-9b-it", True),
    ("Qwen/Qwen2.5-1.5B-Instruct", False),
    ("Qwen/Qwen2.5-0.5B-Instruct", False),
])
def test_wants_four_bit_auto(name, expected):
    assert pinj.wants_four_bit(name, "auto") is expected


def test_wants_four_bit_override_yes_and_no():
    assert pinj.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "yes") is True
    assert pinj.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "no") is False


# ===================================================================================================
# _score_reply -- against a FAKE, duck-typed tokenizer (no real HF tokenizer needed)
# ===================================================================================================
class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return (text or "").split()


def test_score_reply_shape_and_consistency():
    text = "I am so happy and warm, this is wonderful! Great!"
    row = pinj._score_reply(FakeTok(), text)
    assert row["reply"] == text
    assert row["tokens"] == len(text.split())
    assert row["warmth"] >= 1                      # at least one WARM_MARKERS hit + '!' counts
    assert row["warmth_rate"] == pinj.warmth_rate(row["warmth"], row["tokens"])
    assert row["degenerate"] is False
    assert row["degenerate_reason"] == ""


def test_score_reply_flags_degenerate_repeat():
    text = "no no no really truly"
    row = pinj._score_reply(FakeTok(), text)
    assert row["degenerate"] is True
    assert row["degenerate_reason"] == "repeat-3gram"


def test_score_reply_empty_text_is_degenerate_and_safe():
    row = pinj._score_reply(FakeTok(), "")
    assert row["degenerate"] is True
    assert row["tokens"] == 0
    assert row["warmth_rate"] == 0.0


# ===================================================================================================
# _assemble_cell_result -- pure aggregation over fake per-turn rows (no model)
# ===================================================================================================
def _rows(rates):
    return [{"warmth_rate": r, "degenerate": False} for r in rates]


def test_assemble_cell_result_computes_deltas_and_verdict():
    cell = {"id": "test_cell"}
    warm = _rows([0.5, 0.4, 0.05])
    null = _rows([0.06, 0.05, 0.07])          # mean(|.|) noise_floor == 0.06
    base = _rows([0.0, 0.0, 0.0])
    out = pinj._assemble_cell_result(cell, warm, null, base)
    assert out["d_true"] == [0.5, 0.4, 0.05]
    assert out["d_null"] == [0.06, 0.05, 0.07]
    assert out["gate_passed"] is True          # |0.5| > 0.06
    assert out["turns_to_noise"] == 2          # first |d_true| <= 0.06 is index 2 (0.05)
    assert out["degenerate_turns_warm"] == []
    assert out["coherence_caution"] == ""


def test_assemble_cell_result_flags_degenerate_warm_turns():
    cell = {"id": "test_cell"}
    warm = [{"warmth_rate": 0.9, "degenerate": True}, {"warmth_rate": 0.8, "degenerate": False}]
    null = _rows([0.01, 0.01])
    base = _rows([0.0, 0.0])
    out = pinj._assemble_cell_result(cell, warm, null, base)
    assert out["degenerate_turns_warm"] == [0]
    assert "degenerate in the WARM branch" in out["coherence_caution"]


# ===================================================================================================
# CLI parsing (argparse only -- no run()/compare() invoked)
# ===================================================================================================
def test_argparser_defaults():
    a = pinj.build_argparser().parse_args([])
    assert a.model == "Qwen/Qwen2.5-7B-Instruct"
    assert a.four_bit == "auto"
    assert a.smoke is False
    assert a.cells == "all"
    assert a.turns == pinj.DEFAULT_TURNS
    assert a.dose == pinj.DEFAULT_DOSE
    assert a.layer is None
    assert a.axis == "warm"
    assert a.compare is None


def test_argparser_smoke_flag():
    a = pinj.build_argparser().parse_args(["--smoke", "--model", "google/gemma-2-9b-it"])
    assert a.smoke is True
    assert a.model == "google/gemma-2-9b-it"


def test_argparser_compare_takes_multiple_paths():
    a = pinj.build_argparser().parse_args(["--compare", "a.json", "b.json"])
    assert a.compare == ["a.json", "b.json"]


def test_argparser_cells_and_layer_override():
    a = pinj.build_argparser().parse_args(["--cells", "1pos_K_once,phantom", "--layer", "17"])
    assert a.cells == "1pos_K_once,phantom"
    assert a.layer == 17
