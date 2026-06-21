"""Smoke tests for open-dCoder 0.5B — the tiny Dream-family checkpoint CI runs on CPU.

DESIGN: "CI must run on CPU with a tiny checkpoint (open-dCoder 0.5B class)." open-dCoder
is a masked diffusion model distilled into a stock ``Qwen2ForCausalLM``: same shifted head
as Dream, loaded via the ``open_dcoder_adapter`` factory. These auto-skip when torch or the
checkpoint is absent, and they pin the exact-KV-reuse behaviour the shipped mask fix enables
(stock Qwen2 reads ``attention_mask=None`` + a KV cache as *causal* — the adapter never
sends None, so prefix reuse stays bidirectional and exact).
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from huggingface_hub import try_to_load_from_cache

from cloze_lab.models.base import KVState, LoadConfig, ModelAdapter
from cloze_lab.models.dream import OPEN_DCODER_05B

pytestmark = pytest.mark.checkpoint


@pytest.fixture(scope="module")
def adapter():
    if not isinstance(try_to_load_from_cache(OPEN_DCODER_05B, "config.json"), str):
        pytest.skip(f"{OPEN_DCODER_05B} not present in the local HF cache")
    from cloze_lab.models.dream import open_dcoder_adapter

    return open_dcoder_adapter(LoadConfig(model_id=OPEN_DCODER_05B, device="cpu", dtype="bfloat16"))


@pytest.fixture(scope="module")
def board(adapter) -> np.ndarray:
    prompt = adapter.encode("def add(a, b):\n    return a +")
    return np.array(prompt + [adapter.config.mask_token_id] * 4)


def test_config_matches_checkpoint(adapter) -> None:
    assert isinstance(adapter, ModelAdapter)
    cfg = adapter.config
    assert cfg.vocab_size == 151936
    assert cfg.mask_token_id == 151665  # <M>, lives in added_tokens.json, not the config
    assert cfg.eos_token_id == 151643


def test_one_forward_pass_returns_logits(adapter, board) -> None:
    n = len(board)
    masked = [i for i in range(n) if board[i] == adapter.config.mask_token_id]
    result = adapter.forward(board, np.ones((n, n), dtype=bool), logits_for=masked)
    assert result.logits.shape == (len(masked), adapter.config.vocab_size)
    assert result.logits.dtype == np.float32
    assert np.isfinite(result.logits).all()
    assert isinstance(result.kv, KVState)
    assert result.kv.seq_len == n


def test_shifted_head_yields_meaningful_prediction(adapter, board) -> None:
    # Dream-family shifted head: position p is predicted from hidden state p-1. The
    # first masked slot of "return a + ____" is the obvious completion, " b".
    n = len(board)
    first_mask = int(np.flatnonzero(board == adapter.config.mask_token_id)[0])
    result = adapter.forward(board, np.ones((n, n), dtype=bool), logits_for=[first_mask])
    top1 = int(result.logits[0].argmax())
    assert adapter.decode([top1]) == " b"


def test_attn_mask_actually_gates_attention(adapter, board) -> None:
    n = len(board)
    full = np.ones((n, n), dtype=bool)
    blinded = full.copy()
    blinded[n - 1, :4] = False  # last mask slot can no longer see the prompt start
    la = adapter.forward(board, full, logits_for=[n - 1]).logits
    lb = adapter.forward(board, blinded, logits_for=[n - 1]).logits
    assert not np.array_equal(la, lb)


def test_prefix_kv_reuse_is_exact(adapter, board) -> None:
    # The mask fix in action: reusing a frozen prefix and recomputing a contiguous
    # suffix reproduces the full-recompute picks exactly. (Stock Qwen2 would go causal
    # on a None mask + KV cache; the adapter always sends an explicit bidirectional
    # mask, so reuse stays bitwise exact here — tighter than Dream's bf16 epsilon.)
    from cloze_lab.scheduler.blocks import attention_mask

    n = len(board)
    p = int(np.flatnonzero(board == adapter.config.mask_token_id)[0])  # first masked
    mask = attention_mask(n, prompt_len=p, block_len=n - p)  # one block, block-causal
    masked = list(range(p, n))

    full = adapter.forward(board, mask, logits_for=masked)
    reused = adapter.forward(board, mask, kv=full.kv, recompute_kv=list(range(p, n)), logits_for=masked)
    assert np.array_equal(full.logits.argmax(-1), reused.logits.argmax(-1))  # picks exact
    assert np.allclose(full.logits, reused.logits, atol=0.5)  # confidences within bf16 epsilon


def test_end_to_end_denoise_completes(adapter) -> None:
    from cloze_lab.generate import GenerateConfig, generate

    result = generate(
        adapter,
        adapter.encode("def add(a, b):\n    return a +"),
        GenerateConfig(max_new=4, steps=2),
    )
    assert result.text.startswith(" b")
    assert result.events[-1].reason in ("eos", "length")


def test_block_delta_cache_matches_off_end_to_end(adapter) -> None:
    # Real-model KV reuse on the CI checkpoint: delta(full_refresh_every=1) reuses the
    # frozen prefix and recomputes each active block, exactly reproducing the off
    # (full-recompute) run while reusing drawers (cache_hit > 0).
    from cloze_lab.generate import GenerateConfig, generate
    from cloze_lab.scheduler.cache import CacheConfig
    from cloze_lab.scheduler.events import StepStats

    cfg = GenerateConfig(max_new=8, steps=4, block_len=4)
    prompt = adapter.encode("def add(a, b):\n    return a +")
    off = generate(adapter, prompt, cfg, cache=CacheConfig(mode="off"))
    delta = generate(
        adapter, prompt, cfg,
        cache=CacheConfig(mode="delta", full_refresh_every=1),
    )
    assert np.array_equal(off.board, delta.board)  # exact prefix caching
    assert max(e.cache_hit for e in delta.events if isinstance(e, StepStats)) > 0.0
