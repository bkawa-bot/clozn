"""Smoke tests for LLaDAAdapter against the real LLaDA 8B checkpoint (CPU, bf16).

The second real family (DESIGN §4.4 target #2): validates that the ModelAdapter
seam generalizes past Dream. Auto-skip when torch or the checkpoint is absent.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from huggingface_hub import try_to_load_from_cache

from cloze_lab.models.base import KVState, LoadConfig, ModelAdapter
from cloze_lab.models.llada import LLADA_8B_INSTRUCT

pytestmark = pytest.mark.checkpoint


@pytest.fixture(scope="module")
def adapter():
    if not isinstance(try_to_load_from_cache(LLADA_8B_INSTRUCT, "config.json"), str):
        pytest.skip(f"{LLADA_8B_INSTRUCT} not present in the local HF cache")
    from cloze_lab.models.llada import LLaDAAdapter

    return LLaDAAdapter(LoadConfig(model_id=LLADA_8B_INSTRUCT, device="cpu", dtype="bfloat16"))


@pytest.fixture(scope="module")
def board(adapter) -> np.ndarray:
    prompt = adapter.encode("The capital of France is")
    return np.array(prompt + [adapter.config.mask_token_id] * 4)


def test_config_matches_checkpoint(adapter) -> None:
    assert isinstance(adapter, ModelAdapter)
    cfg = adapter.config
    assert cfg.vocab_size == 126464
    assert cfg.mask_token_id == 126336  # convention, not on the tokenizer
    assert cfg.eos_token_id == 126081


def test_one_forward_pass_returns_logits(adapter, board) -> None:
    n = len(board)
    masked = [i for i in range(n) if board[i] == adapter.config.mask_token_id]
    result = adapter.forward(board, np.ones((n, n), dtype=bool), logits_for=masked)
    assert result.logits.shape == (len(masked), adapter.config.vocab_size)
    assert result.logits.dtype == np.float32
    assert np.isfinite(result.logits).all()
    assert isinstance(result.kv, KVState)
    assert result.kv.seq_len == n


def test_in_place_head_predicts_paris(adapter, board) -> None:
    # LLaDA predicts the token AT each masked slot (no AR shift, unlike Dream): the
    # first masked slot of "The capital of France is ____" is Paris.
    first_mask = int(np.flatnonzero(board == adapter.config.mask_token_id)[0])
    result = adapter.forward(board, np.ones((len(board),) * 2, dtype=bool), logits_for=[first_mask])
    assert adapter.decode([int(result.logits[0].argmax())]) == " Paris"


def test_attention_bias_gates_attention(adapter, board) -> None:
    n = len(board)
    full = np.ones((n, n), dtype=bool)
    blinded = full.copy()
    blinded[n - 1, :4] = False  # last mask slot can no longer see the prompt start
    la = adapter.forward(board, full, logits_for=[n - 1]).logits
    lb = adapter.forward(board, blinded, logits_for=[n - 1]).logits
    assert not np.array_equal(la, lb)


def test_kv_reuse_raises(adapter, board) -> None:
    n = len(board)
    full = np.ones((n, n), dtype=bool)
    fresh = adapter.forward(board, full)
    with pytest.raises(NotImplementedError, match="recomputes every pass"):
        adapter.forward(board, full, kv=fresh.kv)
    with pytest.raises(NotImplementedError, match="recomputes every pass"):
        adapter.forward(board, full, recompute_kv=[0])


def test_end_to_end_denoise_answers_paris(adapter) -> None:
    from cloze_lab.generate import GenerateConfig, generate

    result = generate(
        adapter, adapter.encode("The capital of France is"),
        GenerateConfig(max_new=8, steps=4),
    )
    assert "Paris" in result.text
    assert result.events[-1].reason in ("eos", "length")
