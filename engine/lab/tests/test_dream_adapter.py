"""Smoke tests for DreamAdapter against the real Dream 7B checkpoint (CPU, bf16).

Build-order step 2: verify one forward pass returns logits. These auto-skip when
torch or the cached checkpoint is unavailable, so the suite stays green anywhere.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from huggingface_hub import try_to_load_from_cache

from cloze_lab.models.base import KVState, LoadConfig, ModelAdapter

DREAM = "Dream-org/Dream-v0-Instruct-7B"

pytestmark = pytest.mark.checkpoint


@pytest.fixture(scope="module")
def adapter():
    if not isinstance(try_to_load_from_cache(DREAM, "config.json"), str):
        pytest.skip(f"{DREAM} not present in the local HF cache")
    from cloze_lab.models.dream import DreamAdapter

    return DreamAdapter(LoadConfig(model_id=DREAM, device="cpu", dtype="bfloat16"))


@pytest.fixture(scope="module")
def board(adapter) -> np.ndarray:
    prompt = adapter.encode("The capital of France is")
    return np.array(prompt + [adapter.config.mask_token_id] * 4)


def test_config_matches_checkpoint(adapter) -> None:
    assert isinstance(adapter, ModelAdapter)
    cfg = adapter.config
    assert cfg.vocab_size == 152064
    assert cfg.mask_token_id == 151666
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
    # Dream predicts position p from hidden state p-1 (the adapter owns the shift);
    # read the first masked slot of "The capital of France is ____" and expect Paris.
    n = len(board)
    first_mask = int(np.flatnonzero(board == adapter.config.mask_token_id)[0])
    result = adapter.forward(board, np.ones((n, n), dtype=bool), logits_for=[first_mask])
    top1 = int(result.logits[0].argmax())
    assert adapter.decode([top1]) == " Paris"


def test_chat_template_wraps_prompt(adapter) -> None:
    # chat=True wraps the prompt in the instruct template (so Dream-Instruct answers
    # instead of EOS-ing on a raw prompt); raw encode — the golden-fixture path — is
    # unchanged, which is why the dream_paris fixtures stay byte-identical.
    raw = adapter.encode("Count from one to five.")
    chat = adapter.encode("Count from one to five.", chat=True)
    assert len(chat) > len(raw)
    decoded = adapter.decode(chat)
    assert "<|im_start|>" in decoded and "assistant" in decoded


def test_attn_mask_actually_gates_attention(adapter, board) -> None:
    n = len(board)
    full = np.ones((n, n), dtype=bool)
    blinded = full.copy()
    blinded[n - 1, :4] = False  # last mask slot can no longer see the prompt start
    la = adapter.forward(board, full, logits_for=[n - 1]).logits
    lb = adapter.forward(board, blinded, logits_for=[n - 1]).logits
    assert not np.array_equal(la, lb)


def test_end_to_end_denoise_answers_paris(adapter) -> None:
    # Build-order step 4 milestone: the whole loop — adapter, sampling, quota
    # policy, events — denoises a real answer out of a real checkpoint.
    from cloze_lab.generate import GenerateConfig, generate

    result = generate(
        adapter,
        adapter.encode("The capital of France is"),
        GenerateConfig(max_new=4, steps=2),
    )
    assert "Paris" in result.text
    assert result.events[-1].reason in ("eos", "length")


def test_prefix_kv_reuse_is_exact(adapter, board) -> None:
    # Reusing a frozen prefix and recomputing a contiguous suffix must reproduce the
    # full-recompute picks exactly (the K/V of a frozen prefix is invariant under the
    # block-causal one-way law), with confidences within float epsilon.
    from cloze_lab.scheduler.blocks import attention_mask

    n = len(board)
    p = int(np.flatnonzero(board == adapter.config.mask_token_id)[0])  # first masked
    mask = attention_mask(n, prompt_len=p, block_len=n - p)  # one block, block-causal
    masked = [i for i in range(p, n)]

    full = adapter.forward(board, mask, logits_for=masked)
    # recompute only the suffix [p, n), reusing the prompt prefix from full.kv
    reused = adapter.forward(board, mask, kv=full.kv, recompute_kv=list(range(p, n)), logits_for=masked)
    assert np.array_equal(full.logits.argmax(-1), reused.logits.argmax(-1))  # picks exact
    assert np.allclose(full.logits, reused.logits, atol=0.5)  # confidences within bf16 epsilon


def test_scattered_recompute_rejected(adapter, board) -> None:
    # Tier C (non-contiguous recompute) is not expressible via an append-only cache.
    n = len(board)
    full = adapter.forward(board, np.ones((n, n), dtype=bool))
    with pytest.raises(NotImplementedError, match="contiguous-suffix"):
        adapter.forward(board, np.ones((n, n), dtype=bool), kv=full.kv, recompute_kv=[0, 2])


def test_returns_reusable_dream_kv(adapter, board) -> None:
    from cloze_lab.models.base import KVState
    from cloze_lab.models.dream import DreamKV

    n = len(board)
    result = adapter.forward(board, np.ones((n, n), dtype=bool))
    assert isinstance(result.kv, DreamKV) and isinstance(result.kv, KVState)
    assert result.kv.seq_len == n


@pytest.mark.gpu
def test_nf4_quantization_loads_on_gpu_and_predicts() -> None:
    # The §9 study knob: 4-bit weights fit a consumer GPU (~6 GB vs 15 GB bf16) and
    # still produce sane picks. GPU + bitsandbytes only, so gated behind the gpu mark.
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not isinstance(try_to_load_from_cache(DREAM, "config.json"), str):
        pytest.skip(f"{DREAM} not present in the local HF cache")
    from cloze_lab.models.dream import DreamAdapter

    q4 = DreamAdapter(LoadConfig(model_id=DREAM, device="cuda", dtype="bfloat16"), quantization="nf4")
    prompt = q4.encode("The capital of France is")
    board = np.array(prompt + [q4.config.mask_token_id] * 4)
    first = len(prompt)
    out = q4.forward(board, np.ones((len(board),) * 2, dtype=bool), logits_for=[first])
    assert q4.decode([int(out.logits[0].argmax())]) == " Paris"


def test_block_delta_cache_matches_off_end_to_end(adapter) -> None:
    # Core test: real-model KV reuse. In block mode, delta(full_refresh_every=1)
    # reuses the frozen prefix and recomputes each active block, exactly reproducing
    # the off (full-recompute) run — while reusing drawers (cache_hit > 0).
    from cloze_lab.generate import GenerateConfig, generate
    from cloze_lab.scheduler.cache import CacheConfig
    from cloze_lab.scheduler.events import StepStats

    cfg = GenerateConfig(max_new=8, steps=4, block_len=4)
    off = generate(adapter, adapter.encode("The capital of France is"), cfg, cache=CacheConfig(mode="off"))
    delta = generate(
        adapter, adapter.encode("The capital of France is"), cfg,
        cache=CacheConfig(mode="delta", full_refresh_every=1),
    )
    assert np.array_equal(off.board, delta.board)  # exact prefix caching
    assert max(e.cache_hit for e in delta.events if isinstance(e, StepStats)) > 0.0
