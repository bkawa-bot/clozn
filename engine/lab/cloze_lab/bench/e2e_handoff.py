"""End-to-end handoff share: in a real GPU generation, how much of each denoise step is
the logits host-transfer + CPU sampling that the confidence-select kernel removes, vs the
model forward? This is the honest end-to-end framing of the kernel's value.

The kernel's win is structural (move 2*n_masked values instead of n_masked*vocab) and the
isolated step is ~2.6x faster on the 5080 (see kernels/confidence_select bench). But how
much that helps *end to end* depends entirely on the forward's share of wall time. This
measures it on a big model (Dream 7B nf4, forward-dominated) and a small one (open-dCoder
0.5B, forward cheap) so the range is visible, not assumed.

    python -m cloze_lab.bench.e2e_handoff            # needs a CUDA GPU + cached checkpoints

Honest by construction: the forward time is measured as (mean step wall - handoff), and the
handoff (D2H transfer of the per-step logits + the CPU sample_candidates) is timed at the
real per-step scale. "projected tok/s" assumes the kernel drives the handoff to ~0; the real
kernel costs a little (the isolated bench), so treat it as an upper bound on the e2e gain.
"""

from __future__ import annotations

import time

import numpy as np

from cloze_lab.bench.speed import speed_stats
from cloze_lab.generate import GenerateConfig, generate, sample_candidates
from cloze_lab.models.base import LoadConfig
from cloze_lab.scheduler.cache import CacheConfig


def _time_handoff(vocab: int, n_masked: int, device: str, iters: int = 50) -> tuple[float, float]:
    """Per-step (D2H-transfer-ms, cpu-sample-ms) at the real logits scale [n_masked, vocab]."""
    sample_in = np.random.randn(n_masked, vocab).astype(np.float32)
    positions = list(range(n_masked))
    transfer_ms = 0.0
    if device == "cuda":
        import torch

        g = torch.randn(n_masked, vocab, device="cuda", dtype=torch.float32)
        for _ in range(5):
            _ = g.float().cpu().numpy()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = g.float().cpu().numpy()
        torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - t0) / iters * 1000.0
    t0 = time.perf_counter()
    for _ in range(iters):
        sample_candidates(sample_in, positions)
    sample_ms = (time.perf_counter() - t0) / iters * 1000.0
    return transfer_ms, sample_ms


def measure(label: str, adapter, prompt: str, cfg: GenerateConfig, device: str, chat: bool) -> None:
    ids = adapter.encode(prompt, chat=chat) if (chat and adapter.config.family.value == "dream") else adapter.encode(prompt)
    result = generate(adapter, ids, cfg, cache=CacheConfig(mode="off"))
    s = speed_stats(result)
    wall_ms = s.wall_ms
    fwd = s.forwards
    tokens = s.new_tokens
    n_masked = cfg.block_len if cfg.block_len > 0 else cfg.max_new
    transfer_ms, sample_ms = _time_handoff(adapter.config.vocab_size, n_masked, device)
    handoff_ms = transfer_ms + sample_ms
    step_ms = wall_ms / fwd if fwd else 0.0
    forward_ms = max(step_ms - handoff_ms, 0.0)
    handoff_total = handoff_ms * fwd
    share = handoff_total / wall_ms * 100.0 if wall_ms else 0.0
    proj_wall = max(wall_ms - handoff_total, 1e-9)
    proj_tok_s = tokens / (proj_wall / 1000.0)
    print(
        f"{label:18} fwd={fwd:3d} tok={tokens:3d} wall={wall_ms:7.0f}ms  tok/s={s.tok_per_s:6.1f}\n"
        f"{'':18} per step: forward~{forward_ms:6.1f}ms | handoff {handoff_ms:5.2f}ms "
        f"(transfer {transfer_ms:.2f} + sample {sample_ms:.2f})  [n_masked={n_masked}, vocab={adapter.config.vocab_size}]\n"
        f"{'':18} handoff = {share:4.1f}% of wall  ->  projected tok/s if handoff removed: {proj_tok_s:6.1f}\n"
    )


def main() -> None:
    from cloze_lab.models.dream import (
        DREAM_7B_INSTRUCT,
        OPEN_DCODER_05B,
        DreamAdapter,
        open_dcoder_adapter,
    )

    cfg = GenerateConfig(max_new=48, steps=4, block_len=8)
    print("== end-to-end handoff share (CUDA) ==\n")

    dcoder = open_dcoder_adapter(LoadConfig(model_id=OPEN_DCODER_05B, device="cuda", dtype="bfloat16"))
    measure("open-dCoder 0.5B", dcoder, "def add(a, b): return a +", cfg, "cuda", chat=False)
    del dcoder

    dream = DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"), quantization="nf4")
    measure("Dream 7B (nf4)", dream, "Explain what a diffusion language model is in three sentences.", cfg, "cuda", chat=True)


if __name__ == "__main__":
    main()
