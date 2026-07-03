# Local-engine efficiency findings — the white-box-tax investigation

2026-07-02. Measurement-first pass over the C++/ggml serving stack (`engine/core`) + the PyTorch
research substrate, asking one question: **what does observability actually cost, and which
efficiency levers are already under our feet unused?** Thesis under test: the white-box tax
(receipts / taps / SAE readouts) can be driven toward ~zero — black-box speed with glass walls.

Status legend: **MEASURED** = a number from a run in this doc; **ESTIMATED** = arithmetic from
specs or code reading, no run behind it. Estimates are labeled loudly; trust the measured column.

---

## Phase 0 — the lever map (CPU recon, code-verified)

Where each efficiency lever lives, and whether the current stack actually pulls it.
"Wired" = reachable in the live serve path today. "Dark" = present in the linked llama.cpp/API
or in our own tree, but not plumbed to where it would pay.

### Levers already WIRED

| lever | where | state |
|---|---|---|
| GPU weight offload | `cloze_server --gpu-layers` (CLI passes 99 whenever the GPU build is picked) | wired, on |
| Incremental KV decode (AR) | `generate_ar.cpp`: prefill once, then one `llama_decode` per token at `n_past` | wired, on |
| Flash attention | llama default `LLAMA_FLASH_ATTN_TYPE_AUTO` → `cparams.flash_attn=true, auto_fa=true` (llama-context.cpp:197-198); nothing in cloze disables it | wired **by default, silently** — no flag, resolution not surfaced in any log we keep (quiet_log eats INFO) |
| Context pool (concurrent requests) | `cloze_server --workers N` → N contexts over ONE weight copy (`ContextPool`) | wired but **dark in practice**: default 1, `clozn_cli serve` never passes it |
| Diffusion KV reuse (Tier A/B exact prefix, Tier C delta) | request param `cache:"delta"`, `cache.hpp/generate.cpp` | wired, **off by default per request**; diffusion-only — the AR path (all current studio traffic) never touches it |
| Zero-copy device logits (skip full-vocab D2H) | `device_logits_passthrough` + our own llama patch `llama_set_skip_raw_logits` (llama.h:1017) + `confidence_select` CUDA kernel | **implemented but dark twice over**: (a) the server constructs every pooled adapter with passthrough=false (cloze_server.cpp `ContextPool` → `GgmlAdapter(model, n_ctx)`); (b) it only exists on the diffusion `forward()` path — `ar_forward` unconditionally host-copies the FULL vocab logits every token (`model_ggml.cpp:214-215`) |
| On-device SAE readout | `--sae <dir>` (JumpReLU encode + `sae_topk`, ~0.94 GB fp16 device) | wired behind a manual flag; CLI never passes it |
| Throughput self-report | `GenFinished{wall_ms, tok_per_s}` rides every stream (legacy SSE + protocol `kind:"end"`) | wired — the stream is its own benchmark instrument |

### Levers DARK (exist upstream, unplumbed)

| lever | upstream surface | what plumbing would take | expected win (ESTIMATED) |
|---|---|---|---|
| KV-cache quantization | `llama_context_params.type_k/type_v` (default F16; q8_0/q4_* available, [EXPERIMENTAL]) | ~10 lines: two server flags → `cp.type_k/type_v` in `GgmlAdapter::init_context` | KV VRAM ÷2 (q8) to ÷4 (q4). Qwen-7B @ 4096 ctx ≈ 229 MB/context F16 → ~115/57 MB. Pays when `--workers`>1 or long ctx; ~neutral for batch-1 speed |
| Cross-request prompt/prefix caching | `llama_state_seq_get_data/set_data`, `llama_memory_seq_cp` — all in the linked llama.h | per-worker keyed prefix cache + `ar_forward` reuse instead of the unconditional fresh-KV re-prefill (`set_causal(true)` clears memory every request) | TTFT only (decode rate unchanged). Today every studio turn re-forwards system prompt + 16-vector memory prefix + history from scratch |
| Multi-sequence batched decode | `n_seq_max`, per-token `seq_id` in `llama_batch` (default n_seq_max=1; we always use seq 0) | real change: batch scheduler in the serve layer. The `--workers` pool gives N-way concurrency at N× KV and batch-1 GPU efficiency per stream | decode is memory-bandwidth-bound at batch-1, so batch-2 is nearly free **if** the substrate behaves — Phase 2 measures exactly this in PyTorch |
| Speculative / draft decoding | vendored llama.cpp has the machinery (common/speculative, examples) — nothing exported through our adapter | substantial cloze-side work; and NOTE: Llama-1B **cannot** draft for Qwen-7B (different vocab) — would need Qwen2.5-0.5B as draft | typical 1.5-2.5× decode at batch-1 (literature; unmeasured here) |
| `n_threads` tuning | `cp.n_threads` left at GGML default | trivial | ~nil when fully offloaded to GPU |

### The white-box tax surface (what a receipt costs per token, AR path, code-level)

Always paid, even with `features:false` (the current serve floor):
- full-vocab logits D2H copy per token: Qwen-7B vocab 152k × 4 B ≈ **608 KB/token** (ESTIMATED from dims); Llama-1B 128k ≈ 512 KB/token
- host argmax over vocab (`sample_candidates`) + **logit-lens top-5 `partial_sort` per token — unconditional** in `generate_ar.cpp:80` (there is no way to turn lens off in AR mode)
- every event also accumulates into `result.events` (RAM only, no wire cost when non-stream)

Added by `features:true`:
- `llama_set_embeddings(true)` + eval-callback tap: synchronous `ggml_backend_tensor_get` of n_embd f32/token (7B: 14 KB/token D2H)
- concept-probe projections on host (K≈6 dirs × n_embd MACs/token — CPU, trivial)
- `StepActivations` copy (n_embd floats)
- **legacy SSE only**: the raw activations are serialized as *decimal text JSON* — ~30-45 KB of text per token at 7B (`events.cpp:139-150`). Protocol mode `state:"light"` holds them back; `state:"full"` sends base64. So the legacy featureful stream pays a pure wire/format tax that protocol-light does not.

Added by `--sae` (per featureful token): H2D 14 KB + prep kernel + GEMV streaming the 0.94 GB
`W_enc` through the SMs + `sae_topk` + D2H 128 B. Memory-file figure: **~9 ms/readout (previously
measured)**. Two pre-identified fixes, both confirmed present in code:
1. `encode_jumprelu_kernel` (`engine/core/src/sae_encoder.cu:69-73`) loads `W_enc` and `xh` as
   *scalar* `__half` (2 B/lane) — vectorizing to `__half2`/16-B loads is the standard
   bandwidth fix. GEMV floor on this card ≈ 0.94 GB ÷ ~900 GB/s ≈ **1.0-1.3 ms** (ESTIMATED),
   so ~9 ms has real headroom.
2. `kernels/sae_topk/sae_topk.cu:162-171` does `cudaMalloc`(rows×131072 B) + forced
   `cudaStreamSynchronize` + `cudaFree` **on every call** — the file's own comment says "a real
   integration would hoist this to a persistent workspace." The encoder Impl already keeps a
   grow-only workspace (`sae_encoder.cu reserve()`); the mask belongs there.

### Recon verdicts (pre-measurement)

1. The serve path's biggest *structural* inefficiency is not the white-box machinery — it is
   that `ar_forward` copies the whole vocab to host every token while a device-side
   greedy/select kernel + skip-raw-logits patch already exist in-tree, unwired for AR.
2. KV quantization is a 10-line flag away and only matters for memory (workers/ctx), not batch-1 speed.
3. Prompt caching is the TTFT lever the studio actually feels (its memory prefix re-forwards every turn).
4. The SAE tax (~9 ms/token when on) is the one *measured* white-box cost with a known 3-9× reduction path.
5. Lens + confidence are effectively free-riding on copies the server already makes; their tax
   should measure near zero. The heavy taxes to verify: features tap (sync D2H each token),
   legacy-SSE activation JSON, SAE encode.

*(Phase 1 baseline table, Phase 2 batched-receipts proof, and recommendations follow below.)*

---

## Phase 1 — baseline numbers (GPU)

*(pending — GPU occupied by a concurrent 7B consolidation A/B; will fill when free)*

## Phase 2 — the batched-receipts proof

*(pending)*

## Recommendations

*(pending final numbers)*
