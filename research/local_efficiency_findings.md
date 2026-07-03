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

## Phase 1 — the white-box tax, measured (GPU, live `cloze-server`)

`research/bench_whitebox_tax.py` against three server configurations: `:8080` Llama-3.2-1B Q4
(plain boot), `:8081` Qwen2.5-7B Q4 (plain boot), `:8081` Qwen2.5-7B Q4 rebooted with
`--sae ~/.clozn/sae/andyrdt_l15`. Same fixed prompt, greedy, `max_tokens=128`, median of 5 reps
(1 warmup discarded). Numbers are the engine's own `GenFinished` receipt (`tok_per_s`), so they
exclude client/HTTP noise but include per-event serialization cost (on_event runs inline in the
decode loop — that IS the tax being measured). Receipts:
`research/runs/whitebox_tax_{llama1b,qwen7b_plain,qwen7b_sae}.json`.

| server | config | tok/s | vs plain | SSE bytes/req |
|---|---|---|---|---|
| Llama-1B (plain) | plain | 434.0 | 100.0% | 49.9 KiB |
| Llama-1B (plain) | feat-legacy | 323.9 | 74.6% | 5849.0 KiB |
| Llama-1B (plain) | feat-protocol | 419.1 | 96.6% | 181.2 KiB |
| Llama-1B (plain) | state-full | 409.2 | 94.3% | 1555.5 KiB |
| Qwen-7B (plain boot) | plain | 138.2 | 100.0% | 50.1 KiB |
| Qwen-7B (plain boot) | feat-legacy | 117.7 | 85.2% | 9861.6 KiB |
| Qwen-7B (plain boot) | feat-protocol | 136.1 | 98.4% | 181.8 KiB |
| Qwen-7B (plain boot) | state-full | 135.0 | 97.7% | 2580.1 KiB |
| Qwen-7B (**`--sae`** boot) | plain | 139.9 | 100.0% | 50.1 KiB |
| Qwen-7B (**`--sae`** boot) | feat-legacy | 78.2 | 55.9% | 9723.3 KiB |
| Qwen-7B (**`--sae`** boot) | feat-protocol | 85.7 | 61.2% | 410.8 KiB |
| Qwen-7B (**`--sae`** boot) | state-full | 84.8 | 60.6% | 2809.2 KiB |

**Findings, measured not estimated:**

1. **The legacy-SSE tax is real and large, and it is a wire-format tax, not a compute tax** — at
   1B, `feat-legacy` costs 25.4% of throughput and ships **117x** the bytes of `plain` (5849 vs 50
   KiB) for the SAME activations `feat-protocol` ships in 181 KiB while costing only 3.4%. This
   confirms the Phase-0 prediction exactly: raw-decimal-JSON serialization of activation floats is
   pure overhead with a already-shipping fix (protocol mode). At 7B the same pattern holds (85.2%
   vs 98.4%) but the gap is proportionally smaller — decode is slower per-token at 7B, so the fixed
   per-event serialization cost is a smaller fraction of a longer per-token budget.
2. **Lens + confidence really are close to free**, as predicted: `feat-protocol` (light state,
   tap + probes + lens, no raw activations on the wire) costs only 3.4% at 1B and 1.6% at 7B vs
   plain. `state-full` (protocol + base64 activation tensor per token) costs almost the same as
   `feat-protocol` (5.7% / 2.3%) — the base64 tensor payload is not the bottleneck once the JSON
   text tax is removed.
3. **The SAE tax is the single largest, most decisive number in this table.** Comparing
   `feat-protocol` across Qwen-7B's two boots: 98.4% (plain-boot server) -> **61.2%** (`--sae`-boot
   server) — SAE encoding costs roughly **37 percentage points** of throughput on every featureful
   request once active, dwarfing the wire-format tax it sits beside. `feat-legacy` under `--sae`
   drops to 55.9% (vs 85.2% plain-boot) — the SAE tax and the wire tax are close to additive, not
   masking each other. This is a MEASURED number superseding the memory-file's previous "~9 ms/
   readout" estimate; the two pre-identified code fixes (vectorized GEMV loads, hoisted
   `sae_topk` workspace, both `research/local_efficiency_findings.md` Phase-0 above) remain
   unapplied and would attack exactly this number.
4. Both featureful boots' `plain` row (no features requested) costs the same as the truly-plain
   boot (138.2 vs 139.9 tok/s at 7B, within rep noise) — confirms `--sae` imposes ZERO tax on
   requests that don't ask for features, i.e., the SAE encoder path is fully gated behind the
   `features`/`state` request flags as designed, not an always-on background cost.

## Phase 2 — the batched-receipts proof (GPU, direct PyTorch)

`research/bench_batched_receipts.py`: the with/without-memory-block pair generated in ONE batched
`model.generate()` call (batch=2, left-padded) vs two separate batch=1 calls, plus a batch=4 "N
receipts in one call" extrapolation. Same fixed question, greedy, forced length (128 new tokens),
CUDA-synchronized walls, median of 5 reps (1 warmup discarded). Models mirror the studio's real
configs: Qwen2.5-1.5B-Instruct **bf16**, Qwen2.5-7B-Instruct **nf4**. Receipt:
`research/runs/batched_receipts.json`.

| model | batch=1 wall | batch=2 wall | batch=4 wall | batch-2/1 ratio | batch-4/1 ratio | verdict |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B bf16 | 2.47 s | 2.55 s | 2.48 s | **1.031** | 1.003 | <=1.3: **CONFIRMED** |
| Qwen2.5-7B nf4 | 3.18 s | 5.80 s | 5.91 s | **1.821** | 1.854 | >1.3: **NOT confirmed** |

**The headline result is scale/quantization-dependent, and the split is informative, not just a
pass/fail.** At 1.5B bf16 the differential-receipt idea works exactly as hypothesized: batch=2
costs 3.1% more wall-clock than batch=1 (essentially free — memory-bandwidth-bound decode holds),
and batch=4 costs 0.3% more (four generations for the price of one). Per-rep variance is tight at
every batch size (all 5 reps within ~2.44-2.76s for batch=1), so this is a clean, repeatable
result, not noise.

At 7B nf4 the same idea does NOT confirm: batch=2 costs **82% more** wall-clock than batch=1. The
shape of the failure is itself informative and was eyeballed (house rule) rather than taken as a
single ratio: walls are tight WITHIN each batch size (batch=1: 3.15-3.21s; batch=2: 5.80-5.80s,
essentially variance-free) but there is a large, discrete step from batch=1 to batch=2 — and then
**batch=4 costs almost nothing more than batch=2** (5.91 vs 5.80s, ratio 1.02). That "jump once,
then flat" shape (not a smooth per-sequence scaling law) is consistent with a batch-size-dependent
kernel-dispatch or dequantization-path change in the nf4 execution graph rather than a genuinely
bandwidth-bound regime breaking down gradually — ESTIMATED explanation, not verified by profiling
here; the measured fact is just the shape of the three numbers.

**Practical reading:** the differential-receipt idea (batch the with/without-memory pair into one
call) is a real, near-zero-cost win **for bf16 models at this scale**, and genuinely does NOT pay
off as cheaply for the 7B nf4 config the studio actually runs — quantization and/or scale confound
here (same open confound the memory files flag for slotmem_qwen: "nf4 vs scale confounded, bf16-7B
doesn't fit 16GB" — this bench inherits that same limitation). Wiring batched ablation into
`replay.py` would need its OWN dose-response receipt at the studio's real model before shipping,
not an assumption that the 1.5B result transfers.

## Recommendations (ranked by the numbers above, not by vibe)

1. **Ship the SAE encoder optimizations before shipping SAE-on-by-default.** The measured 37-point
   throughput hit (98.4% -> 61.2% at `feat-protocol`) is the single biggest number in this whole
   investigation and the one with a known, already-scoped fix (vectorize `sae_topk`'s GEMV loads,
   hoist its per-call `cudaMalloc` into the persistent workspace) — do this before any product
   surface turns SAE readouts on by default.
2. **Legacy SSE mode should be deprecated or its use discouraged for streaming featureful
   requests** — `feat-protocol`/`state:"light"` gets nearly all the same information (tap +
   probes + lens; base64 tensor available via `state:"full"` at almost no extra cost) for a
   fraction of the wire bytes and throughput cost. There is no measured case where legacy-SSE
   featureful beats protocol mode.
3. **Batched differential receipts are a real, close-to-free lever for the smaller/bf16 tier**
   (1.5B) but need their OWN measurement, not an assumption of transfer, before being wired into
   `replay.py` for the studio's actual 7B nf4 hot path — the 82% cost at 7B nf4 means a naive port
   would make every receipt materially slower, the opposite of the intended win.
4. Zero-copy device logits (skip the full-vocab D2H every AR token) remains the single largest
   UNMEASURED structural lever from Phase 0 — it's implemented and dark twice over (passthrough=
   false at pool construction; AR path never calls the code path that would use it). Worth a
   dedicated measurement pass since it is architecturally distinct from every number in this doc
   (host-copy elimination, not a feature/format cost).
5. KV-cache quantization and cross-request prompt caching remain correctly deprioritized by Phase
   0's analysis — they target memory/TTFT, not the batch-1 decode-rate numbers this investigation
   measured, and nothing in Phase 1/2 changes that assessment.
