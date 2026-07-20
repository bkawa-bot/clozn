# scripts/tracer — the causal-tracer validation + SAE feature studies

Reproduction scripts for the numbers quoted in `notes/CIRCUIT_TRACER_DESIGN.md` §5b–§5e and in
`docs/BACKLOG.md` item 9. All of them drive a **live cloze-server** over HTTP; none of them need
torch (the one exception is the W_dec export, which is a one-shot in `engine/core/tools/`).

| script | what it measures | needs |
|---|---|---|
| `causal_trace_battery.py` | 16 prompts × 7 categories through `clozn.analysis.tracer`; aggregate S4 predicted-vs-observed scorecard, verdict spread, node strength tiers, interaction gaps | any AR model + a J-lens sidecar |
| `sae_fidelity_vs_concentration.py` | per prompt: **causal fidelity** of the SAE reconstruction (substitute `h → h_hat`, re-score) vs **concentration** (do a few features carry the site?) + explained variance | Qwen2.5-7B + `~/.clozn/sae/andyrdt_l15` incl. `w_dec.f16.bin` |
| `sae_joint_vs_random.py` | joint ablation of the top-k features vs a matched **random-k** control, k = 1…all — the control that decides "sparse circuit" vs "distributed" | same |
| `layer_position_map.py` | **run this first.** (layer × position) mean-ablation map — where causal mass actually lives, before assuming any artifact can see it | any AR model |
| `edge_depth_profile.py` | routed fraction into the final position by *capture depth* — the profile that exposes when a single routed number is meaningless (§5f) | any AR model |

## Running them

```bash
# engine, matched to the artifact you're testing
cloze-server ~/.clozn/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080 --gpu-layers 99 \
    --workers 2 --jlens ~/.clozn/artifacts/jlens/qwen2.5-7b-v1

python scripts/tracer/sae_fidelity_vs_concentration.py
```

Each writes a JSON alongside its console table.

## Four traps these scripts encode (each cost a wrong result first time)

1. **Position 0 is an attention sink.** Residual norm ~220× typical; the SAE emits 118 175 active
   features there and explained variance goes to −11 160. It is far out of distribution and will
   dominate any activation-ranked candidate list. `sae_fidelity_vs_concentration.py` excludes it
   explicitly — an earlier pass that didn't found nothing but the token `'In'`.
2. **Sum-of-singles ≠ joint ablation.** Interaction gaps on this stack run ≈ −60% (and −73% on the
   7B), so adding up single-feature deltas badly understates a set's joint effect. The first pass
   reported "top-12 features = 6.3% of the site" by summing; the joint arms in
   `sae_joint_vs_random.py` put the same quantity at 10–45%. Always run the joint arm.
3. **A multi-token entity's causal mass sits at its LAST tokens.** "Zorblax" spans positions 9–12;
   position 9 carries +0.12 and position 12 carries **+8.58**. Under causal attention only the
   final tokens have seen the whole name. Scan the whole span — `layer_position_map.py` does.
4. **A cross-position `routed_fraction` is a lower bound, not a measurement.** Single-site patching
   can't hold the path (the source re-supplies it downstream), and holding the whole destination
   column doesn't fix it either, because the last layer materializes only the logit rows
   (`inp_out_ids`) and is unpatchable. Late-layer sources read a flat 0.0% at every depth — which
   is physically impossible, and is the tell. Same-column edges are unaffected. See §5f.

## Headline results (Qwen2.5-7B, andyrdt_l15, 2026-07-20)

- Causal-trace S4 scorecard: **88/96 = 91.7%** correct flip predictions (9B, 16 prompts).
- SAE reconstruction preserves **~99.9% of causal content** while capturing only **57% of variance**
  — variance and function dissociate.
- **No single feature is load-bearing** (~0.5% of a site each, ~47 active), but **top-8/16 jointly
  carry 10–45%** vs 0–6% for matched random-k: real feature-level structure, not sparse.

Scope: one SAE, one layer, one model, a handful of prompts. These are measurements of *this
dictionary at layer 15 of this model*, not general claims about SAEs or circuits.
