# kernels/ — GPU kernels

Optional, validated against a CPU reference. Today: the **confidence-select** kernel
(fused sample/confidence/top-k) that moves the diffusion commit step on-device.

Tomorrow (Roadmap phase 3): **interp kernels** — sparse top-k for SAE/transcoder inference,
and batched on-device activation harvesting. The confidence-select top-k is the seed: SAE
inference is `encoder matmul → top-k sparsify → decoder matmul`, and we already have the
top-k. Repointing it at the feature dimension is the bridge from diffusion sampling to
interpretability at scale.
