# CLAUDE.md — legible-interior

A research project on **compression under constraint** and the **interpretability tax**:
does forcing an internal representation to be *legible* (sparse) cost capability — and,
the load-bearing question, are sparse units actually *interpretable*?

**Read `HANDOFF.md` first — it is the source of truth.** It carries the full thesis, the
landscape (Titans, Nested Learning, Coconut, Huginn), the experiment already run (the
top-k bottleneck "interpretability tax" curve), and the ranked next rungs. This file only
adds local facts the handoff can't know.

## Your first task — the open crux (do this before any scale-up)

From HANDOFF.md's "⚠️ THE open crux": **sparse ≠ interpretable.** The prior run showed the
code *can* be sparse cheaply (a free plateau down to ~8 active units, a cliff only at k=1),
but never checked whether those units *mean* anything nameable.

1. Reproduce the tiny char model (tinyshakespeare, 8-char context, top-k bottleneck).
2. Take the **k=8** model. For each bottleneck unit, find its max-activating contexts.
   Are they human-describable ("fires after a space", "tracks an open quote")?
3. Quantify: predict each unit's activation from simple interpretable features; report the
   variance explained. Sparse-and-meaningful → the bet survives a real test.
   Sparse-but-inscrutable → the dream is in trouble, and that's the most important result.

## Local environment (this machine)

- GPU: **RTX 5080, 16 GB** (sm_120). CUDA 13.3 toolkit installed. Plenty for the toy
  models in the handoff and for putting a real model (GPT-2 / Pythia-70M / TinyStories)
  on the bits/char ladder (next rung).
- OS: Windows 11; default shell is PowerShell. A POSIX `bash` is also available.
- HuggingFace downloads on this PC need `HF_HUB_DISABLE_SYMLINKS=1` (they crash otherwise,
  WinError 1314).
- A working PyTorch + CUDA playground already exists next door at
  `C:\Users\brigi\src\cloze\lab` (torch, transformers, numpy) — mirror its setup or make a
  fresh venv here.

## Ethos (from the handoff — keep it)

Be ruthlessly honest about what each result does and does **not** show. State the caveats
louder than the wins. The number from an experiment matters far less than whether it
green-lights or kills the next rung. Log every run. Let the *result*, not the vision, decide.
