# docs/ — architecture & technical

The design docs, indexed. Read top-down: the synthesis first, then the per-layer deep dives.

## Start here

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the synthesis: one product, the layers, the
  state-stream protocol, the interp maturity ladder. *How the whole thing fits.*
- **[ROADMAP.md](ROADMAP.md)** — the consolidated map: what's done, the v1 cut, what's next.

## Per-layer design

- **[DESIGN.md](DESIGN.md)** — the **engine** runtime architecture (scheduler §5, the L0 ggml
  adapter, the KV-cache tiers, the white-box taps). The deep "how the runtime works"; the
  engine code's `DESIGN §x` references point here. *(Written when the engine was the standalone
  `cloze` repo — read "cloze" as "the engine layer.")*
- **[TECHNICAL.md](TECHNICAL.md)** — the honest engineering account of the engine: what's fast,
  what it cost in quality, the measurements behind every claim.
- **[STUDIO.md](STUDIO.md)** — the studio UI: pages, panels, and what each surface shows.
- **[MODEL_SUPPORT.md](MODEL_SUPPORT.md)** — which model families run, and on which paths.
- **[WORKSPACE_LENS.md](WORKSPACE_LENS.md)** — the J-lens: how it's fitted, what it can and
  cannot claim, and the trace fixture format.
- **[EXPLAIN_THIS_ANSWER_SPEC.md](EXPLAIN_THIS_ANSWER_SPEC.md)** — the explain/receipts spec
  (M1 assembly, causal receipts, the honesty rules the endpoints enforce).
- **[RUNTIME_SPLIT.md](RUNTIME_SPLIT.md)** — how the Python package splits between the pure
  library and the served runtime.

## Protocol

- **[../protocol/README.md](../protocol/README.md)** — the state-stream contract the engine
  emits and the studio consumes.

The four non-negotiable invariants (honesty-first, the seam, tests-as-oracle,
substrate-agnostic) hold across all of the above — see
[ARCHITECTURE.md](ARCHITECTURE.md).
