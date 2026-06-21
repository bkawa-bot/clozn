# docs/ — architecture & technical

The design docs, indexed. Read top-down: the synthesis first, then the per-layer deep dives.

## Start here

- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** — the synthesis: one product, the layers, the
  state-stream protocol, the interp maturity ladder. *How the whole thing fits.*
- **[../ROADMAP.md](../ROADMAP.md)** — the plan from here to the memory frontier, phased and
  tasked.

## Per-layer design

- **[DESIGN.md](DESIGN.md)** — the **engine** runtime architecture (scheduler §5, the L0 ggml
  adapter, the KV-cache tiers, the white-box taps). The deep "how the runtime works"; the
  engine code's `DESIGN §x` references point here. *(Written when the engine was the standalone
  `cloze` repo — read "cloze" as "the engine layer." A pass to retitle it for the monorepo is a
  tracked follow-up.)*
- **[TECHNICAL.md](TECHNICAL.md)** — the honest engineering account of the engine: what's fast,
  what it cost in quality, the measurements behind every claim.
- **[../inspector/DESIGN.md](../inspector/DESIGN.md)** — the **inspector** architecture: the
  state-stream spine, the ops, the substrate-agnostic `StateSource` seam, the memory verbs.
- **[../research/HANDOFF.md](../research/HANDOFF.md)** — the **research** thesis: compression
  under constraint, the legible-interior bet, the open cruxes.

## Protocol

- **[../protocol/README.md](../protocol/README.md)** — the state-stream contract (the keystone
  that collapses the engine's events and the inspector's `StateStep` into one). Authored in
  Roadmap phase 1.

The four [non-negotiable invariants](../ARCHITECTURE.md#carried-over-invariants-non-negotiable)
(honesty-first, the seam, tests-as-oracle, substrate-agnostic) hold across all of the above.
