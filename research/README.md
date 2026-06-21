# research/ — the legibility science

The exploratory thread that decides *what* interp methods are worth building, at toy scale,
fast. This is what used to be the `legible-interior` repo.

The one idea underneath: **compression under constraint** — and the bet that you can build a
persistent, adaptive internal state that is **legible by construction** ("grow up without
becoming opaque"). The interpretability-tax experiments (does forcing a representation to be
sparse cost capability?) live here.

Feeds methods *up* into the [inspector](../inspector); depends on nothing below it. Results,
not the vision, decide which rungs get built — and which get cut, loudly. See `HANDOFF.md`.
