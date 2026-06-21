# inspector/ — the white-box product

The thing you actually use. The state-stream spine, the white-box **ops**
(snapshot / restore / diff / edit / probe / steer + causal verify), the **memory**
(persist / associate / atlas), the per-token features, feature discovery, and the viz.
This is what used to be the `clozn` Python package.

It owns *all* product surface and drives the [engine](../engine) through the
[protocol](../protocol) — delegating hot paths down (it never does heavy compute itself).
Lightweight sources (e.g. RWKV via `transformers`) can also feed the same spine directly.

Substrate-agnostic by construction: a new model family is a new `StateSource`, not a fork.
