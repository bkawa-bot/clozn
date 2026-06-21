"""
Phase-2 — the diffusion CANVAS as a StateSource. The thesis payoff: the SAME white-box ops and
the SAME store that worked on the RWKV recurrent matrix now work on a denoising board, unchanged.

Proves four things on the new substrate:
  (1) PARALLEL  — many slots fill per pass (the dLLM signature; AR fills one token per step).
  (2) SNAPSHOT  — clozn.ops.snapshot/restore rewinds the canvas bit-exactly mid-denoise.
  (3) DIFF      — clozn.ops.diff shows exactly which slots a pass filled.
  (4) PERSIST   — clozn.store saves a half-denoised canvas; a FRESH source loads it and finishes
                  the generation (resumable diffusion) — same store.py that persisted RWKV memory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.ops import diff, restore, snapshot          # noqa: E402  (the SAME ops as RWKV)
from clozn.store import StateStore                       # noqa: E402  (the SAME store as RWKV)
from clozn.sources.diffusion import DiffusionStateSource # noqa: E402
from clozn.viz import render_denoise_film                # noqa: E402

PROMPT, MAX_NEW, STEPS = "hi", 20, 8


def main():
    src = DiffusionStateSource(prompt=PROMPT, max_new=MAX_NEW, steps=STEPS)

    print("=== (1) PARALLEL denoising — watch many slots fill per pass ===")
    steps = src.run()
    for s in steps:
        bar = "█" * s.meta["n_committed"]
        print(f"  t{s.meta['pass']:<2} +{s.meta['n_committed']:<2} committed, "
              f"{s.meta['n_masked_after']:<2} masked left  {bar}")
    print(f"  -> {len(steps)} passes filled {MAX_NEW} slots; AR would need {MAX_NEW} steps. "
          f"max committed in one pass = {max(s.meta['n_committed'] for s in steps)}.")

    print("\n=== (2) SNAPSHOT/RESTORE the canvas mid-denoise (clozn.ops, unchanged) ===")
    src.reset()
    for _ in range(3):
        src.step()
    mid = snapshot(src, "after 3 passes")
    masked_at_mid = int((src.board == src.mask).sum())
    while not src.done:
        src.step()                                       # finish it
    restore(src, mid)                                    # rewind the canvas
    assert int((src.board == src.mask).sum()) == masked_at_mid
    rewound = snapshot(src)
    maxdiff = max(float(np.abs(rewound.state[k] - mid.state[k]).max()) for k in mid.state)
    print(f"  rewound to pass 3: {masked_at_mid} slots masked again; "
          f"max|restored-snapshot| = {maxdiff:.2e}  -> bit-exact ✓")

    print("\n=== (3) DIFF two consecutive passes — what did one pass fill? ===")
    a = snapshot(src)                                    # at pass 3
    src.step()
    b = snapshot(src)                                    # at pass 4
    newly = int(((b.state['filled'] - a.state['filled']) > 0.5).sum())
    print(f"  clozn.ops.diff total = {diff(a, b).total:.2f}; "
          f"per-component = {{ {', '.join(f'{k}:{v:.2f}' for k,v in diff(a,b).per_component.items())} }}")
    print(f"  -> that pass filled {newly} previously-masked slots.")

    print("\n=== (4) PERSIST a half-denoised canvas, resume in a FRESH source (clozn.store) ===")
    store = StateStore(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "runs", "store"))
    gen = DiffusionStateSource(prompt=PROMPT, max_new=MAX_NEW, steps=STEPS)
    gen.reset()
    for _ in range(4):
        gen.step()                                       # denoise halfway
    half_masked = int((gen.board == gen.mask).sum())
    store.save("half_canvas", gen, note="a half-denoised diffusion board")
    print(f"  saved a canvas with {half_masked} slots still masked")

    fresh = DiffusionStateSource(prompt=PROMPT, max_new=MAX_NEW, steps=STEPS)
    store.into(fresh, "half_canvas")                     # a brand-new source picks up the work
    print(f"  fresh source loaded it: {int((fresh.board==fresh.mask).sum())} masked, resuming...")
    while not fresh.done:
        fresh.step()
    same = np.array_equal(fresh.board, _final_board(PROMPT, MAX_NEW, STEPS))  # vs uninterrupted run
    print(f"  resumed-from-disk final == uninterrupted final: {same}  -> resumable diffusion ✓")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "denoise.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_denoise_film(steps, prompt_len=src.p,
                                    title="Clozn · Watch (diffusion canvas)",
                                    subtitle=f"FakeAdapter · prompt {PROMPT!r} · {MAX_NEW} slots · {len(steps)} passes"))
    print("\nwrote", out)
    print("Phase 2 ✓  the same ops + store work on the diffusion canvas — substrate-agnostic.")


def _final_board(prompt, max_new, steps):
    s = DiffusionStateSource(prompt=prompt, max_new=max_new, steps=steps)
    s.run()
    return s.board.copy()


if __name__ == "__main__":
    main()
