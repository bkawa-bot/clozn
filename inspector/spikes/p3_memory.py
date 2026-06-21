"""
Phase-3 M1 — legible personal memory: associative recall over internal states (real RWKV-4).

Store several mind-states (the state after reading different things), then ask, for a NEW context,
"what does this remind you of?" — keyed by the shape of thought, not by text matching. Plus the
write-gate that keeps the shelf sparse (don't re-store what you already know). Built on store.py
+ the StateSource seam, so it's exact and substrate-agnostic. Honest: 169M model, we report the
actual ranking whatever it is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from clozn.memory import MemoryShelf                  # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource     # noqa: E402
from clozn.store import StateStore                     # noqa: E402
from clozn.viz import render_memory_shelf              # noqa: E402

MEMORIES = {
    "france":  "The capital of France is Paris.",
    "japan":   "The capital of Japan is Tokyo.",
    "math":    "Two plus two equals four. Three times three is nine.",
    "weather": "It rained all day and the grey sky never cleared.",
    "cooking": "Add the flour and eggs, then bake the cake for an hour.",
}
QUERIES = {
    "a geography question": "The capital of Germany is",
    "an arithmetic question": "Seven plus five equals",
    "a weather remark": "The clouds were dark and it started to",
}


def main():
    runs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    shelf = MemoryShelf(StateStore(os.path.join(runs, "shelf")), component="att_num")
    src = RwkvStateSource()

    print("=== building the shelf (one saved mind-state per topic) ===")
    for name, text in MEMORIES.items():
        src.reset(); src.feed(text)
        shelf.remember(name, src, note=text)
        print(f"  remembered {name!r:<10} <- {text!r}")

    print("\n=== write-gate: try to re-store a near-duplicate of 'france' ===")
    src.reset(); src.feed(MEMORIES["france"])
    stored = shelf.remember("france_again", src, note="dup", gate=0.97)
    print(f"  remember('france_again', gate=0.97) -> stored={stored}  "
          f"({'kept shelf sparse (skipped)' if not stored else 'stored'})")

    print("\n=== associative recall: 'what does this remind you of?' ===")
    last_matches = None
    for label, q in QUERIES.items():
        src.reset(); src.feed(q)
        matches = shelf.nearest(src, k=len(MEMORIES))
        top = matches[0]
        print(f"  {label:<22} {q!r}")
        print(f"     -> reminds it of: " + ", ".join(f"{m.name}({m.similarity:+.2f})" for m in matches[:3]))
        print(f"     -> top match: {top.name!r}")
        last_matches = (q, matches)

    # render the last query's shelf ranking
    q, matches = last_matches
    out = os.path.join(runs, "memory_shelf.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_memory_shelf(q, [(m.name, m.similarity, m.note) for m in matches],
                                    subtitle="RWKV-4-169m · att_num · shelf-centered cosine"))
    print("\nwrote", out)
    print("Phase 3 M1 ✓  associative memory over internal states — the same store, a new verb.")


if __name__ == "__main__":
    main()
