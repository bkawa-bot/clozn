"""
Phase-0 spike — prove the architecture end-to-end, today.

Runs the toy delta-rule memory through the Spine, snapshots mid-sequence, keeps writing,
diffs, then restores — demonstrating the four claims Clozn is built on:
  (1) the evolving state streams as inspectable StateSteps (writes / rank / energy per token),
  (2) you can probe "what's currently stored,"
  (3) the state is a graspable object: snapshot -> restore *rewinds the memory*,
  (4) diff shows exactly what a write changed.

Pure numpy — runs anywhere. Swap ToyRecurrentSource -> FlaRecurrentSource for the real model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

from clozn.ops import diff, restore, snapshot          # noqa: E402
from clozn.spine import Spine, StateStep               # noqa: E402
from clozn.sources.toy_recurrent import ToyRecurrentSource  # noqa: E402


class Logger:
    def on_step(self, st: StateStep) -> None:
        print(f"  t{st.step:<2} wrote {st.meta['wrote']:<4} "
              f"rank={st.meta['rank']:<2} energy={st.meta['energy']:.2f}")


def recalls(src: ToyRecurrentSource) -> str:
    return "  ".join(f"{t}={src.recall(t):+.2f}" for t in ("A", "B", "C"))


VOCAB = ["A", "B", "C", "D", ".", "?"]
src = ToyRecurrentSource(VOCAB, d=16)
spine = Spine(src, [Logger()])

print("=== (1) stream the state: write A, B, then filler ===")
for _ in spine.run(["A", "B", ".", "."]):
    pass
print(f"  (2) probe what's stored:  {recalls(src)}")

snap = snapshot(src, "after A,B + filler")
print("\n[snapshot taken]")

print("\n=== keep going (no reset): write C, which overwrites memory ===")
for x in ["C", "."]:
    st = src.step(x)
    print(f"  t{st.step:<2} wrote {st.meta['wrote']:<4} rank={st.meta['rank']:<2} energy={st.meta['energy']:.2f}")
snap2 = snapshot(src, "after C")
print(f"  probe:  {recalls(src)}    <- C now present")

print("\n=== (4) diff: what did writing C change? ===")
d = diff(snap, snap2)
print(f"  state delta (Frobenius) = {d.total:.3f}   per-component = "
      f"{ {k: round(v, 3) for k, v in d.per_component.items()} }")

print("\n=== (3) restore the snapshot: rewind the memory ===")
restore(src, snap)
print(f"  probe:  {recalls(src)}    <- C gone, A/B back")
print("\n  the internal state is a graspable, restorable object. spine + ops work. ✓")
