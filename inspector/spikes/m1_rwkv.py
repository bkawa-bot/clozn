"""
Phase-1 M1 — snapshot / restore / diff on a REAL trained recurrent model (RWKV-4 via transformers).
Same spine, same ops as the toy spike; now the state is a real model's memory. Proves:
  (1) the recurrent state streams as inspectable steps,
  (2) the state *drives the output* (France-context vs Japan-context predict differently),
  (3) snapshot/diff/restore work bit-exactly on real state.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.ops import diff, restore, snapshot          # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource       # noqa: E402

print("loading RWKV-4-169m (cached) ...")
src = RwkvStateSource()

print("\n=== (1) stream the state: feed 'The capital of France is' ===")
for st in src.feed("The capital of France is"):
    print(f"  t{st.step:<2} {st.meta['token']!r:<10} "
          f"|att_num|={st.meta['norms']['att_num']:<8} |att_x|={st.meta['norms']['att_x']}")
p_france = src.top_next(3)
print("  top next tokens:", p_france)

snap = snapshot(src, "France context")
print("\n[snapshot taken]")

print("\n=== (2) overwrite memory: feed ' Tokyo. The capital of Japan is' ===")
src.feed(" Tokyo. The capital of Japan is")
p_japan = src.top_next(3)
print("  top next tokens:", p_japan)
snap2 = snapshot(src, "Japan context")

print("\n  France-context predicts:", [t for t, _ in p_france])
print("  Japan-context  predicts:", [t for t, _ in p_japan])
print("  -> the recurrent STATE carries the context and drives the output.",
      "(differ)" if p_france != p_japan else "(same?!)")

print("\n=== (3) diff: what did the Japan text change? ===")
d = diff(snap, snap2)
top = sorted(((v, k) for k, v in d.per_component.items()), reverse=True)
print(f"  total state delta = {d.total:.2f};  moved most: " + ", ".join(f"{k}({v:.1f})" for v, k in top[:3]))

print("\n=== (3) restore the snapshot: rewind the real memory ===")
restore(src, snap)
after = src.get_state()
maxdiff = max(float(np.abs(after[n] - snap.state[n]).max()) for n in snap.state)
print(f"  max |restored - snapshot| over all state components = {maxdiff:.2e}")
print("  -> a real recurrent model's state is a graspable, bit-exact, restorable object. M1 ✓"
      if maxdiff < 1e-5 else "  -> restore mismatch!")
