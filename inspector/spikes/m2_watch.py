"""Phase-1 M2 — render the Watch cockpit for a real RWKV-4 run."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"), exist_ok=True)

from clozn.sources.hf_rwkv import RwkvStateSource   # noqa: E402
from clozn.viz import render_state_film             # noqa: E402

src = RwkvStateSource()
text = "The capital of France is Paris. Two plus two equals"
steps = src.feed(text)

print("logit-lens thought stream (token -> what it predicts next):")
for s in steps:
    print(f"  {s.meta['token']!r:<12} -> {s.meta['top1']!r}")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "watch.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(render_state_film(steps, title="Clozn · Watch — RWKV-4 reading", subtitle=repr(text)))
print("\nwrote", out)
