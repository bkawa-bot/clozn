"""Per-token feature attribution: which named features light up as RWKV-4 reads a text."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.features import feature_film               # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource      # noqa: E402
from clozn.viz import render_feature_film              # noqa: E402

TEXT = "I love this wonderful gift. Did she walk home alone?"


def main():
    print("loading RWKV-4-169m, fitting concept axes, streaming the text ...")
    src = RwkvStateSource()
    toks, rows, M = feature_film(src, TEXT)

    norm = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-9)
    print(f"\ntext: {TEXT!r}\n")
    head = "feature \\ token  " + " ".join(f"{t.strip()[:5]:>5}" for t in toks)
    print(head)
    for c, (name, pl, nl) in enumerate(rows):
        cells = " ".join(f"{norm[c,t]:+5.1f}" for t in range(len(toks)))
        print(f"  {name:<14} {cells}   (+{pl} / -{nl})")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "features.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_feature_film(toks, rows, M, subtitle=f"RWKV-4-169m · {TEXT!r}"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
