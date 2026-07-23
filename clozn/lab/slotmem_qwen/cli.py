"""Manual entrypoint for Qwen slot-memory smoke runs."""
from __future__ import annotations

import argparse
import os

from .sweep import run, sweep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="capacity sweep N=10..200 (select/express + null)")
    ap.add_argument("--out", default=os.path.expanduser("~/.clozn/slotmem_qwen1p5b.json"))
    a = ap.parse_args(argv)
    if a.sweep:
        return sweep(a.model, a.layer, a.out.replace(".json", "_sweep.json"))
    return run(a.model, a.layer, a.out.replace(".json", "_smoke.json") if a.smoke else a.out, smoke=a.smoke)


if __name__ == "__main__":
    main()
