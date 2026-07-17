"""gen_dial_calibration.py -- write ~/.clozn/dial_calibration.json from the shipped library, ONCE.

The companion to deploy_dial_library.py. That script REGISTERS the 27 library-only dials (the ones that
don't exist as a steering.AXES built-in) so they show up as sliders; THIS script writes the per-model
RANGE file that caps ALL 33 shipped dials -- both the 27 library dials and the 6 that overlap a built-in
(warm, playful, formal, concise, poetic, concrete). clozn_server.py's /steer/axes reads this file (via
_dial_calibration / _with_calibration) to serve each dial's calibrated `max` + `usable_range` +
`works`, and heavn Patch uses those to cap each dial slider and label its usable range.

No model / no GPU: the ranges are already baked into research/dial_library_shipped.json's `ship_range`
(the human-curated conservative range from the autocalibrate sweep). This is a
pure JSON transform, so it's split out from deploy_dial_library.py (which needs the loaded 7B to compute
the 27 directions). Run either order; both are idempotent overwrites of their own file.

    python research/gen_dial_calibration.py            # write it
    python research/gen_dial_calibration.py --check     # print what it WOULD write, touch nothing
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED_LIBRARY_PATH = os.path.join(HERE, "..", "..", "clozn", "data", "dial_library_shipped.json")


def build_calibration(shipped_path: str = SHIPPED_LIBRARY_PATH) -> dict:
    """The {name: {usable_max, usable_range, derail_point, works, category}} map /steer/axes expects,
    built straight from each shipped dial's curated `ship_range`. `works` is True for every shipped dial
    (a dial only survives curation into dial_library_shipped.json IF it works -- see dial_library_findings
    .md's 71->47->33 pipeline); `derail_point` is left None (the ship_range top is already the conservative
    human-dropped-back ceiling, so the raw derail point isn't what caps the slider)."""
    with open(shipped_path, encoding="utf-8") as f:
        dials = json.load(f)["dials"]
    calib = {}
    for d in dials:
        lo, hi = d["ship_range"]
        calib[d["name"]] = {"usable_max": hi, "usable_range": [lo, hi], "derail_point": None,
                            "works": True, "category": d["category"]}
    return calib


def dest_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".clozn", "dial_calibration.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write ~/.clozn/dial_calibration.json from the shipped "
                                             "dial library's curated ranges (no model needed).")
    ap.add_argument("--check", action="store_true", help="print the plan only; write nothing")
    args = ap.parse_args(argv)

    calib = build_calibration()
    dest = dest_path()
    if args.check:
        print(f"[check] would write {len(calib)} dial range(s) -> {dest}")
        print("  sample: warm ->", calib.get("warm"))
        return None

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"wrote {dest} -- {len(calib)} dials")
    return calib


if __name__ == "__main__":
    main()
