"""Script entry point for Clozn's managed product-runtime acceptance gate."""
from __future__ import annotations

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from clozn.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main(["smoke", *sys.argv[1:]]))
