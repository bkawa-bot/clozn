"""Server config -- path resolution + process-wide startup constants for clozn.server.app.

Imported FIRST by app.py so REPO_ROOT/DEMO exist and engine/client is on sys.path by the time app.py
goes on to `from cloze_engine import ...`. These side effects are deliberately module-load-time.

TORCH-FREE BY CONSTRUCTION: this runs on the PRODUCT import path, so it must not pull in anything only
the lab needs. The `engine/lab` sys.path entry (for the Dream substrate's cloze_lab) and the HF hub
symlink workaround used to live here; they moved to the `clozn lab` entry point (clozn/lab/app.py). A
product `clozn serve` process needs neither, and keeping them off this path is part of what makes the
product package physically unable to reach the Torch lab code.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # clozn/server/config.py -> repo root is two levels up
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))  # the engine white-box SDK (product; torch-free)
DEMO = os.path.abspath(os.path.expanduser(
    os.environ.get("CLOZN_STUDIO_DIR", os.path.join(REPO_ROOT, "studio"))
))

CLOZN_DIR = os.path.join(os.path.expanduser("~"), ".clozn")     # studio memory + personality persist here
