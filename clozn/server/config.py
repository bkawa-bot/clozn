"""Server config -- path resolution + process-wide startup constants for clozn.server.app.

Imported FIRST by app.py (before the engine client SDK) so REPO_ROOT/DEMO exist and engine/client +
engine/lab are already on sys.path by the time app.py goes on to `from cloze_engine import ...`. The
side effects here (sys.path mutation, the stdout encoding, the HF env default) are deliberately
module-load-time, exactly as they were when this all lived at the top of clozn_server.py/app.py --
moving them into their own module changes nothing about when they run, only where they're written.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # clozn/server/config.py -> repo root is two levels up
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "lab"))    # so the dream substrate can import cloze_lab
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))  # the engine white-box SDK
DEMO = os.path.join(REPO_ROOT, "studio")

CLOZN_DIR = os.path.join(os.path.expanduser("~"), ".clozn")     # studio memory + personality persist here
