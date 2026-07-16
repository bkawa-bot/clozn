"""clozn -- the product Python backend.

The package is being reorganized around product domains: CLI, runs, receipts, replay, behavior, readouts,
memory, substrates, profiles, and server glue. See RUNTIME_SPLIT.md for how this relates to engine/.
"""

# The single source of version truth: root pyproject.toml reads this via
# `[tool.setuptools.dynamic] version = {attr = "clozn.__version__"}` instead of duplicating the string, so
# a release only ever needs one edit. `clozn version` (clozn/cli/commands/version.py) prints it verbatim.
__version__ = "0.1.0"
