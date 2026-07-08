"""python -m clozn -- entry point for the stdlib-only CLI (see clozn/cli.py)."""
import sys

from clozn.cli import main

if __name__ == "__main__":
    sys.exit(main())
