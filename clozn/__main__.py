"""python -m clozn -- entry point for the stdlib-only CLI."""
import sys

from clozn.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
