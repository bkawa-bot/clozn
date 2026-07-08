#!/usr/bin/env bash
# clozn launcher (posix) -- runs the stdlib-only CLI with whatever python is on PATH.
cd "$(dirname "$0")" && exec python -m clozn "$@"
