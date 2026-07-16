"""``clozn version`` -- print the installed version and, when available, the git commit it was built from.

The version string has exactly one source of truth: `clozn.__version__` (clozn/__init__.py), which the
root pyproject.toml also reads (`[tool.setuptools.dynamic] version = {attr = "clozn.__version__"}`) so a
release only ever needs one edit. The commit is best-effort: a `pip install`ed release ships no `.git`
directory (that's expected, not broken), so its absence is silently omitted rather than treated as an
error -- only a source checkout (or `pip install -e .`) has one to report.
"""
from __future__ import annotations

import os
import subprocess

import clozn


def _git_commit() -> "str | None":
    """Short commit hash of the git checkout `clozn` is importable from, plus a `-dirty` suffix if the
    working tree has uncommitted changes; None outside a git checkout (e.g. an installed release)."""
    here = os.path.dirname(os.path.abspath(clozn.__file__))
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here, capture_output=True, text=True, timeout=3,
        )
        if rev.returncode != 0:
            return None
        commit = rev.stdout.strip()
        if not commit:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=here, capture_output=True, text=True, timeout=3,
        )
        if status.returncode == 0 and status.stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return None   # no `git` on PATH, or the call itself failed -- report the version alone


def cmd_version(_args) -> int:
    commit = _git_commit()
    print(f"clozn {clozn.__version__}" + (f" ({commit})" if commit else ""))
    return 0
