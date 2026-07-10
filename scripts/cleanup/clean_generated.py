"""Remove generated local artifacts from the Clozn checkout.

This is intentionally scoped to repo-local caches and build outputs. It skips
virtualenvs and the local llama.cpp checkout so cleanup does not damage
developer environments or third-party source trees.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".venv-jlens", ".venv-sae", "llama.cpp"}


def _under_skipped_dir(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part in SKIP_DIRS for part in rel.parts)


def _remove_dir(path: Path) -> bool:
    if not path.exists() or _under_skipped_dir(path):
        return False
    shutil.rmtree(path)
    return True


def main() -> int:
    removed: list[Path] = []

    for name in ("__pycache__", ".pytest_cache"):
        for path in ROOT.rglob(name):
            if path.is_dir() and _remove_dir(path):
                removed.append(path)

    for path in ROOT.rglob("*.pyc"):
        if path.is_file() and not _under_skipped_dir(path):
            path.unlink()
            removed.append(path)

    for pattern in (
        "engine/core/build-*",
        "engine/kernels/**/build",
        "engine/kernels/**/build-*",
    ):
        for path in ROOT.glob(pattern):
            if path.is_dir() and _remove_dir(path):
                removed.append(path)

    for path in sorted(removed):
        print(path.relative_to(ROOT))
    print(f"removed {len(removed)} generated artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
