"""Shared helpers for shipped and user steering dial libraries."""
from __future__ import annotations

import json
import os


def load_library_file(path: str) -> dict:
    """Load a dial library JSON file, returning an empty dict for missing or broken files."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
