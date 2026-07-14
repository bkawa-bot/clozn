"""Readout modules, loaded lazily so product imports never pull PyTorch lab code."""
from __future__ import annotations

import importlib

__all__ = ["atlas_concepts", "workspace_lens", "sae7b"]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
