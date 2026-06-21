"""
clozn.store — persist internal state across sessions (exact, inspectable, diffable).

A recurrent model's working memory is a fixed-size bag of named tensors, so it can be written
to disk and rehydrated into a *fresh* session — giving a model a place to remember that isn't
re-reading its own transcript (the "memory beyond chain-of-thought" idea). Saved exactly
(np.savez, no lossy round-trip), with a sidecar JSON manifest so a store is browsable.

This is the substrate under the future Persist-Memory feature (DESIGN.md): name a state, save
it, list the store, load one back into a live source, diff or edit before restoring.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .ops import Snapshot, restore, snapshot
from .spine import StateSource


class StateStore:
    """A directory of saved states: <name>.npz (tensors) + <name>.json (manifest)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _paths(self, name: str) -> tuple[str, str]:
        return os.path.join(self.root, name + ".npz"), os.path.join(self.root, name + ".json")

    def save(self, name: str, src_or_snap, note: str = "") -> str:
        """Persist a live source's state (or a Snapshot) under `name`. Returns the npz path."""
        snap = src_or_snap if isinstance(src_or_snap, Snapshot) else snapshot(src_or_snap, name)
        npz, js = self._paths(name)
        np.savez(npz, **snap.state)
        manifest = {
            "name": name,
            "label": snap.label or name,
            "note": note,
            "components": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                              "norm": round(float(np.linalg.norm(v)), 4)}
                           for k, v in snap.state.items()},
        }
        with open(js, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return npz

    def load(self, name: str) -> Snapshot:
        """Read a saved state back as a Snapshot (exact)."""
        npz, js = self._paths(name)
        with np.load(npz) as d:
            state = {k: d[k].copy() for k in d.files}
        label = name
        if os.path.exists(js):
            with open(js, encoding="utf-8") as f:
                label = json.load(f).get("label", name)
        return Snapshot(label, state)

    def into(self, source: StateSource, name: str) -> None:
        """Rehydrate a saved state directly into a live source (restore from disk)."""
        restore(source, self.load(name))

    def list(self) -> list[dict]:
        """The store's manifests, for a browsable memory shelf."""
        out = []
        for fn in sorted(os.listdir(self.root)):
            if fn.endswith(".json"):
                with open(os.path.join(self.root, fn), encoding="utf-8") as f:
                    out.append(json.load(f))
        return out
