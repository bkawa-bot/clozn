"""Native engine activation steering adapter."""
from __future__ import annotations

import json
import os

import numpy as np

from . import axes


class EngineSteer:
    """Tone dials on the native engine using harvested residual directions."""

    def __init__(self, engine_client, layer=14):
        self.ec = engine_client
        self.layer = layer
        self.vecs = {}
        self.custom = {}
        self.strength = {}
        self.base, self.resid_norm = 1.0, 0.0
        self.ready = False

    def load_library(self, path):
        """Load shipped dial metadata into self.custom without harvesting directions eagerly."""
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for name, v in data.items():
                self.custom[name] = {
                    "pos": v["pos"],
                    "neg": v["neg"],
                    "max": float(v.get("max", 0.5)),
                    "poles": [name, "neutral"],
                }
        except Exception:
            pass

    def compute(self, seeds=None) -> dict:
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        # First-use signal: harvesting every base + library dial direction is ~30-40s of sequential
        # engine round-trips and is otherwise SILENT, so a first dial-touching turn reads as a hang
        # (a known usage-test papercut). stderr + ASCII-only (Windows cp1252).
        import sys
        n_dials = len(axes.AXES) + sum(1 for n in self.custom if n not in self.vecs)
        print(f"[steer] harvesting {n_dials} tone-dial directions; first use, ~30-40s...",
              file=sys.stderr, flush=True)
        norms = []
        for name, ax in axes.AXES.items():
            pos = np.mean(
                [self.ec.harvest(ax["pos"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            neg = np.mean(
                [self.ec.harvest(ax["neg"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            norms += [float(np.linalg.norm(pos)), float(np.linalg.norm(neg))]
            d = pos - neg
            self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        for name, entry in self.custom.items():
            if name in self.vecs:
                continue
            pos = np.mean(
                [self.ec.harvest(entry["pos"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            neg = np.mean(
                [self.ec.harvest(entry["neg"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            d = pos - neg
            self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.08 * self.resid_norm
        self.ready = True
        return {"resid_norm": round(self.resid_norm, 1), "base": round(self.base, 1), "axes": list(self.vecs)}

    def add_custom(self, name, pos, neg, mx=0.5, seeds=None) -> dict:
        """Register a user-defined engine dial using the same diff-of-means recipe."""
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        name = name.strip()[:24]
        pos_v = np.mean(
            [self.ec.harvest(pos + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
            axis=0,
        )
        neg_v = np.mean(
            [self.ec.harvest(neg + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
            axis=0,
        )
        d = pos_v - neg_v
        self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        self.custom[name] = {"pos": pos, "neg": neg, "max": float(mx), "poles": [name, "neutral"],
                              "source": "user"}
        return self.custom[name]

    def remove_custom(self, name):
        self.custom.pop(name, None)
        self.vecs.pop(name, None)
        self.strength.pop(name, None)

    def save_custom(self, path):
        """Persist only user-created dials, not shipped-library metadata."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {k: {"pos": v["pos"], "neg": v["neg"], "max": v["max"]}
                 for k, v in self.custom.items() if v.get("source") == "user"},
                f,
            )

    def load_custom(self, path):
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self.add_custom(k, v["pos"], v["neg"], float(v.get("max", 0.5)))
        except Exception:
            pass

    def set(self, name, value):
        mx = (axes.AXES.get(name) or self.custom.get(name) or {}).get("max", 1.5)
        self.strength[name] = max(-mx, min(mx, float(value)))

    @staticmethod
    def _text(r):
        ch = r.get("choices") if isinstance(r, dict) else None
        if ch:
            return ch[0].get("text") or ch[0].get("message", {}).get("content") or ""
        return (r.get("text") or "") if isinstance(r, dict) else str(r)

    def generate(self, prompt, strength=None, max_new=70):
        """Generate through the engine with active dials applied."""
        if not self.ready:
            self.compute()
        s = (self.strength if getattr(self, "_engaged", False) else {}) if strength is None else strength
        active = {k: v for k, v in s.items() if v and k in self.vecs}
        if not active:
            return self._text(self.ec.complete(prompt, max_tokens=max_new))
        vec = np.zeros_like(next(iter(self.vecs.values())))
        for k, v in active.items():
            vec = vec + self.base * float(v) * self.vecs[k]
        return self._text(self.ec.intervene(prompt, vector=vec.tolist(), coef=1.0,
                                             layer=self.layer, max_tokens=max_new))

    def steer_vector(self, strength):
        """Return the summed, pre-scaled tone direction for active dials, or None."""
        if not self.ready:
            self.compute()
        active = {k: v for k, v in (strength or {}).items() if v and k in self.vecs}
        if not active:
            return None
        vec = np.zeros_like(next(iter(self.vecs.values())))
        for k, v in active.items():
            vec = vec + self.base * float(v) * self.vecs[k]
        n = float(np.linalg.norm(vec))
        cap = self.base * 1.3
        if n > cap:
            vec = vec * (cap / n)
        return vec.tolist()

    def clear(self):
        self.strength = {}

    def engage(self):
        self._engaged = True

    def disengage(self):
        self._engaged = False

    def active(self):
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.strength, f)

    def load_state(self, path):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.strength = {k: float(v) for k, v in json.load(f).items()}
            except Exception:
                pass
