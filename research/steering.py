"""steering.py -- real-time TONE dials via contrastive activation steering.

A tone trait (warm, concise, formal, playful) is a DIRECTION in the residual stream: the mean activation
difference between answers that have the trait and answers that have its opposite. We compute that
direction once per (model, axis) from a few contrastive prompt pairs, then at generation time add
`strength * direction` to a mid-layer residual via a forward hook -- a continuous, real-time, composable
knob, no training, instantly draggable.

This is the OTHER half of the legible+tunable tool (see [[clozn-legible-tunable-pivot]]): topical/persona
traits go through the fast-weight CARDS (self_teach consolidation); low-dimensional TONE axes go HERE,
because they genuinely ARE single directions and you want to dial them live. Over-bleed is fine here --
an "always warm" model is correct; that's why these are sliders and not gated cards.

Built to wrap an already-loaded backbone (the clozn server's Qwen-7B), so the tool runs one model.
"""
from __future__ import annotations

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Each axis: a + pole and a - pole, expressed as system instructions. The steering vector is
# mean(resid | +pole) - mean(resid | -pole) over a few neutral seed prompts -- so it isolates the TONE,
# not any content. Slider value (strength) then scales it: + = toward the first pole, - = toward the second.
AXES = {
    "warm":    {"pos": "Respond in a warm, caring, encouraging tone.",
                "neg": "Respond in a cold, detached, clinical tone.",          "poles": ("warm", "detached")},
    "concise": {"pos": "Respond extremely concisely, in one short sentence.",
                "neg": "Respond at great length, with elaborate detail.",       "poles": ("concise", "verbose")},
    "formal":  {"pos": "Respond in very formal, professional language.",
                "neg": "Respond in very casual, slangy, relaxed language.",     "poles": ("formal", "casual")},
    "playful": {"pos": "Respond in a playful, witty, lighthearted tone.",
                "neg": "Respond in a serious, sober, no-nonsense tone.",        "poles": ("playful", "serious")},
}

# Neutral user turns to elicit the contrast on. Varied so the captured direction is tone, not topic.
SEED_PROMPTS = [
    "Tell me about your weekend plans.",
    "Can you help me with a problem at work?",
    "What do you think about this idea I have?",
    "Give me some advice for today.",
    "Explain what you can help with.",
    "I just got back from a trip.",
]


class SteeringControl:
    """Computes + applies the tone-axis steering vectors on a loaded causal LM (Qwen2-style)."""

    def __init__(self, model, tok, layer: int | None = None):
        self.model, self.tok = model, tok
        n = model.config.num_hidden_layers
        self.layer = layer if layer is not None else n // 2     # mid layer (Qwen-7B: 28 -> 14)
        self.vecs: dict[str, torch.Tensor] = {}                 # axis -> raw diff direction [H] (residual scale)
        self.strength: dict[str, float] = {}                    # axis -> current slider value (+/-)
        self._handle = None
        self._layers = model.model.layers                       # the decoder blocks

    @torch.no_grad()
    def _last_resid(self, system: str, user: str) -> torch.Tensor:
        """Residual at the LAST prompt token (the one that decides the first reply token), at self.layer."""
        ids = self.tok.apply_chat_template([{"role": "system", "content": system},
                                            {"role": "user", "content": user}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1]   # output of self.layer
        return hs[0, -1].float()

    @torch.no_grad()
    def compute(self, seeds=SEED_PROMPTS) -> dict:
        """Build every axis vector as mean(+pole) - mean(-pole) over the seeds. Returns each vector's norm
        (so we can sanity-check they're non-trivial and pick a sensible default strength range)."""
        out = {}
        for name, ax in AXES.items():
            pos = torch.stack([self._last_resid(ax["pos"], s) for s in seeds]).mean(0)
            neg = torch.stack([self._last_resid(ax["neg"], s) for s in seeds]).mean(0)
            self.vecs[name] = (pos - neg)                       # raw residual-scale diff; strength multiplies it
            out[name] = round(float(self.vecs[name].norm()), 2)
        return out

    def set(self, name: str, value: float):
        """Slider: value 0 = off, +x = toward the first pole, -x = toward the second. Typical |x| ~ 0..1.5."""
        self.strength[name] = float(value)

    def clear(self):
        self.strength = {}

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        add = None
        for name, s in self.strength.items():
            if s and name in self.vecs:
                v = (s * self.vecs[name]).to(h.device, h.dtype)
                add = v if add is None else add + v
        if add is not None:
            h = h + add
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def engage(self):
        if self._handle is None:
            self._handle = self._layers[self.layer].register_forward_hook(self._hook)

    def disengage(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def active(self) -> dict:
        return {k: v for k, v in self.strength.items() if v}

    @torch.no_grad()
    def generate(self, prompt: str, max_new=100) -> str:
        """Plain single-turn generate on the bare backbone (NO memory prefix) -- whatever steering is
        currently engaged applies via the hook. Used to A/B a dial: baseline vs steered on one prompt."""
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id or 0)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
