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

import json
import os

import numpy as np
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
    "curious": {"pos": "Respond with curiosity, wondering aloud and asking thoughtful questions.",
                "neg": "Respond flatly, just stating facts with no curiosity.",  "poles": ("curious", "matter-of-fact")},
    "poetic":  {"pos": "Respond poetically, with vivid imagery and metaphor.",
                "neg": "Respond plainly and literally, with no figurative language.", "poles": ("poetic", "plain")},
    "technical": {"pos": "Respond technically, with precise terminology and detail.",
                  "neg": "Respond in simple, everyday language anyone could follow.", "poles": ("technical", "simple")},
    # --- cognitive / behavioral axes (steer HOW it thinks). Finickier than tone: each has a calibrated
    #     "max" safe strength (validated by dogfooding -- candid is gold at ~0.4 but degenerates by ~0.6). ---
    "candid":  {"pos": "Respond with candid, critical pushback: point out flaws and disagree when it is warranted, do not just validate.",
                "neg": "Respond agreeably and supportively: validate the user's view and avoid disagreement.", "poles": ("candid", "agreeable"), "max": 0.45},
    "confident": {"pos": "Respond with confident, decisive assertions and no hedging.",
                  "neg": "Respond cautiously and tentatively, hedging with qualifications and uncertainty.", "poles": ("confident", "tentative"), "max": 1.0},
    "concrete": {"pos": "Respond with concrete, specific detail and vivid, particular examples.",
                 "neg": "Respond abstractly, in general high-level concepts with no specifics.", "poles": ("concrete", "abstract"), "max": 0.5},
}

# Neutral user turns to elicit the contrast on. Varied (asking / sharing / venting / deciding; factual,
# emotional, practical; short + medium) so the averaged direction captures TONE, not any one topic.
# More + more-varied seeds => a lower-variance, less topic-leaky estimate of the same axis (chat-like on
# purpose -- the dial is applied during chat, so off-domain seeds would only dilute it).
SEED_PROMPTS = [
    "Tell me about your weekend plans.",
    "Can you help me with a problem at work?",
    "What do you think about this idea I have?",
    "Give me some advice for today.",
    "Explain what you can help with.",
    "I just got back from a trip.",
    "I'm trying to decide between two job offers.",
    "Why is the sky blue?",
    "I had a rough day and just need to vent.",
    "What's a good book to read this month?",
    "Walk me through how to set up a budget.",
    "My friend and I had a disagreement yesterday.",
    "I'm thinking about picking up a new hobby.",
    "Summarize what's been going on in the news.",
    "Honestly, is it worth learning to cook?",
    "Tell me something interesting.",
    "I keep procrastinating on a big project.",
    "What should I make for dinner tonight?",
]


class SteeringControl:
    """Computes + applies the tone-axis steering vectors on a loaded causal LM (Qwen2-style)."""

    def __init__(self, model, tok, layer: int | None = None):
        self.model, self.tok = model, tok
        n = model.config.num_hidden_layers
        self.layer = layer if layer is not None else n // 2     # mid layer (Qwen-7B: 28 -> 14)
        self.vecs: dict[str, torch.Tensor] = {}                 # axis -> UNIT diff direction [H]
        self.custom: dict[str, dict] = {}                       # USER-DEFINED dials: name -> {pos, neg, max, poles}
        self.strength: dict[str, float] = {}                    # axis -> current slider value (+/-)
        self.base = 1.0                                         # per-model scale (set by compute())
        self.resid_norm = 0.0
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
        """Build every axis as a UNIT direction mean(+pole) - mean(-pole) over the seeds, and auto-calibrate
        `base` from the residual norm so slider 1.0 is a safe, clearly-on effect on ANY model (raw diff
        magnitudes vary per axis/model; normalizing makes the dials consistent). Hook adds slider*base*dir.
        Calibrated to this backbone: ~0.14*|resid| is clearly-on; >~0.4*|resid| breaks coherence."""
        out, norms = {}, []
        for name, ax in AXES.items():
            pv = [self._last_resid(ax["pos"], s) for s in seeds]
            nv = [self._last_resid(ax["neg"], s) for s in seeds]
            pos, neg = torch.stack(pv).mean(0), torch.stack(nv).mean(0)
            norms += [float(x.norm()) for x in pv + nv]
            d = pos - neg
            self.vecs[name] = d / (d.norm() + 1e-8)             # UNIT direction
            out[name] = round(float(d.norm()), 2)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.85 * self.resid_norm                      # slider 1.0 -> ~0.85*|resid|: clearly on (raw
        #                                                         tests: ~1x|resid| good, ~2.5x breaks); cap UI ~1.8
        return {"raw_norms": out, "resid_norm": round(self.resid_norm, 1), "base": round(self.base, 2)}

    @torch.no_grad()
    def add_custom(self, name: str, pos: str, neg: str, mx: float = 0.5, seeds=SEED_PROMPTS) -> dict:
        """A USER-DEFINED dial: the exact same recipe as the built-ins, on arbitrary poles. mean(+pole) -
        mean(-pole) over the seeds -> a unit direction stored under `name` next to the static AXES; the
        slider scales it like any other. Custom dials get a conservative `max` (the safe ceiling varies per
        axis and is hard to auto-detect -- the user can nudge it)."""
        name = name.strip()[:24]
        pv = [self._last_resid(pos, s) for s in seeds]
        nv = [self._last_resid(neg, s) for s in seeds]
        d = torch.stack(pv).mean(0) - torch.stack(nv).mean(0)
        self.vecs[name] = d / (d.norm() + 1e-8)
        self.custom[name] = {"pos": pos, "neg": neg, "max": float(mx), "poles": [name, "neutral"]}
        return self.custom[name]

    def remove_custom(self, name: str):
        self.custom.pop(name, None)
        self.vecs.pop(name, None)
        self.strength.pop(name, None)

    def save_custom(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: {"pos": v["pos"], "neg": v["neg"], "max": v["max"]} for k, v in self.custom.items()}, f)

    def load_custom(self, path: str):
        if not os.path.isfile(path):
            return
        try:
            for k, v in json.load(open(path, encoding="utf-8")).items():
                self.add_custom(k, v["pos"], v["neg"], float(v.get("max", 0.5)))
        except Exception:
            pass

    # On-distribution headroom for the COMBINED push. One strong dial should steer hard, but the SUM of
    # many dials must not overwhelm the residual and shove the model off-distribution -- that's what mutes
    # subtler biases (the learned-memory soft-prefix). So we bound ||sum-of-dials|| to a fraction of the
    # ACTUAL per-position residual norm, leaving the rest of the residual (>= 1-k of it) for everything the
    # model was already doing. This is a magnitude ceiling only: direction is untouched. k ~ 0.6-0.8 keeps
    # steering firmly in the model's real operating range; below the ceiling (a single moderate dial) the
    # cap never bites, so single-dial behavior is unchanged. The goal is on-distribution steering, NOT
    # muting tone -- a lone strong dial still pushes to its full per-axis "max".
    MAX_DELTA_FRAC = 0.75

    def set(self, name: str, value: float):
        """Slider: value 0 = off, +x = toward the first pole, -x = toward the second. Typical |x| ~ 0..1.5.
        Capped to the axis's per-axis "max" -- cognitive/custom axes degenerate above their sweet spot."""
        mx = (AXES.get(name) or self.custom.get(name) or {}).get("max", 1.5)
        self.strength[name] = max(-mx, min(mx, float(value)))

    def clear(self):
        self.strength = {}

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        add = None
        for name, s in self.strength.items():
            if s and name in self.vecs:
                v = (s * self.base * self.vecs[name]).to(h.device, h.dtype)
                add = v if add is None else add + v
        if add is not None:
            h = h + self._cap_delta(add, h)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def _cap_delta(self, add: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Bound the summed steering delta to MAX_DELTA_FRAC * ||residual|| PER POSITION, direction intact.

        `add` is a single [H] vector (same push at every position); `h` is [..., H], so the residual norm
        varies token-to-token. We scale `add` down only where ||add|| exceeds the per-position ceiling, so:
          * a single moderate dial (||add|| below the ceiling everywhere) passes through untouched, and
          * a stack of strong dials is clamped to the ceiling -- staying on-distribution so the memory bias
            in the residual survives instead of being drowned by an oversized push.
        Uses the same float reduction on device (no host sync in the common path); returns a [..., H] delta
        broadcast-added to h by the caller."""
        add_norm = add.norm()
        if float(add_norm) <= 0.0:
            return add
        ceil = self.MAX_DELTA_FRAC * h.norm(dim=-1, keepdim=True)   # [..., 1] per-position budget
        # scale = min(1, ceil / ||add||): 1 where the push fits, <1 where it must be trimmed to the budget.
        scale = torch.clamp(ceil / add_norm, max=1.0)              # [..., 1]
        return add * scale                                          # broadcasts [H] * [..., 1] -> [..., H]

    def engage(self):
        if self._handle is None:
            self._handle = self._layers[self.layer].register_forward_hook(self._hook)

    def disengage(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def active(self) -> dict:
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path: str):
        """Persist just the slider values (the vectors are cheap to recompute on boot)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.strength, f)

    def load_state(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as f:
                self.strength = {k: float(v) for k, v in json.load(f).items()}
            return True
        except Exception:
            return False

    @torch.no_grad()
    def generate(self, prompt: str, max_new=100) -> str:
        """Plain single-turn generate on the bare backbone (NO memory prefix) -- whatever steering is
        currently engaged applies via the hook. Used to A/B a dial: baseline vs steered on one prompt."""
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                  repetition_penalty=1.3, no_repeat_ngram_size=3,   # trim steering loops in the A/B
                                  pad_token_id=self.tok.eos_token_id or 0)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


class DreamSteering(SteeringControl):
    """Tone dials for a Dream-family DIFFUSION adapter (the cloze_lab ModelAdapter).

    Same idea and same hook/dial machinery as SteeringControl -- a tone axis is a residual-stream
    direction, added live at a mid layer scaled by the slider. Two differences for diffusion:
      * residuals are captured with a forward HOOK driven through the adapter's own forward (Dream's
        custom modeling doesn't expose output_hidden_states cleanly, and the adapter handles its
        non-causal mask / shifted head), and we mean-pool over ALL prompt tokens (diffusion attends
        bidirectionally -- there is no privileged 'last' token like in AR);
      * the dials apply during the iterative DENOISING forwards -- engage the hook, run a denoise,
        and every unmasking pass is steered.
    """

    def __init__(self, adapter, layer: int | None = None):
        self.ad = adapter
        self.model = adapter._model
        self.tok = adapter._tok
        self.dev = adapter._device
        base_model = getattr(self.model, "model", self.model)        # DreamModel vs *ForCausalLM wrapper
        self._layers = base_model.layers
        n = len(self._layers)
        self.layer = layer if layer is not None else n // 2
        self.vecs, self.strength = {}, {}
        self.base, self.resid_norm = 1.0, 0.0
        self._handle = None

    @torch.no_grad()
    def _resid(self, text: str) -> torch.Tensor:
        """Mean-pooled residual at self.layer for a chat-wrapped prompt, captured via a hook on the
        layer while the ADAPTER drives the forward (so Dream's mask/positions are handled correctly)."""
        ids = self.ad.encode(text, chat=True)
        board = np.asarray(ids, dtype=np.int64)
        n = len(ids)
        attn = np.ones((n, n), dtype=bool)                            # full (non-causal) attention
        cap = {}

        def grab(m, i, o):
            cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()

        hh = self._layers[self.layer].register_forward_hook(grab)
        try:
            self.ad.forward(board, attn)
        finally:
            hh.remove()
        return cap["h"][0].float().mean(0)                            # [H], mean over tokens

    @torch.no_grad()
    def compute(self, seeds=SEED_PROMPTS) -> dict:
        out, norms = {}, []
        for name, ax in AXES.items():
            pos = torch.stack([self._resid(ax["pos"] + "\n\n" + s) for s in seeds]).mean(0)
            neg = torch.stack([self._resid(ax["neg"] + "\n\n" + s) for s in seeds]).mean(0)
            norms += [float(pos.norm()), float(neg.norm())]
            d = pos - neg
            self.vecs[name] = d / (d.norm() + 1e-8)
            out[name] = round(float(d.norm()), 2)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.85 * self.resid_norm
        return {"raw_norms": out, "resid_norm": round(self.resid_norm, 1), "base": round(self.base, 2)}


class EngineSteer:
    """The SAME tone dials, but on the C++ engine (cloze-server) instead of the HF backbone -- so they
    work on ANY GGUF the engine has loaded, with no PyTorch and no hooks. An axis is a contrastive
    direction from /harvest (diff-in-means on the +pole/-pole prompts, unit-normalized); generation
    applies the active dials by summing them into one vector pushed through /intervene. Calibrated like
    SteeringControl but for the engine's residual scale: base = 0.08*|resid| (validated in
    engine_steer_spike -- slider 1.0 clearly-on and coherent). No discovery, no SAE: pure activation
    arithmetic, which is exactly why it ports to any model."""

    def __init__(self, engine_client, layer=14):
        self.ec = engine_client
        self.layer = layer
        self.vecs = {}                  # axis -> unit np.ndarray [n_embd]
        self.strength = {}              # axis -> slider value
        self.base, self.resid_norm = 1.0, 0.0
        self.ready = False

    def compute(self, seeds=SEED_PROMPTS) -> dict:
        norms = []
        for name, ax in AXES.items():
            pos = np.mean([self.ec.harvest(ax["pos"] + "\n\n" + s, layer=self.layer).activations.mean(0)
                           for s in seeds], axis=0)
            neg = np.mean([self.ec.harvest(ax["neg"] + "\n\n" + s, layer=self.layer).activations.mean(0)
                           for s in seeds], axis=0)
            norms += [float(np.linalg.norm(pos)), float(np.linalg.norm(neg))]
            d = pos - neg
            self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.08 * self.resid_norm
        self.ready = True
        return {"resid_norm": round(self.resid_norm, 1), "base": round(self.base, 1), "axes": list(self.vecs)}

    def set(self, name, value):
        mx = AXES.get(name, {}).get("max", 1.5)      # cap per-axis (cognitive axes degenerate past their sweet spot)
        self.strength[name] = max(-mx, min(mx, float(value)))

    @staticmethod
    def _text(r):
        ch = r.get("choices") if isinstance(r, dict) else None
        if ch:
            return ch[0].get("text") or ch[0].get("message", {}).get("content") or ""
        return (r.get("text") or "") if isinstance(r, dict) else str(r)

    def generate(self, prompt, strength=None, max_new=70):
        """Generate through the engine with the active dials applied (the no-dial path is a plain
        completion, so this doubles as the baseline for an A/B)."""
        if not self.ready:
            self.compute()
        s = self.strength if strength is None else strength
        active = {k: v for k, v in s.items() if v and k in self.vecs}
        if not active:
            return self._text(self.ec.complete(prompt, max_tokens=max_new))
        vec = np.zeros_like(next(iter(self.vecs.values())))
        for k, v in active.items():
            vec = vec + self.base * float(v) * self.vecs[k]
        return self._text(self.ec.intervene(prompt, vector=vec.tolist(), coef=1.0,
                                             layer=self.layer, max_tokens=max_new))

    def steer_vector(self, strength):
        """The summed, pre-scaled tone direction for the active dials (or None) -- so another engine call
        can apply tone alongside something else (e.g. /v1/completions WITH a memory prefix: tone + memory)."""
        if not self.ready:
            self.compute()
        active = {k: v for k, v in (strength or {}).items() if v and k in self.vecs}
        if not active:
            return None
        vec = np.zeros_like(next(iter(self.vecs.values())))
        for k, v in active.items():
            vec = vec + self.base * float(v) * self.vecs[k]
        n = float(np.linalg.norm(vec))            # cap the blend so several dials at once can't over-crank to garbage
        cap = self.base * 1.3                      # ~the single-dial coherent ceiling (slider 1.3)
        if n > cap:
            vec = vec * (cap / n)
        return vec.tolist()
