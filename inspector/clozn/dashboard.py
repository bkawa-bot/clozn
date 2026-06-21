"""
clozn.dashboard — ONE inspector entry across every substrate (Phase 2.4).

The whole point of the StateSource seam: a single code path inspects them all. This module
picks a substrate by selector, builds the right StateSource, then drives it through the SAME
`Spine` to the SAME viz consumers — no `if substrate == ...` in the drive loop. Adding a model
family is a new entry in `SUBSTRATES`, never a change to how the dashboard runs.

    selector            StateSource
    --------            -----------
    toy                 ToyRecurrentSource     (pure numpy, no deps, no network — the live demo)
    rwkv                RwkvStateSource         (RWKV-4 via transformers; gated — needs a checkpoint)
    engine-ar           EngineStateSource(substrate="autoregressive")   (engine over HTTP)
    engine-diffusion    EngineStateSource(substrate="diffusion")        (engine over HTTP)

The engine-* selectors are wired against the EngineStateSource interface; they need a running
server only at run time. The drive + render path is identical for all four.

    python -m clozn.dashboard --source toy --prompt "abacus"
    python -m clozn.dashboard --source rwkv --prompt "The capital of France is Paris."
    python -m clozn.dashboard --source engine-ar --prompt "Hello" --base-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .spine import Spine, StateSource, StateStep
from .viz import _probe_svg, _statefilm_svg, render_dashboard

# A default toy vocabulary that covers a friendly demo prompt out of the box (every char a token).
_TOY_VOCAB = list("abcdefghijklmnopqrstuvwxyz .,!?")


# --------------------------------------------------------------------------------------------------
# Substrate registry — selector -> (build a StateSource, human label). THE seam: one entry per family.
# --------------------------------------------------------------------------------------------------
def _build_toy(prompt: str, **opts: Any) -> StateSource:
    from .sources.toy_recurrent import ToyRecurrentSource
    # widen the vocab so any prompt character is a valid token (toy keys are per-symbol)
    vocab = sorted(set(_TOY_VOCAB) | set(prompt or "abc")) or list("abc")
    return ToyRecurrentSource(vocab, d=int(opts.get("d", 16)), seed=int(opts.get("seed", 0)))


def _build_rwkv(prompt: str, **opts: Any) -> StateSource:
    from .sources.hf_rwkv import RwkvStateSource
    return RwkvStateSource(name=opts.get("model", "RWKV/rwkv-4-169m-pile"),
                           device=opts.get("device", "cpu"))


def _build_engine(substrate: str) -> Callable[..., StateSource]:
    def build(prompt: str, **opts: Any) -> StateSource:
        from .sources.engine import DEFAULT_BASE_URL, EngineStateSource
        return EngineStateSource(base_url=opts.get("base_url", DEFAULT_BASE_URL),
                                 substrate=substrate,
                                 **{k: v for k, v in opts.items() if k not in ("base_url", "model", "device", "d", "seed")})
    return build


@dataclass
class Substrate:
    """One registry entry: how to build the source + how it reads (used only for the OPTIONAL views,
    never for the core drive loop)."""
    build: Callable[..., StateSource]
    label: str
    component: str | None = None          # the canonical state component for probe/feature views
    probeable: bool = False               # exposes the RwkvStateSource probe surface (.tok, ._last_logits)


SUBSTRATES: dict[str, Substrate] = {
    "toy":              Substrate(_build_toy, "toy delta-rule recurrent memory (numpy)", component="S"),
    "rwkv":             Substrate(_build_rwkv, "RWKV-4-169m recurrent state (transformers)",
                                  component="att_num", probeable=True),
    "engine-ar":        Substrate(_build_engine("autoregressive"),
                                  "Clozn engine · autoregressive (HTTP)", component="hidden"),
    "engine-diffusion": Substrate(_build_engine("diffusion"),
                                  "Clozn engine · diffusion canvas (HTTP)", component="board"),
}


def list_substrates() -> list[str]:
    return list(SUBSTRATES)


def build_source(selector: str, prompt: str = "", **opts: Any) -> StateSource:
    """Dispatch a substrate selector to the right StateSource. This is the seam, in one function."""
    if selector not in SUBSTRATES:
        raise ValueError(f"unknown substrate {selector!r}; choose one of {sorted(SUBSTRATES)}")
    return SUBSTRATES[selector].build(prompt, **opts)


# --------------------------------------------------------------------------------------------------
# Prompt -> step inputs.  Source-agnostic: ask the source how it consumes a prompt; the DRIVE LOOP
# never branches on substrate. (RWKV/recurrent step on token ids; the toy steps on symbols; the
# engine steps on the whole prompt string in one shot.)
# --------------------------------------------------------------------------------------------------
def prepare_inputs(source: StateSource, prompt: str, *, max_steps: int = 24) -> list[Any]:
    """Turn `prompt` into the list of per-step inputs `Spine.run` will feed `source.step`.

    Uses only capabilities the source advertises (duck-typed off the seam), so there is no
    substrate switch here either:
      - a tokenizer (`encode`)      -> step on token ids   (RWKV and friends)
      - a per-symbol vocab (`vocab`)-> step on symbols      (the toy)
      - otherwise                   -> one step on the whole prompt (the engine: one run = one step)
    """
    enc = getattr(source, "encode", None)
    if callable(enc):
        ids = list(enc(prompt))
        return ids[:max_steps] if max_steps else ids
    vocab = getattr(source, "vocab", None)
    if vocab is not None:
        syms = [c for c in prompt if c in set(vocab)]
        return syms[:max_steps] if max_steps else syms
    return [prompt]                                  # engine-style: a single step drives a whole run


# --------------------------------------------------------------------------------------------------
# The consumer that captures the stream (a pure observer — invariant 4: it never owns the state).
# --------------------------------------------------------------------------------------------------
@dataclass
class Recorder:
    steps: list[StateStep] = field(default_factory=list)

    def on_step(self, step: StateStep) -> None:
        self.steps.append(step.copy())               # copy: consumers read, never mutate the source


# --------------------------------------------------------------------------------------------------
# The unified entry.
# --------------------------------------------------------------------------------------------------
def _runs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def _probe_panel(source: StateSource, component: str) -> tuple[str, str] | None:
    """OPTIONAL Probe + Verify view — only for sources that expose the recurrent probe surface
    (.tok / ._last_logits). Returns None (gracefully) for substrates without it, so the dashboard
    degrades instead of special-casing."""
    try:
        from .probes import probe_and_verify
        r = probe_and_verify(source, name="sentiment", component=component)
    except Exception:
        return None
    label = "PROBE + VERIFY · is ‘sentiment’ decodable from the state, and causal?"
    svg = _probe_svg(r.alphas, r.scores, r.decodability, r.verify, "sentiment",
                     "Clozn · Probe + Verify", f"{component}")
    return (label, svg)


def dashboard(selector: str, prompt: str, *, out: str | None = None,
              do_probe: bool = True, max_steps: int = 24, **opts: Any) -> str:
    """Drive ANY substrate through the Spine to the viz, and write one dashboard HTML. Returns the path.

    ONE source-agnostic code path: build the source for `selector`, run it through `Spine` with a
    `Recorder` consumer, render the captured stream. The Watch view is substrate-agnostic (it reads
    only the StateStep contract); Probe is added when the substrate supports it. No branch in here
    keys on the substrate — that is the seam doing its job.
    """
    if selector not in SUBSTRATES:
        raise ValueError(f"unknown substrate {selector!r}; choose one of {sorted(SUBSTRATES)}")
    sub = SUBSTRATES[selector]

    source = build_source(selector, prompt, **opts)
    inputs = prepare_inputs(source, prompt, max_steps=max_steps)

    # --- the one drive loop: Spine fans every StateStep to the consumers (identical for all sources)
    rec = Recorder()
    list(Spine(source, [rec]).run(inputs))           # drain the generator; Recorder captured the run
    steps = rec.steps

    panels: list[tuple[str, str]] = []
    panels.append((f"WATCH · state-write intensity per step  ·  substrate: {sub.label}",
                   _watch_svg(steps)))

    if do_probe and sub.probeable and sub.component:
        panel = _probe_panel(source, sub.component)
        if panel:
            panels.append(panel)

    html = render_dashboard(panels, title="Clozn · Inspector",
                            subtitle=f"{selector} · reading {prompt!r}  ·  {len(steps)} steps")
    if out is None:
        out = os.path.join(_runs_dir(), f"dashboard_{selector}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def _watch_svg(steps: list[StateStep]) -> str:
    """The Watch panel's raw SVG (substrate-agnostic), for composing into the dashboard."""
    comps = sorted({k for s in steps for k in s.state})
    return _statefilm_svg(steps, comps, "Clozn · Watch", "")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="python -m clozn.dashboard",
        description="One inspector, every substrate: drive a substrate through the Spine to a dashboard.")
    ap.add_argument("--source", default="toy", choices=list_substrates(),
                    help="which substrate to inspect (default: toy — pure numpy, no deps)")
    ap.add_argument("--prompt", default="the quick brown fox jumps", help="prompt to read")
    ap.add_argument("--out", default=None, help="output HTML path (default: runs/dashboard_<source>.html)")
    ap.add_argument("--base-url", default=None, help="engine base URL for engine-* sources")
    ap.add_argument("--max-steps", type=int, default=24, help="cap the number of steps driven")
    ap.add_argument("--no-probe", action="store_true", help="skip the Probe+Verify panel")
    args = ap.parse_args(argv)

    opts: dict[str, Any] = {}
    if args.base_url:
        opts["base_url"] = args.base_url

    print(f"inspecting {args.prompt!r} via {args.source} ...")
    out = dashboard(args.source, args.prompt, out=args.out,
                    do_probe=not args.no_probe, max_steps=args.max_steps, **opts)
    print("wrote", out)


if __name__ == "__main__":
    main()
