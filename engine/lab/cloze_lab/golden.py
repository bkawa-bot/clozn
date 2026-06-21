"""Golden fixtures — the oracle (DESIGN invariant 3, build-order step 6).

A golden case pins (prompt, seed, policy) -> the exact per-step token picks and
their confidences for a known adapter. Replaying a case re-runs the loop and
compares: **picks must match exactly** (positions + token ids are discrete — any
drift is a real divergence), **confidences within epsilon** (float reduction
order differs across devices, so never assert bitwise equality on sums).

These same JSON files will validate the future C++ core: it reads the inputs,
runs its own loop, and must reproduce the picks exactly with confidences inside
epsilon. ``replay`` returns a structured ``ReplayReport`` rather than asserting,
so the §5.5/§8 divergence bench can reuse the same comparison primitive.

This module stays import-light (numpy only); the Dream adapter is imported
lazily so the FakeAdapter oracle path needs no torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cloze_lab.generate import GenerateConfig, GenerateResult, generate
from cloze_lab.models.base import ModelAdapter
from cloze_lab.scheduler.cache import CacheConfig
from cloze_lab.scheduler.events import GenFinished, StepStats, TokensCommitted
from cloze_lab.scheduler.policies import ConfidenceTopK, Threshold, UnmaskPolicy
from cloze_lab.scheduler.stepper import AdaptiveStepper, FixedStepper, StepController

FORMAT = "cloze-golden/1"
DEFAULT_CONF_EPSILON = 1e-6


# --------------------------------------------------------------------------- #
# Serializable case structure (mirrors the §5.1 event payloads verbatim)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class GoldenCommit:
    pos: int
    id: int
    conf: float


@dataclass(frozen=True, slots=True)
class GoldenStep:
    step: int
    commit: list[GoldenCommit]
    remaining: int


@dataclass(frozen=True, slots=True)
class GoldenModel:
    """Identity of the adapter the case was recorded under; replay rebuilds it."""

    adapter: str  # "FakeAdapter" | "DreamAdapter"
    family: str
    vocab_size: int
    mask_token_id: int
    eos_token_id: int | None
    adapter_args: dict  # kwargs to reconstruct the adapter (FakeAdapter) or LoadConfig (Dream)


@dataclass(frozen=True, slots=True)
class GoldenCase:
    name: str
    model: GoldenModel
    prompt_text: str
    prompt_ids: list[int]
    config: dict  # GenerateConfig fields
    policy: dict  # {"name": ..., **params}
    steps: list[GoldenStep]
    final: dict  # board, text, reason, new_tokens, steps_total
    stepper: dict | None = None  # {"name": "fixed"|"adaptive", ...}; None => fixed(config.steps)
    cache: dict | None = None  # CacheConfig fields; None => mode "off"
    format: str = FORMAT

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @staticmethod
    def from_dict(d: dict) -> GoldenCase:
        if d.get("format") != FORMAT:
            raise ValueError(f"unknown golden format {d.get('format')!r}, expected {FORMAT!r}")
        model = GoldenModel(**d["model"])
        steps = [
            GoldenStep(
                step=s["step"],
                commit=[GoldenCommit(**c) for c in s["commit"]],
                remaining=s["remaining"],
            )
            for s in d["steps"]
        ]
        return GoldenCase(
            name=d["name"],
            model=model,
            prompt_text=d["prompt_text"],
            prompt_ids=d["prompt_ids"],
            config=d["config"],
            policy=d["policy"],
            steps=steps,
            final=d["final"],
            stepper=d.get("stepper"),  # absent in v1-era fixtures => fixed(config.steps)
            cache=d.get("cache"),  # absent => mode "off"
            format=d["format"],
        )

    @staticmethod
    def read(path: str | Path) -> GoldenCase:
        return GoldenCase.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Reconstruction (adapter + policy from a recorded spec)
# --------------------------------------------------------------------------- #
def build_policy(spec: dict) -> UnmaskPolicy:
    name = spec["name"]
    if name == "confidence_topk":
        return ConfidenceTopK(k=spec.get("k"))
    if name == "threshold":
        return Threshold(tau=spec["tau"], min_commit=spec.get("min_commit", 1))
    raise ValueError(f"unknown policy {name!r}")


def build_stepper(spec: dict | None, config: GenerateConfig) -> StepController:
    """Reconstruct the step controller. None (pre-stepper fixtures) => fixed(config.steps)."""
    if spec is None or spec["name"] == "fixed":
        return FixedStepper(steps=spec["steps"] if spec else config.steps)
    if spec["name"] == "adaptive":
        return AdaptiveStepper(t_max=spec["t_max"])
    raise ValueError(f"unknown stepper {spec['name']!r}")


def build_cache(spec: dict | None) -> CacheConfig:
    """Reconstruct the cache config. None (pre-cache fixtures) => mode 'off'."""
    return CacheConfig(**spec) if spec else CacheConfig()


def build_adapter(model: GoldenModel) -> ModelAdapter:
    if model.adapter == "FakeAdapter":
        from cloze_lab.models.fake import FakeAdapter

        return FakeAdapter(**model.adapter_args)
    if model.adapter == "DreamAdapter":
        from cloze_lab.models.base import LoadConfig
        from cloze_lab.models.dream import DreamAdapter

        return DreamAdapter(LoadConfig(**model.adapter_args))
    if model.adapter == "OpenDCoderAdapter":
        # Dream-family sibling, loaded via the open_dcoder_adapter factory (stock
        # Qwen2ForCausalLM + explicit mask token). The tiny CPU-CI checkpoint.
        from cloze_lab.models.base import LoadConfig
        from cloze_lab.models.dream import open_dcoder_adapter

        return open_dcoder_adapter(LoadConfig(**model.adapter_args))
    raise ValueError(f"unknown adapter {model.adapter!r}")


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def _steps_from_result(result: GenerateResult) -> list[GoldenStep]:
    """Fold the event stream into per-step golden rows (consumer-only)."""
    commits: dict[int, list[GoldenCommit]] = {}
    remaining: dict[int, int] = {}
    for event in result.events:
        if isinstance(event, TokensCommitted):
            commits[event.t] = [
                GoldenCommit(pos=i.pos, id=i.id, conf=i.conf) for i in event.items
            ]
        elif isinstance(event, StepStats):
            remaining[event.t] = event.remaining  # key by global t (per-block step repeats)
    return [
        GoldenStep(step=t, commit=commits[t], remaining=remaining[t])
        for t in sorted(commits)
    ]


def record(
    name: str,
    adapter: ModelAdapter,
    adapter_spec: GoldenModel,
    prompt_text: str,
    config: GenerateConfig,
    *,
    policy_spec: dict,
    stepper_spec: dict | None = None,
    cache_spec: dict | None = None,
) -> GoldenCase:
    """Run the loop and capture its picks + confidences as a golden case.

    ``stepper_spec`` None records the fixed(config.steps) default explicitly;
    ``cache_spec`` None records mode 'off'.
    """
    prompt_ids = adapter.encode(prompt_text)
    if stepper_spec is None:
        stepper_spec = {"name": "fixed", "steps": config.steps}
    if cache_spec is None:
        cache_spec = {"mode": "off"}
    result = generate(
        adapter,
        prompt_ids,
        config,
        policy=build_policy(policy_spec),
        stepper=build_stepper(stepper_spec, config),
        cache=build_cache(cache_spec),
    )
    finished = result.events[-1]
    assert isinstance(finished, GenFinished)
    return GoldenCase(
        name=name,
        model=adapter_spec,
        prompt_text=prompt_text,
        prompt_ids=[int(i) for i in prompt_ids],
        config={
            "max_new": config.max_new,
            "steps": config.steps,
            "temperature": config.temperature,
            "seed": config.seed,
            "block_len": config.block_len,
        },
        policy=policy_spec,
        stepper=stepper_spec,
        cache=cache_spec,
        steps=_steps_from_result(result),
        final={
            "board": [int(i) for i in result.board],
            "text": result.text,
            "reason": finished.reason,
            "new_tokens": finished.new_tokens,
            "steps_total": finished.steps_total,
        },
    )


# --------------------------------------------------------------------------- #
# Replay + comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Mismatch:
    step: int
    kind: str  # "picks" | "confidence" | "remaining" | "final"
    detail: str


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Structured diff between a golden case and a fresh run (reusable by bench)."""

    picks_match: bool
    max_conf_delta: float
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.picks_match and not self.mismatches


def _picks(step: GoldenStep) -> list[tuple[int, int]]:
    return [(c.pos, c.id) for c in step.commit]


def replay(case: GoldenCase, adapter: ModelAdapter, *, conf_epsilon: float = DEFAULT_CONF_EPSILON) -> ReplayReport:
    """Re-run the case's inputs through ``adapter`` and diff against the oracle."""
    config = GenerateConfig(**case.config)
    fresh = record(
        case.name, adapter, case.model, case.prompt_text, config,
        policy_spec=case.policy, stepper_spec=case.stepper, cache_spec=case.cache,
    )

    mismatches: list[Mismatch] = []
    picks_match = True
    max_conf_delta = 0.0

    if len(fresh.steps) != len(case.steps):
        picks_match = False
        mismatches.append(
            Mismatch(-1, "picks", f"step count {len(fresh.steps)} != golden {len(case.steps)}")
        )
    for golden_step, fresh_step in zip(case.steps, fresh.steps):
        gp, fp = _picks(golden_step), _picks(fresh_step)
        if gp != fp:
            picks_match = False
            mismatches.append(Mismatch(golden_step.step, "picks", f"{fp} != golden {gp}"))
            continue  # confidence comparison is meaningless once picks diverge
        if golden_step.remaining != fresh_step.remaining:
            mismatches.append(
                Mismatch(
                    golden_step.step,
                    "remaining",
                    f"{fresh_step.remaining} != golden {golden_step.remaining}",
                )
            )
        for gc, fc in zip(golden_step.commit, fresh_step.commit):
            delta = abs(gc.conf - fc.conf)
            max_conf_delta = max(max_conf_delta, delta)
            if delta > conf_epsilon:
                mismatches.append(
                    Mismatch(
                        golden_step.step,
                        "confidence",
                        f"pos {gc.pos}: |{fc.conf} - {gc.conf}| = {delta:.3e} > {conf_epsilon:.0e}",
                    )
                )

    for key in ("board", "text", "reason", "new_tokens", "steps_total"):
        if fresh.final[key] != case.final[key]:
            mismatches.append(Mismatch(-1, "final", f"{key}: {fresh.final[key]!r} != golden {case.final[key]!r}"))

    return ReplayReport(picks_match=picks_match, max_conf_delta=max_conf_delta, mismatches=mismatches)


def assert_replay(case: GoldenCase, adapter: ModelAdapter | None = None, *, conf_epsilon: float = DEFAULT_CONF_EPSILON) -> None:
    """Replay and raise AssertionError with a readable diff on any divergence."""
    adapter = adapter or build_adapter(case.model)
    report = replay(case, adapter, conf_epsilon=conf_epsilon)
    if not report.ok:
        lines = "\n".join(f"  [{m.kind}] step {m.step}: {m.detail}" for m in report.mismatches)
        raise AssertionError(
            f"golden {case.name!r} diverged (max_conf_delta={report.max_conf_delta:.3e}):\n{lines}"
        )
