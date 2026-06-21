"""The pass loop (build-order step 4): adapter + policy + events, end-to-end ugly.

Whole-sequence fixed(T) denoising: full board, full bidirectional attention,
full recompute every pass — no blocks, no cache, no adaptive stopping. Those
arrive as refactors (DESIGN §5.3–§5.5) against the golden outputs this loop
pins. Invariant 4 lives here: this loop writes tokens to the board and never
touches KV; the adapter computes KV and never touches the board.

Also hosts ``sample_candidates``, the numpy reference of the §4.3
confidence-select kernel contract: per masked position, the sampled token and
its post-temperature probability (max-prob confidence; margin/entropy variants
are DESIGN open question #3).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from cloze_lab.models.base import FloatArray, IntArray, ModelAdapter
from cloze_lab.scheduler.events import (
    BlockFinalized,
    BlockStarted,
    CommitItem,
    Event,
    GenFinished,
    GenStarted,
    ReviseItem,
    StepStats,
    TokensCommitted,
    TokensRevised,
)
from cloze_lab.scheduler.blocks import BlockPlan, attention_mask
from cloze_lab.scheduler.cache import CacheConfig, CacheManager
from cloze_lab.scheduler.policies import Candidate, ConfidenceTopK, RevisionPolicy, UnmaskPolicy
from cloze_lab.scheduler.stepper import FixedStepper, StepController, StepOutcome


@dataclass(frozen=True, slots=True)
class GenerateConfig:
    """Knobs of the naive loop; block/cache/effort knobs join with §5.3–§5.5."""

    max_new: int
    steps: int  # fixed(T), per block
    temperature: float = 0.0  # 0 = greedy
    seed: int = 0
    block_len: int = 0  # 0 = whole-sequence; L > 0 = semi-AR blocks (DESIGN §5.4)

    def __post_init__(self) -> None:
        if self.max_new < 1:
            raise ValueError(f"max_new must be >= 1, got {self.max_new}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.block_len < 0:
            raise ValueError(f"block_len must be >= 0, got {self.block_len}")


@dataclass(frozen=True, slots=True)
class GenerateResult:
    board: IntArray
    text: str
    events: tuple[Event, ...]


def sample_candidates(
    logits: FloatArray,
    positions: Sequence[int],
    *,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[Candidate]:
    """Logits rows -> (token, confidence) per position; §4.3's CPU reference.

    Greedy (temperature=0): argmax token, confidence = its raw softmax prob.
    Sampled (temperature>0): draw from softmax(logits/T) with ``rng``;
    confidence = the drawn token's post-temperature probability.
    """
    if len(positions) != logits.shape[0]:
        raise ValueError(f"{len(positions)} positions but {logits.shape[0]} logits rows")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature > 0 and rng is None:
        raise ValueError("sampling (temperature > 0) requires an rng")

    x = logits.astype(np.float64)
    if temperature > 0:
        x = x / temperature
    x -= x.max(axis=1, keepdims=True)
    probs = np.exp(x)
    probs /= probs.sum(axis=1, keepdims=True)

    out: list[Candidate] = []
    for row, pos in enumerate(positions):
        if temperature == 0:
            token = int(probs[row].argmax())
        else:
            token = int(rng.choice(probs.shape[1], p=probs[row]))
        out.append(Candidate(pos=int(pos), token_id=token, confidence=float(probs[row, token])))
    return out


def truncate_at_eos(span: Sequence[int], eos_token_id: int | None) -> list[int]:
    """Generated ids up to (excluding) the first EOS; the whole span if none."""
    ids = [int(i) for i in span]
    if eos_token_id is None or eos_token_id not in ids:
        return ids
    return ids[: ids.index(eos_token_id)]


def generate(
    adapter: ModelAdapter,
    prompt_ids: Sequence[int],
    config: GenerateConfig,
    *,
    policy: UnmaskPolicy | None = None,
    stepper: StepController | None = None,
    cache: CacheConfig | None = None,
    reviser: RevisionPolicy | None = None,
    on_event: Callable[[Event], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerateResult:
    """Denoise ``max_new`` masked slots after the prompt.

    ``stepper`` controls how many passes run (DESIGN §5.3); it defaults to
    ``FixedStepper(config.steps)`` — exactly the original fixed(T) loop. ``cache``
    controls K/V reuse (§5.5); it defaults to ``CacheConfig(mode="off")`` — full
    recompute every pass, exact. Pass ``mode="delta"`` for Tier C reuse. Emits the
    §5.1 event stream (returned on the result and streamed to ``on_event``).
    ``clock`` is injectable so golden tests can pin timing exactly.

    ``reviser`` (§5.2 ``remask_lowconf``) is an optional revision policy: each step it
    may re-mask already-committed *active-block* tokens whose recomputed confidence
    fell, freeing them to be re-predicted (the "model changes its mind" feature). It
    is strictly opt-in — with ``reviser=None`` the commit path is byte-identical to
    before, so the goldens are untouched. Revisions are confined to the active block,
    so frozen blocks (Tier A/B) are never retracted and stay exactly cached.
    """
    prompt = [int(i) for i in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must be non-empty")
    if policy is None:
        policy = ConfidenceTopK()
    if stepper is None:
        stepper = FixedStepper(config.steps)
    cache_mgr = CacheManager(config=cache or CacheConfig())
    mcfg = adapter.config
    rng = np.random.default_rng(config.seed)

    p = len(prompt)
    n = p + config.max_new
    board = np.array(prompt + [mcfg.mask_token_id] * config.max_new, dtype=np.int64)
    plan = BlockPlan(prompt_len=p, max_new=config.max_new, block_len=config.block_len)
    block_list = plan.blocks()
    mask = mcfg.mask_token_id
    eos = mcfg.eos_token_id

    events: list[Event] = []

    def emit(event: Event) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    def has_eos(lo: int, hi: int) -> bool:
        return eos is not None and eos in board[lo:hi]

    t_start = clock()
    emit(GenStarted(t=0, prompt_tokens=p, block_len=config.block_len, max_new=config.max_new))

    t = 0  # global pass counter, monotonic across blocks
    last_t = 0
    gen_end = p  # exclusive end of the region we have actually generated
    revision_counts: dict[int, int] = {}  # per-position lifetime re-masks (reviser cap)

    for block in block_list:
        emit(BlockStarted(t=t, block=block.index, span=block.span))
        working_len = block.end
        attn = attention_mask(working_len, p, config.block_len)
        block_steps = 0
        for step in range(stepper.steps_cap):
            masked = [i for i in range(block.start, block.end) if board[i] == mask]
            if not masked:
                break
            t0 = clock()
            view = board[:working_len]
            active = (block.start, block.end)
            frozen_prefix = config.block_len > 0  # block-causal => prefix is exactly frozen
            # When revising, also score the active block's already-committed tokens so
            # the reviser can see their recomputed confidence. Committed positions are
            # always within the active block, so frozen blocks are never touched.
            committed_active = (
                [i for i in range(block.start, block.end) if board[i] != mask]
                if reviser is not None
                else []
            )
            want = sorted(masked + committed_active)
            plan = cache_mgr.plan(view, active=active, block_step=step, frozen_prefix=frozen_prefix)
            fwd = adapter.forward(
                view, attn, kv=plan.kv, recompute_kv=plan.recompute_kv, logits_for=want
            )
            cache_mgr.observe(view, fwd.kv, plan.recompute_kv, active=active, frozen_prefix=frozen_prefix)
            ctx = stepper.context(step)
            cands = sample_candidates(fwd.logits, want, temperature=config.temperature, rng=rng)
            masked_set = set(masked)
            selection = policy.select(
                [c for c in cands if c.pos in masked_set], ctx
            )
            for c in selection.commit:
                board[c.pos] = c.token_id

            # Revision (§5.2): re-mask low-confidence committed tokens in the active
            # block. Re-masked positions look "changed" to the cache, so they recompute
            # next pass — the same Tier C path committed tokens already use.
            revised: tuple[ReviseItem, ...] = ()
            if reviser is not None:
                to_revise = reviser.revisions(
                    [c for c in cands if c.pos not in masked_set], ctx, revision_counts
                )
                items = []
                for c in to_revise:
                    items.append(ReviseItem(pos=c.pos, old=int(board[c.pos]), id=c.token_id, conf=c.confidence))
                    board[c.pos] = mask
                    revision_counts[c.pos] = revision_counts.get(c.pos, 0) + 1
                revised = tuple(items)
                if revised:
                    emit(TokensRevised(t=t, block=block.index, items=revised))

            block_steps = step + 1
            ms = (clock() - t0) * 1000.0
            remaining = sum(1 for i in range(block.start, block.end) if board[i] == mask)
            emit(
                TokensCommitted(
                    t=t,
                    block=block.index,
                    items=tuple(
                        CommitItem(pos=c.pos, id=c.token_id, conf=c.confidence)
                        for c in selection.commit
                    ),
                )
            )
            emit(
                StepStats(
                    t=t, block=block.index, step=step,
                    committed=len(selection.commit), remaining=remaining,
                    ms=ms, cache_hit=plan.cache_hit,
                )
            )
            last_t = t
            t += 1
            if not stepper.should_continue(
                StepOutcome(step=step, n_committed=len(selection.commit), n_masked_after=remaining)
            ):
                break

        gen_end = block.end
        block_text = adapter.decode(truncate_at_eos(board[block.start:block.end], eos))
        emit(BlockFinalized(t=last_t, block=block.index, text=block_text, steps_used=block_steps))

        # EOS commits finish the run; remaining blocks are intentionally not generated.
        if has_eos(p, gen_end) and block is not block_list[-1]:
            break

    kept = truncate_at_eos(board[p:gen_end], eos)
    masked_in_gen = int((board[p:gen_end] == mask).sum())
    if has_eos(p, gen_end):
        reason = "eos"  # the model's explicit stop signal wins over leftover holes
    elif masked_in_gen:
        reason = "steps_exhausted"  # a block's step budget left holes, no EOS
    else:
        reason = "length"
    text = adapter.decode(kept)
    wall_ms = (clock() - t_start) * 1000.0
    tok_per_s = len(kept) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0

    emit(
        GenFinished(
            t=last_t,
            reason=reason,
            new_tokens=len(kept),
            wall_ms=wall_ms,
            steps_total=t,
            tok_per_s=tok_per_s,
        )
    )
    return GenerateResult(board=board, text=text, events=tuple(events))


def infill(
    adapter: ModelAdapter,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    gap: int,
    config: GenerateConfig,
    *,
    policy: UnmaskPolicy | None = None,
    stepper: StepController | None = None,
    reviser: RevisionPolicy | None = None,
    on_event: Callable[[Event], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerateResult:
    """Fill ``gap`` masked slots *between* a prefix and a suffix — native dLLM infilling.

    This is a capability autoregressive models structurally lack: the board
    is ``prefix + [MASK]*gap + suffix`` and the masked middle is denoised under **full
    bidirectional attention**, so every filled slot sees the fixed right-context
    (``suffix``) as well as the left. Whole-sequence and full-recompute (exact) — no
    blocks or KV cache; infill is a one-shot fill where correctness, not cache reuse,
    is the point. ``reviser`` works here too (the model can re-mask a low-confidence
    fill and reconsider it with both sides in view).

    Returns a ``GenerateResult`` whose ``board`` is the whole ``prefix+fill+suffix``
    sequence and whose ``text`` is the decoded fill (just the gap).
    """
    prefix = [int(i) for i in prefix_ids]
    suffix = [int(i) for i in suffix_ids]
    if gap < 1:
        raise ValueError(f"gap must be >= 1, got {gap}")
    if not prefix and not suffix:
        raise ValueError("infill needs a prefix or a suffix for context")
    if policy is None:
        policy = ConfidenceTopK()
    if stepper is None:
        stepper = FixedStepper(config.steps)
    mcfg = adapter.config
    mask = mcfg.mask_token_id
    rng = np.random.default_rng(config.seed)

    board = np.array(prefix + [mask] * gap + suffix, dtype=np.int64)
    n = board.shape[0]
    lo, hi = len(prefix), len(prefix) + gap  # the fill region [lo, hi)
    attn = np.ones((n, n), dtype=bool)  # full bidirectional: the gap sees both sides

    events: list[Event] = []

    def emit(event: Event) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    t_start = clock()
    emit(GenStarted(t=0, prompt_tokens=len(prefix), block_len=0, max_new=gap))
    emit(BlockStarted(t=0, block=0, span=(lo, hi)))

    revision_counts: dict[int, int] = {}
    t = 0
    last_t = 0
    steps_used = 0
    for step in range(stepper.steps_cap):
        masked = [i for i in range(lo, hi) if board[i] == mask]
        if not masked:
            break
        t0 = clock()
        committed_gap = (
            [i for i in range(lo, hi) if board[i] != mask] if reviser is not None else []
        )
        want = sorted(masked + committed_gap)
        fwd = adapter.forward(board, attn, logits_for=want)
        ctx = stepper.context(step)
        cands = sample_candidates(fwd.logits, want, temperature=config.temperature, rng=rng)
        masked_set = set(masked)
        selection = policy.select([c for c in cands if c.pos in masked_set], ctx)
        for c in selection.commit:
            board[c.pos] = c.token_id

        if reviser is not None:
            to_revise = reviser.revisions(
                [c for c in cands if c.pos not in masked_set], ctx, revision_counts
            )
            items = []
            for c in to_revise:
                items.append(ReviseItem(pos=c.pos, old=int(board[c.pos]), id=c.token_id, conf=c.confidence))
                board[c.pos] = mask
                revision_counts[c.pos] = revision_counts.get(c.pos, 0) + 1
            if items:
                emit(TokensRevised(t=t, block=0, items=tuple(items)))

        steps_used = step + 1
        ms = (clock() - t0) * 1000.0
        remaining = sum(1 for i in range(lo, hi) if board[i] == mask)
        emit(
            TokensCommitted(
                t=t, block=0,
                items=tuple(CommitItem(pos=c.pos, id=c.token_id, conf=c.confidence) for c in selection.commit),
            )
        )
        emit(StepStats(t=t, block=0, step=step, committed=len(selection.commit), remaining=remaining, ms=ms, cache_hit=0.0))
        last_t = t
        t += 1
        if not stepper.should_continue(
            StepOutcome(step=step, n_committed=len(selection.commit), n_masked_after=remaining)
        ):
            break

    fill_text = adapter.decode(board[lo:hi])
    emit(BlockFinalized(t=last_t, block=0, text=fill_text, steps_used=steps_used))

    remaining = int((board[lo:hi] == mask).sum())
    reason = "steps_exhausted" if remaining else "length"
    wall_ms = (clock() - t_start) * 1000.0
    filled = gap - remaining
    tok_per_s = filled / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
    emit(
        GenFinished(
            t=last_t, reason=reason, new_tokens=filled,
            wall_ms=wall_ms, steps_total=t, tok_per_s=tok_per_s,
        )
    )
    return GenerateResult(board=board, text=fill_text, events=tuple(events))
