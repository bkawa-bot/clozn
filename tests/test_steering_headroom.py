"""Pure-logic tests for the combined-steering headroom cap in the HF steering adapter.

The bug: a stack of strong tone dials sums to a delta big enough to saturate the residual stream, shoving
the model off-distribution and muting subtler biases (the learned-memory soft-prefix). The honest fix is
SteeringControl._cap_delta -- it bounds ||sum-of-dials|| to MAX_DELTA_FRAC * ||residual|| PER POSITION,
direction untouched. These tests exercise ONLY that norm math on fake tensors: no model, no tokenizer, no
GPU. We call _cap_delta as an unbound method against a tiny stand-in that carries just MAX_DELTA_FRAC, so
importing the steering package is all that's needed.

Invariants asserted:
  * a SMALL delta (one moderate dial) is passed through byte-for-byte unchanged -- the cap must not bite;
  * a LARGE summed delta is rescaled so its per-position norm == k * ||residual|| (the ceiling), and its
    DIRECTION is preserved (only magnitude changes);
  * the cap is per-position: with residuals of different norms, each position gets its own budget.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
import clozn.behavior.steering as steering  # noqa: E402

K = steering.SteeringControl.MAX_DELTA_FRAC


class _Stub:
    """Minimal carrier: _cap_delta only reads self.MAX_DELTA_FRAC and the two tensor args."""
    MAX_DELTA_FRAC = K


def cap(add, h):
    return steering.SteeringControl._cap_delta(_Stub(), add, h)


def test_small_delta_is_untouched():
    # residual of norm ~10; a lone moderate dial pushing with norm ~1 (well under k*10) must pass through.
    H = 16
    h = torch.ones(1, 1, H)                       # ||h|| = sqrt(16) = 4.0
    add = torch.zeros(H)
    add[0] = 0.5                                   # ||add|| = 0.5, ceiling = k*4 = 3.0 -> no clamp
    out = cap(add, h)
    assert torch.allclose(out, add), "a small single-dial delta must not be rescaled"


def test_large_delta_rescaled_to_ceiling():
    # A big summed push: rescaled so its per-position norm == k * ||residual||, direction preserved.
    H = 8
    h = torch.full((1, 1, H), 3.0)                # ||h|| = sqrt(8)*3 = 8.4853
    resid_norm = float(h.norm(dim=-1)[0, 0])
    add = torch.full((H,), 5.0)                   # ||add|| = sqrt(8)*5 = 14.14 >> ceiling
    out = cap(add, h)[0, 0]                        # [..., H] -> [H] at the single position
    ceiling = K * resid_norm
    assert torch.isclose(out.norm(), torch.tensor(ceiling), atol=1e-4), \
        f"capped norm {float(out.norm())} != ceiling {ceiling}"
    # direction unchanged: capped delta is a positive scalar multiple of the original.
    cos = torch.dot(out, add) / (out.norm() * add.norm())
    assert torch.isclose(cos, torch.tensor(1.0), atol=1e-5), "direction must be preserved by the cap"


def test_cap_is_per_position():
    # Two positions with different residual norms get different budgets from the SAME summed delta.
    H = 4
    h = torch.stack([torch.full((H,), 1.0),       # ||h0|| = 2.0
                     torch.full((H,), 4.0)]).unsqueeze(0)  # ||h1|| = 8.0  -> shape [1, 2, H]
    add = torch.full((H,), 3.0)                   # ||add|| = 6.0
    out = cap(add, h)[0]                           # [2, H]
    n0, n1 = float(out[0].norm()), float(out[1].norm())
    # position 0: ceiling = k*2 = 1.5 < 6 -> clamped to 1.5; position 1: ceiling = k*8 = 6.0 == ||add|| -> untouched.
    assert torch.isclose(torch.tensor(n0), torch.tensor(K * 2.0), atol=1e-4)
    assert torch.isclose(torch.tensor(n1), torch.tensor(6.0), atol=1e-4)


def test_zero_delta_is_safe():
    # A degenerate all-zero push must not divide-by-zero; it stays zero.
    H = 4
    h = torch.full((1, 1, H), 2.0)
    add = torch.zeros(H)
    out = cap(add, h)
    assert torch.allclose(out, torch.zeros_like(add)), "zero delta must remain zero"
