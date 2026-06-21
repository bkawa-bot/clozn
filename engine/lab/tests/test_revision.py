"""remask_lowconf revision policy (DESIGN §5.2) — the "model changes its mind" feature.

Torch-free: the RemaskLowConf policy is pure logic, and the end-to-end path runs on
the FakeAdapter oracle. Verifies revisions fire, respect the per-position cap (so the
loop terminates), stay confined to the generated region, and are strictly opt-in.
"""

import pytest

from cloze_lab.generate import GenerateConfig, generate
from cloze_lab.models.fake import FakeAdapter
from cloze_lab.scheduler.events import GenFinished, TokensRevised
from cloze_lab.scheduler.policies import Candidate, RemaskLowConf, RevisionPolicy, StepContext


class TestRemaskLowConfPolicy:
    def test_is_a_revision_policy(self) -> None:
        assert isinstance(RemaskLowConf(tau_revise=0.5), RevisionPolicy)

    def test_picks_below_tau_only(self) -> None:
        pol = RemaskLowConf(tau_revise=0.5)
        committed = [
            Candidate(pos=2, token_id=9, confidence=0.30),
            Candidate(pos=5, token_id=4, confidence=0.80),  # above tau — keep
            Candidate(pos=7, token_id=1, confidence=0.10),
        ]
        out = pol.revisions(committed, StepContext(step=0), {})
        assert [c.pos for c in out] == [2, 7]  # below tau, pos-ascending

    def test_respects_per_position_cap(self) -> None:
        pol = RemaskLowConf(tau_revise=0.5, max_revisions=2)
        committed = [Candidate(pos=2, token_id=9, confidence=0.3), Candidate(pos=7, token_id=1, confidence=0.1)]
        # pos 7 already hit the cap -> excluded; pos 2 still eligible.
        out = pol.revisions(committed, StepContext(step=0), {7: 2})
        assert [c.pos for c in out] == [2]

    def test_validates_params(self) -> None:
        with pytest.raises(ValueError):
            RemaskLowConf(tau_revise=1.5)
        with pytest.raises(ValueError):
            RemaskLowConf(tau_revise=0.5, max_revisions=0)


class TestRevisionEndToEnd:
    def test_revisions_fire_respect_cap_and_terminate(self) -> None:
        adapter = FakeAdapter(seed=7)
        prompt = adapter.encode("hello cloze")
        cfg = GenerateConfig(max_new=4, steps=12)
        # tau_revise=1.0 re-masks every committed token once (cap=1): a hard stress
        # test that the cap guarantees termination despite aggressive revision.
        result = generate(adapter, prompt, cfg, reviser=RemaskLowConf(tau_revise=1.0, max_revisions=1))

        revised = [e for e in result.events if isinstance(e, TokensRevised)]
        assert revised, "expected at least one revision"

        mask = adapter.config.mask_token_id
        p = len(prompt)
        per_pos: dict[int, int] = {}
        for ev in revised:
            for it in ev.items:
                assert it.old != mask  # a real, previously-committed token was retracted
                assert it.conf < 1.0  # below tau_revise
                assert p <= it.pos < p + cfg.max_new  # within the generated region only
                per_pos[it.pos] = per_pos.get(it.pos, 0) + 1
        assert all(v <= 1 for v in per_pos.values())  # max_revisions cap honored
        assert isinstance(result.events[-1], GenFinished)  # it terminated

    def test_reviser_is_opt_in(self) -> None:
        adapter = FakeAdapter(seed=7)
        prompt = adapter.encode("hello cloze")
        cfg = GenerateConfig(max_new=4, steps=8)
        # No reviser -> no revision events, byte-identical commit path (the goldens
        # are pinned to exactly this).
        result = generate(adapter, prompt, cfg)
        assert not [e for e in result.events if isinstance(e, TokensRevised)]
