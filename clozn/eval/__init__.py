"""clozn.eval -- OUTCOME-grounded evaluation + calibration.

The actuarial journal (clozn.runs.actuary) and calibrated_trust rest on an ACCEPTANCE PROXY: "did the
user/tests not reject this run". They say so plainly and never claim factual correctness. This package
adds the tier they flag as missing: calibration against TRUTH on a labeled eval set.

  * outcome.py      -- named matchers (exact / numeric / mcq) that turn (prediction, gold) into a hard
                       correct/incorrect bool. "Correct" is never a vibe -- each matcher reports HOW it
                       matched, and an ungradeable item is None (excluded from coverage, never counted wrong).
  * calibration.py  -- Brier, ECE (vs truth, not proxy), and a risk-coverage curve / AURC for SELECTIVE
                       generation. Pure over (score, correct) pairs; the score's provenance is the caller's.
  * probes.py       -- a small built-in factual probe set + a dependency-light runner, so `report` can be
                       computed on REAL model answers, not only synthetic pairs.

The honesty stance carries over: this is real correctness on THIS eval set, not a universal guarantee. The
score used for ranking (answer-span probability) is the same per-token confidence actuary bins by -- this
package never synthesizes an "overall confidence" number.
"""
