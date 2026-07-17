"""Loader + validator for the matched-pair dial exemplar bank (dial_exemplars.json).

WHY THIS EXISTS. A dial's steering direction is a vector in the residual stream, derived by contrasting the
dial's two poles. The default recipe (clozn.behavior.steering.axes) contrasts ONE instruction pair
("Respond warmly" vs "Respond coldly") across a few seed prompts and mean-differences the result. That's
cheap and works on the Qwen2.5 family, but it carves a much fainter axis on some models -- measured: on
Qwen3.5-9B every dial's effect margin came back near-zero under the instruction recipe, while the same
harness steered Qwen2.5-14B strongly. The vectors were re-derived on Qwen3.5's own activations either way
(the raw norms differ per model, so nothing is "borrowed"); what didn't transfer was the RECIPE.

This bank is the stronger recipe: many MATCHED exemplar pairs per dial (the same message written at each
pole), difference-averaged -- the standard contrastive-activation-steering (CAA) construction. Two
independent knobs are being upgraded at once vs. the default:
  * stimulus:    real styled TEXT instead of an instruction *about* the style (captures the actual output
                 distribution, not the model's reading of a command -- which matters exactly when
                 instruction-following is what's weak);
  * aggregation: a mean over MANY pairs instead of one pair x a few seeds (robustness).
A third knob -- fitting a linear probe on the same pair activations instead of subtracting centroids --
composes with this bank unchanged: same data, different final aggregation. Left for later escalation.

THE MATCHED-PAIR RULE (the thing that makes or breaks it). Each pair must hold CONTENT fixed and vary only
the styled attribute. Random pos texts vs random neg texts differ in topic/length/structure too, and those
nuisance directions leak into the mean difference and pollute the axis. Matched rewrites cancel the content
and isolate the style -- that is the whole point of "contrastive". The instruction recipe gets this for free
(identical seed, opposite instruction); an exemplar bank only gets it if the pairs are deliberately matched.

Stdlib only; no torch, no model. Pure data + checks, so it stays importable anywhere and testable offline.

    python scripts/calibration/dial_exemplars.py --validate     # check the bank (run after editing)
    python scripts/calibration/dial_exemplars.py --list         # dial roster + pair counts
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "dial_exemplars.json")

# A dial with fewer than this many pairs still loads, but is flagged: the mean-difference is what buys the
# robustness, and averaging 2-3 pairs barely beats the single-instruction recipe it is meant to replace.
MIN_RECOMMENDED_PAIRS = 8


def load(path: str = DEFAULT_PATH) -> dict:
    """The raw bank dict. Raises (loudly) on a missing/malformed file -- this is a build-time asset, not a
    runtime input to degrade over: a silently-empty bank would quietly fall back to the weak recipe."""
    with open(path, encoding="utf-8") as f:
        bank = json.load(f)
    if not isinstance(bank, dict) or not isinstance(bank.get("dials"), dict):
        raise ValueError(f"{path!r}: expected an object with a 'dials' object")
    return bank


def validate(bank: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors are structural -- a consumer cannot derive a direction from this dial.
    Warnings are quality signals (too few pairs, suspiciously similar poles) that still load."""
    errors: list[str] = []
    warnings: list[str] = []
    dials = bank.get("dials", {})
    if not dials:
        errors.append("bank has no dials")
    for name, d in dials.items():
        where = f"dials.{name}"
        if not isinstance(d, dict):
            errors.append(f"{where}: not an object")
            continue
        poles = d.get("poles")
        if not (isinstance(poles, list) and len(poles) == 2 and all(isinstance(p, str) and p for p in poles)):
            errors.append(f"{where}.poles: need exactly two non-empty pole names [positive, negative]")
        for key in ("pos_instruction", "neg_instruction"):
            if not (isinstance(d.get(key), str) and d[key].strip()):
                errors.append(f"{where}.{key}: missing or empty")
        mx = d.get("max", None)
        if mx is not None and not (isinstance(mx, (int, float)) and mx > 0):
            errors.append(f"{where}.max: must be a positive number or null")
        pairs = d.get("pairs")
        if not isinstance(pairs, list):
            errors.append(f"{where}.pairs: missing or not a list")
            continue
        for i, p in enumerate(pairs):
            at = f"{where}.pairs[{i}]"
            if not isinstance(p, dict):
                errors.append(f"{at}: not an object")
                continue
            pos, neg = p.get("pos"), p.get("neg")
            if not (isinstance(pos, str) and pos.strip()):
                errors.append(f"{at}.pos: missing or empty")
            if not (isinstance(neg, str) and neg.strip()):
                errors.append(f"{at}.neg: missing or empty")
            if isinstance(pos, str) and isinstance(neg, str) and pos.strip() == neg.strip():
                errors.append(f"{at}: pos and neg are identical -- the difference would be the zero vector")
            # schema v2: each pair carries the user `prompt` both replies answer, so activations can be
            # read IN CHAT CONTEXT (template(user=prompt) + reply, pooled over the reply's tokens) --
            # matching the distribution steering is actually applied in. A missing prompt still loads
            # (bare-text derivation) but is warned: it's the honest-but-weaker mode.
            prompt = p.get("prompt")
            if prompt is not None and not (isinstance(prompt, str) and prompt.strip()):
                errors.append(f"{at}.prompt: present but empty -- omit it entirely for bare-text mode")
            elif prompt is None:
                warnings.append(f"{at}: no prompt -- will derive from bare text (weaker than chat-context)")
        if len(pairs) < MIN_RECOMMENDED_PAIRS:
            warnings.append(f"{where}: only {len(pairs)} pair(s); {MIN_RECOMMENDED_PAIRS}+ recommended "
                            "(the averaging is what buys robustness over the single-instruction recipe)")
    return errors, warnings


def dial_names(bank: dict) -> list[str]:
    return sorted(bank.get("dials", {}))


def pairs_for(bank: dict, name: str) -> list[tuple[str, str]]:
    """[(pos_text, neg_text), ...] for one dial -- the exact shape a CAA-style derivation consumes:
    for each pair, read the residual of pos and of neg at the steering layer, subtract, then average over
    pairs and unit-normalize."""
    d = bank.get("dials", {}).get(name) or {}
    return [(p["pos"], p["neg"]) for p in d.get("pairs", [])
            if isinstance(p, dict) and isinstance(p.get("pos"), str) and isinstance(p.get("neg"), str)]


def ready(bank: dict, min_pairs: int = MIN_RECOMMENDED_PAIRS) -> list[str]:
    """Dials with enough matched pairs to be worth deriving from this bank rather than the instruction
    recipe. The rest can (and should) fall back until someone adds pairs -- an honest partial adoption."""
    return [n for n in dial_names(bank) if len(pairs_for(bank, n)) >= min_pairs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--validate", action="store_true", help="structural + quality check; nonzero exit on error")
    ap.add_argument("--list", action="store_true", dest="do_list", help="dial roster + pair counts")
    args = ap.parse_args(argv)

    bank = load(args.path)
    if args.do_list or not args.validate:
        print(f"\ndial exemplar bank -- schema v{bank.get('schema_version')} -- {len(bank.get('dials', {}))} dials\n")
        print(f"  {'dial':12}{'poles':28}{'pairs':>6}  {'max':>5}")
        for n in dial_names(bank):
            d = bank["dials"][n]
            poles = " -> ".join(d.get("poles", ["?", "?"]))
            mx = d.get("max")
            flag = "" if len(pairs_for(bank, n)) >= MIN_RECOMMENDED_PAIRS else "   (add more)"
            print(f"  {n:12}{poles:28}{len(pairs_for(bank, n)):>6}  {str(mx if mx is not None else '-'):>5}{flag}")
        rdy = ready(bank)
        print(f"\n  ready for CAA derivation ({MIN_RECOMMENDED_PAIRS}+ pairs): {rdy or 'none yet'}")
        print(f"  falling back to the instruction recipe: {[n for n in dial_names(bank) if n not in rdy]}\n")

    if args.validate:
        errors, warnings = validate(bank)
        for w in warnings:
            print(f"  warn: {w}")
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\n  {len(errors)} error(s), {len(warnings)} warning(s)\n")
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
