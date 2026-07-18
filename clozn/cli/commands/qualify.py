"""commands.qualify -- `clozn qualify-whitebox <gguf>`: an honest capability matrix for a GGUF.

This is the "will white-box actually work on THIS model" question, answered from three real, existing
sources -- never a guess, never a hardcoded per-model claim baked into this file:

  1. `clozn.artifacts.contracts.gguf_identity` -- the exact, model-agnostic identity (architecture,
     hidden_size, layer_count, vocab_size, tokenizer/chat-template digests, the whole-file sha256) read
     straight off the GGUF header (see that module's docstring: contracts are tied to an exact tokenizer
     and residual-space contract, never "two filenames look similar").
  2. `docs/qualification/wave1.json` -- the qualification LEDGER: which model families have actually been
     through the live smoke/qualification process, and what each layer's status is (core / white_box /
     dials / jlens). This is a project-wide record, not something on this machine.
  3. `clozn/artifacts/<CLOZN_HOME>/artifacts/{jlens,sae}/**/manifest.json` -- the artifact contracts
     actually INSTALLED locally, validated via `contracts.find_compatible_artifact` -- the exact same
     lookup `clozn.cli.engine_process.spawn_engine` performs at boot time for `--jlens` (mirrored here,
     not reinvented), so this command's verdict for j-lens matches what a real `clozn serve` would do.

Two families of feature, gated differently, and the whole point of this module is to never blur them:

  - receipts / explain / rewrite ride the engine's GENERIC teacher-forced `/score` and templated `/chat`
    -- no per-model artifact, no calibration table, works on any GGUF the C++ engine can load. These are
    reported qualified unconditionally, with a caveat when this exact family has no wave1 smoke record
    (untested, not "individually broken").
  - steering / j-lens / SAE need per-model data: a calibrated steer TAP LAYER
    (`clozn.server.substrates._ENGINE_MODELS`), a fitted+qualified J-lens artifact, or a contract-qualified
    SAE readout (none exist yet -- see contracts.py's own "future SAE/readout artifacts" language). An
    unrecognized architecture, or a recognized one with no calibration recorded, is reported honestly as
    NOT qualified, with the specific reason -- never silently assumed to work.

Model-free throughout (tests/test_qualify_cli.py): everything above is header reads, JSON reads, and
dict lookups -- no engine, no GPU, no Torch.
"""
from __future__ import annotations

import json
import os

from clozn.artifacts import contracts
from clozn.cli import formatting as fmt

_WAVE1_RELATIVE = ("docs", "qualification", "wave1.json")

# name -> what actually backs the feature, used verbatim in every reason string so the "why" always names
# the real mechanism, not a vague label.
_CORE_FEATURES = {
    "receipts": "leave-one-out causal ablation over the engine's generic teacher-forced /score",
    "explain": "hesitation + active-influence narration built on the same generic /score + trace machinery",
    "rewrite": "constrained AR chat rewrite (Route D) -- a plain chat() call templated for this GGUF",
}


def _feature(name: str, qualified: bool, reason: str, evidence: dict | None = None) -> dict:
    d = {"feature": name, "qualified": bool(qualified), "reason": reason}
    if evidence:
        d["evidence"] = evidence
    return d


def _status_passed(value) -> bool:
    v = str(value or "").lower()
    return v.startswith("passed") or v.startswith("qualified")


# ------------------------------------------------------------------------------------------- wave1 ledger

def _default_wave1_path() -> str:
    from clozn.cli.engine_process import REPO
    return os.path.join(REPO, *_WAVE1_RELATIVE)


def load_wave1(path: str | None = None) -> list:
    """The `models` list out of docs/qualification/wave1.json, or [] on any read/parse problem -- a
    missing or corrupt ledger just means no wave1 match is possible; every feature that would have used
    it honestly reports "not in the wave1 qualification ledger" instead of crashing this command."""
    path = path or _default_wave1_path()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, list) else []


def find_wave1_match(identity: dict, *, checkpoint: str | None = None, wave1_models: list | None = None):
    """Best wave1.json entry for this GGUF identity, most to least exact:

        1) exact GGUF sha256 -- the precise qualified checkpoint file
        2) same architecture + tokenizer_sha256 -- same model family, a different file/quant
        3) an explicit --checkpoint source_id -- the weakest signal (a user assertion, not derived from
           the file itself); refused (see below) if it disagrees with the GGUF's own architecture

    Returns (entry_or_None, match_kind), match_kind in {"sha256", "family", "checkpoint",
    "checkpoint_architecture_mismatch", None}. The mismatch case returns the entry too (so callers can
    report exactly what was asserted) but callers must NOT treat it as a positive match."""
    models = wave1_models if wave1_models is not None else load_wave1()
    sha = str(identity.get("sha256") or "").lower()
    arch = identity.get("architecture")
    tok = identity.get("tokenizer_sha256")

    for m in models:
        g = m.get("gguf") or {}
        if sha and str(g.get("sha256", "")).lower() == sha:
            return m, "sha256"
    for m in models:
        g = m.get("gguf") or {}
        if arch and tok and g.get("architecture") == arch and g.get("tokenizer_sha256") == tok:
            return m, "family"
    if checkpoint:
        needle = checkpoint.strip().lower()
        for m in models:
            if str(m.get("source_id", "")).lower() == needle:
                g = m.get("gguf") or {}
                if arch and g.get("architecture") and g.get("architecture") != arch:
                    return m, "checkpoint_architecture_mismatch"
                return m, "checkpoint"
    return None, None


# ---------------------------------------------------------------------------------- per-feature qualifiers

def _core_feature_qualification(name: str, wave1_entry: dict | None) -> dict:
    basis = _CORE_FEATURES[name]
    core_status = ((wave1_entry or {}).get("status") or {}).get("core")
    white_box_status = ((wave1_entry or {}).get("status") or {}).get("white_box")
    if wave1_entry is not None and _status_passed(core_status):
        reason = (f"{basis}; engine-generic (works on any GGUF the engine loads), and this family passed "
                 f"the wave1 core smoke ladder ({core_status!r})")
    else:
        reason = (f"{basis}; engine-generic (works on any GGUF the engine loads) -- this exact model has "
                 "not been run through the wave1 smoke ladder, so treat it as untested, not individually "
                 "broken")
    return _feature(name, True, reason,
                    evidence={"wave1_core_status": core_status, "wave1_white_box_status": white_box_status})


def steer_qualification(model_family, steer_layer, wave1_entry: dict | None) -> dict:
    """Steering needs a calibrated TAP LAYER for this model's family (clozn.server.substrates._ENGINE_MODELS
    -- the exact table EngineSubstrate.__init__ consults at boot) -- and, once that exists, the wave1
    ledger's own `dials` status decides whether the dial DIRECTIONS riding that tap are themselves
    qualified (a family can have a known tap layer but still carry only legacy/global dial vectors that
    need per-model recalibration -- see wave1.json's qwen2.5 entry for exactly this case)."""
    dials_status = ((wave1_entry or {}).get("status") or {}).get("dials")
    if model_family is None:
        return _feature("steering", False,
                        "unrecognized model family -- no filename match in the engine's steer-tap registry "
                        "(clozn.server.substrates._ENGINE_MODELS), so tone dials have no known injection "
                        "point for this architecture",
                        evidence={"model_family": None, "steer_layer": None})
    if steer_layer is None:
        return _feature("steering", False,
                        f"model family {model_family!r} is recognized but carries no calibrated steer tap "
                        "layer yet (steer_layer is None in _ENGINE_MODELS) -- dials would fall back to the "
                        "engine's own generic mid-depth guess, not a per-model calibrated tap",
                        evidence={"model_family": model_family, "steer_layer": None})
    if dials_status is not None and not _status_passed(dials_status):
        return _feature("steering", False,
                        f"steer tap layer {steer_layer} is calibrated for {model_family!r}, but the wave1 "
                        f"ledger flags dial status as {dials_status!r} -- treat active dials as unverified "
                        "for this exact checkpoint",
                        evidence={"model_family": model_family, "steer_layer": steer_layer,
                                  "wave1_dials_status": dials_status})
    return _feature("steering", True,
                    f"calibrated steer tap layer {steer_layer} for family {model_family!r}"
                    + (f" (wave1 dials status: {dials_status!r})" if dials_status else ""),
                    evidence={"model_family": model_family, "steer_layer": steer_layer,
                              "wave1_dials_status": dials_status})


def jlens_qualification(identity: dict, artifact_root: str, wave1_entry: dict | None) -> dict:
    """Mirrors clozn.cli.engine_process.spawn_engine's own `find_compatible_artifact("jlens", ...)` lookup
    exactly, so this command's verdict matches what a real `clozn serve` on this machine would actually do
    -- not a re-derived guess. A locally installed, contract-valid artifact is the ONLY thing that makes
    j-lens qualified; the wave1 ledger alone (a project-wide record of what has been qualified in
    principle) never is, since it says nothing about what's on THIS machine."""
    try:
        local_dir = contracts.find_compatible_artifact("jlens", identity, artifact_root)
    except contracts.ArtifactContractError as error:
        return _feature("jlens", False,
                        f"a local J-lens artifact under {artifact_root} claims this GGUF but failed "
                        f"contract validation: {error}")
    if local_dir:
        return _feature("jlens", True, f"qualified local artifact found: {local_dir}",
                        evidence={"artifact_dir": local_dir})
    wave1_status = ((wave1_entry or {}).get("status") or {}).get("jlens")
    if wave1_entry is not None:
        return _feature("jlens", False,
                        f"no local artifact installed under {artifact_root} -- the wave1 ledger records "
                        f"this family's jlens status as {wave1_status!r}, but nothing is qualified on this "
                        "machine until an artifact is fetched/fitted and its manifest validates against "
                        "this exact GGUF sha256",
                        evidence={"wave1_jlens_status": wave1_status})
    return _feature("jlens", False,
                    "unrecognized architecture/tokenizer -- no J-lens has ever been fitted or qualified "
                    "for this model (not in the wave1 ledger, and no local artifact claims it)")


def sae_qualification(identity: dict, artifact_root: str) -> dict:
    """SAE readouts have no artifact-contract type shipped yet -- `clozn serve --sae <dir>` loads a
    user-supplied directory validated only by a dimension check inside the engine at load time, not by a
    manifest (see clozn/artifacts/contracts.py's own "future SAE/readout artifacts" framing). The
    find_compatible_artifact("sae", ...) lookup below is forward-compatible plumbing for when that
    contract ships; today it will essentially always find nothing, and that absence IS the honest answer."""
    try:
        local_dir = contracts.find_compatible_artifact("sae", identity, artifact_root)
    except contracts.ArtifactContractError as error:
        return _feature("sae", False,
                        f"a local SAE artifact under {artifact_root} claims this GGUF but failed contract "
                        f"validation: {error}")
    if local_dir:
        return _feature("sae", True, f"qualified local artifact found: {local_dir}",
                        evidence={"artifact_dir": local_dir})
    return _feature("sae", False,
                    "no SAE readout artifact is contract-qualified for any model yet -- `--sae <dir>` is "
                    "loaded and dimension-checked by the engine at boot, not validated by an artifact "
                    "manifest (see clozn/artifacts/contracts.py)")


# ------------------------------------------------------------------------------------------- the matrix

def build_capability_matrix(identity: dict, *, checkpoint: str | None = None,
                            artifact_root: str | None = None, wave1_models: list | None = None) -> dict:
    """The full report: model identity + wave1 ledger match + per-feature qualification + a qualified/
    not-qualified summary. Pure (no I/O beyond the wave1/artifact reads already threaded through as
    arguments or defaulted below) -- never raises; a bad --checkpoint or a missing ledger degrades to an
    honest "no match" rather than an error."""
    # Re-exported off clozn.server.app (never clozn.server.substrates directly): substrates.py itself does
    # `from clozn.server import app as ctx` at module scope, so importing substrates before app has fully
    # loaded is a circular-import trap (see tests/test_engine_model_registry.py's own import for the same
    # sanctioned pattern).
    from clozn.server import app as engine_app

    wave1_entry, match_kind = find_wave1_match(identity, checkpoint=checkpoint, wave1_models=wave1_models)
    checkpoint_warning = None
    if match_kind == "checkpoint_architecture_mismatch":
        checkpoint_warning = (
            f"--checkpoint {checkpoint!r} names a wave1 entry whose architecture "
            f"({(wave1_entry.get('gguf') or {}).get('architecture')!r}) does not match this GGUF's own "
            f"({identity.get('architecture')!r}) -- ignoring that wave1 entry rather than trusting the "
            "assertion")
        wave1_entry, match_kind = None, None

    if artifact_root is None:
        artifact_root = _default_artifact_root()

    model_family, engine_info = engine_app._engine_model_info(identity.get("filename") or "")
    steer_layer = engine_info.get("steer_layer")

    features = [
        _core_feature_qualification("receipts", wave1_entry),
        _core_feature_qualification("explain", wave1_entry),
        _core_feature_qualification("rewrite", wave1_entry),
        steer_qualification(model_family, steer_layer, wave1_entry),
        jlens_qualification(identity, artifact_root, wave1_entry),
        sae_qualification(identity, artifact_root),
    ]
    qualified = [f["feature"] for f in features if f["qualified"]]
    not_qualified = [f["feature"] for f in features if not f["qualified"]]

    report = {
        "identity": dict(identity),
        "checkpoint": checkpoint,
        "checkpoint_warning": checkpoint_warning,
        "model_family": model_family,
        "wave1_match": None,
        "features": features,
        "summary": {"qualified": qualified, "not_qualified": not_qualified},
    }
    if wave1_entry is not None:
        report["wave1_match"] = {
            "source_id": wave1_entry.get("source_id"),
            "family": wave1_entry.get("family"),
            "match_kind": match_kind,
            "status": wave1_entry.get("status"),
        }
    return report


def _default_artifact_root() -> str:
    """The SAME artifact root clozn.cli.engine_process.spawn_engine resolves at boot -- reused verbatim
    (not reproduced independently) so this command's j-lens/SAE verdicts never drift from what a real
    `clozn serve` on this machine would do."""
    from clozn.cli import main as ctx
    return os.environ.get("CLOZN_ARTIFACTS_DIR") or os.path.join(ctx.HOME, "artifacts")


# ------------------------------------------------------------------------------------------------- render

def format_report(report: dict) -> str:
    """Pure JSON(build_capability_matrix result) -> text render, no I/O -- testable on a canned dict
    exactly like commands.test.format_test_report / commands.quant_check.format_ladder."""
    ident = report.get("identity") or {}
    lines = [f"{fmt.BOLD}qualify-whitebox{fmt.RST}: {ident.get('filename', '?')}"]
    for label, key in (("architecture", "architecture"), ("hidden_size", "hidden_size"),
                       ("layer_count", "layer_count"), ("vocab_size", "vocab_size"),
                       ("quantization", "quantization"), ("sha256", "sha256"),
                       ("tokenizer_sha256", "tokenizer_sha256")):
        lines.append(f"  {label:<16} {ident.get(key)}")

    wm = report.get("wave1_match")
    lines.append("")
    if wm:
        lines.append(f"wave1 ledger match ({wm.get('match_kind')}): {wm.get('source_id')} "
                     f"[{wm.get('family')}]")
        for k, v in (wm.get("status") or {}).items():
            lines.append(f"  {k:<10} {v}")
    else:
        lines.append("wave1 ledger: no match -- this GGUF/family has not been through qualification")
    if report.get("checkpoint_warning"):
        lines.append("")
        lines.append(f"{fmt.DIM}note: {report['checkpoint_warning']}{fmt.RST}")

    lines.append("")
    lines.append(f"{fmt.BOLD}capability matrix{fmt.RST}")
    features = report.get("features") or []
    width = max((len(f["feature"]) for f in features), default=8)
    for f in features:
        mark = f"{fmt.BOLD}QUALIFIED{fmt.RST}" if f["qualified"] else f"{fmt.DIM}not qualified{fmt.RST}"
        box = "x" if f["qualified"] else " "
        lines.append(f"  [{box}] {f['feature']:<{width}}  {mark}")
        lines.append(f"        {fmt.DIM}{f['reason']}{fmt.RST}")

    summary = report.get("summary") or {}
    q, nq = summary.get("qualified") or [], summary.get("not_qualified") or []
    lines.append("")
    lines.append(f"{len(q)}/{len(features)} features qualified: {', '.join(q) or '(none)'}")
    if nq:
        lines.append(f"not qualified: {', '.join(nq)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------- CLI

def add_subparser(sub):
    """Registers `clozn qualify-whitebox` on an argparse subparsers object (own function so its wiring is
    testable without dispatching or touching clozn/cli/main.py -- mirrors commands.quant_check /
    commands.eval's add_subparser)."""
    pq = sub.add_parser(
        "qualify-whitebox",
        help="report which white-box features (receipts, explain, rewrite, steering, j-lens, SAE) are "
             "qualified for a GGUF, from the artifact contracts + wave1 qualification ledger "
             "(model-free -- reads only the GGUF header, no engine, no GPU)")
    pq.add_argument("gguf", help="a GGUF path, known short name, or fuzzy filename fragment (resolved the "
                    "same way as `clozn run`'s model arg)")
    pq.add_argument("--json", action="store_true", help="print the raw capability matrix as JSON")
    pq.add_argument("--checkpoint", default=None, metavar="ORG/MODEL",
                    help="cross-reference a HuggingFace checkpoint id (e.g. Qwen/Qwen2.5-7B-Instruct) "
                         "against the wave1 qualification ledger, when the GGUF's own sha256/tokenizer "
                         "doesn't already match an entry")
    pq.set_defaults(fn=cmd_qualify)
    return pq


def cmd_qualify(args) -> int:
    """`clozn qualify-whitebox <gguf> [--checkpoint ORG/MODEL] [--json]` -- model-free: reads the GGUF
    header (contracts.gguf_identity), matches it against the wave1 ledger and any locally installed
    j-lens/SAE artifacts, and reports the honest capability matrix (or --json for the raw dict). Always
    returns 0 -- an unqualified feature is a normal, informative answer, not a command failure; only a
    bad/unreadable GGUF path raises CloznError (main() turns that into a clean one-line exit 1)."""
    from clozn.cli.commands.models import resolve_model

    path = resolve_model(args.gguf)
    identity = contracts.gguf_identity(path)
    report = build_capability_matrix(identity, checkpoint=getattr(args, "checkpoint", None))

    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report))
    return 0
