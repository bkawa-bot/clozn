"""Command-line validation and replay for Clozn experiment manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from ._transport import CloznClientError, CloznProtocolError
from .batch import run_manifest_batch
from .compare import compare_batch_runs
from .engine import EngineClient
from .manifests import InterventionManifest
from .models import require_object
from .patch_manifests import PatchSweepManifest
from .provenance import capture_reproducibility
from .reporting import write_ci_reports


def _load(path: str):
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CloznProtocolError(f"invalid manifest JSON: {exc}") from None
    obj = require_object(raw, "manifest")
    schema = obj.get("schema")
    if schema == InterventionManifest.SCHEMA:
        return InterventionManifest.from_json(obj)
    if schema == PatchSweepManifest.SCHEMA:
        return PatchSweepManifest.from_json(obj)
    raise CloznProtocolError(f"unsupported manifest schema: {schema!r}")


def _write_or_print(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m clozn_client")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate and identify a manifest")
    validate.add_argument("manifest")

    replay = sub.add_parser("replay", help="replay a scoring or patch manifest")
    replay.add_argument("manifest")
    replay.add_argument("--engine-url", default="http://127.0.0.1:8091")
    replay.add_argument("--timeout", type=float, default=900.0)
    replay.add_argument("--output", "-o")

    batch = sub.add_parser("batch", help="replay multiple manifests into an indexed result directory")
    batch.add_argument("paths", nargs="+")
    batch.add_argument("--engine-url", default="http://127.0.0.1:8091")
    batch.add_argument("--timeout", type=float, default=900.0)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--index")
    batch.add_argument("--fail-fast", action="store_true")

    provenance = sub.add_parser("provenance", help="capture engine and environment provenance")
    provenance.add_argument("manifests", nargs="+")
    provenance.add_argument("--engine-url", default="http://127.0.0.1:8091")
    provenance.add_argument("--timeout", type=float, default=900.0)
    provenance.add_argument("--output", "-o")
    provenance.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")

    compare = sub.add_parser("compare", help="compare two batch indexes and gate regressions")
    compare.add_argument("baseline_index")
    compare.add_argument("candidate_index")
    compare.add_argument("--max-metric-delta", type=float, default=0.0)
    compare.add_argument("--output", "-o")
    compare.add_argument("--junit", help="write a JUnit XML report")
    compare.add_argument("--github-summary", help="write a GitHub step-summary Markdown report")
    compare.add_argument("--github-annotations", help="write GitHub Actions error annotations")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "provenance":
            manifests = [_load(path) for path in args.manifests]
            metadata: dict[str, str] = {}
            for item in args.metadata:
                if "=" not in item:
                    raise ValueError("--metadata entries must use KEY=VALUE")
                key, value = item.split("=", 1)
                if not key:
                    raise ValueError("--metadata keys must be non-empty")
                metadata[key] = value
            record = capture_reproducibility(
                EngineClient(args.engine_url, timeout=args.timeout),
                (manifest.sha256 for manifest in manifests),
                metadata=metadata,
            )
            _write_or_print(record.to_json(), args.output)
            return 0

        if args.command == "compare":
            comparison = compare_batch_runs(
                args.baseline_index, args.candidate_index,
                max_metric_delta=args.max_metric_delta,
            )
            _write_or_print(comparison.to_json(), args.output)
            write_ci_reports(
                comparison,
                junit=args.junit,
                github_summary=args.github_summary,
                github_annotations=args.github_annotations,
            )
            return 1 if comparison.regressions else 0

        if args.command == "batch":
            paths: list[Path] = []
            for raw in args.paths:
                candidate = Path(raw)
                if candidate.is_dir():
                    paths.extend(sorted(candidate.rglob("*.manifest.json")))
                else:
                    paths.append(candidate)
            unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
            manifests = [(str(path), _load(str(path))) for path in unique_paths]
            engine = EngineClient(args.engine_url, timeout=args.timeout)
            batch_result = run_manifest_batch(
                engine, manifests, args.output_dir, continue_on_error=not args.fail_fast
            )
            index_path = args.index or str(Path(args.output_dir) / "index.json")
            batch_result.write(index_path)
            _write_or_print(batch_result.to_json(), None)
            return 1 if batch_result.failed else 0

        manifest = _load(args.manifest)
        if args.command == "validate":
            payload = {
                "schema": manifest.SCHEMA,
                "name": manifest.name,
                "sha256": manifest.sha256,
                "valid": True,
            }
            _write_or_print(json.dumps(payload, sort_keys=True, indent=2) + "\n", None)
            return 0

        engine = EngineClient(args.engine_url, timeout=args.timeout)
        if isinstance(manifest, InterventionManifest):
            result = engine.run_manifest(manifest)
            payload = result.to_json_object()
            payload["manifest_name"] = manifest.name
            text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        else:
            text = engine.run_patch_manifest(manifest).to_json()
        _write_or_print(text, args.output)
        return 0
    except (OSError, ValueError, CloznClientError) as exc:
        print(f"clozn-client: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
