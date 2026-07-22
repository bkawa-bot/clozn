"""Replay a saved intervention manifest against an explicitly selected native engine."""
from __future__ import annotations

import argparse

from clozn_client import EngineClient, InterventionManifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--engine", default="http://127.0.0.1:8091")
    args = parser.parse_args()

    manifest = InterventionManifest.load(args.manifest)
    result = EngineClient(args.engine).replay_manifest(manifest)
    print(result.to_json(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
