"""Build and run a small attention-knockout scan with a reproducible manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from clozn_client import (
    AttentionKnockout,
    EngineClient,
    InterventionArm,
    InterventionManifest,
    ScoreRequest,
)


def build_manifest() -> InterventionManifest:
    # Token positions must come from the exact worker/tokenizer used for the run. These positions are
    # illustrative; inspect /harvest or a recorded trace before adapting the example to another prompt.
    request = ScoreRequest(
        prompt="Context: Paris is in France.\nAnswer:",
        continuation=" Paris",
        topk=3,
    )
    return InterventionManifest(
        name="capital-source-knockout",
        request=request,
        expected_health={"capabilities": {"attn_knockout": True}},
        arms=(
            InterventionArm(
                name="cut-context-span",
                attention_knockout=tuple(
                    AttentionKnockout(layer=layer, queries=(8,), keys=(2, 3, 4))
                    for layer in range(4)
                ),
                metadata={"role": "candidate", "source_span": [2, 5]},
            ),
            InterventionArm(
                name="matched-control",
                attention_knockout=tuple(
                    AttentionKnockout(layer=layer, queries=(8,), keys=(5, 6, 7))
                    for layer in range(4)
                ),
                metadata={"role": "control", "source_span": [5, 8]},
            ),
        ),
        metadata={
            "method": "teacher-forced attention-edge knockout",
            "limits": "Example positions are illustrative; no causal label is inferred automatically.",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:8091")
    parser.add_argument("--save-manifest", type=Path)
    args = parser.parse_args()

    manifest = build_manifest()
    if args.save_manifest is not None:
        manifest.save(args.save_manifest)
        print(f"saved {args.save_manifest}  sha256={manifest.sha256}")

    result = EngineClient(args.engine).run_manifest(manifest)
    print(result.to_json(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
