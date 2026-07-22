"""Inspect one recorded Clozn run through the public gateway."""
from __future__ import annotations

import argparse
import json

from clozn_client import CloznClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--gateway", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    client = CloznClient(args.gateway)
    run = client.run(args.run_id)
    timeline = client.timeline(run.id)
    receipt = client.export_receipt(run.id)

    print(json.dumps({
        "run": run.raw,
        "timeline_event_kinds": [event.get("kind") for event in timeline.events],
        "receipt_schema": receipt.raw.get("schema"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
