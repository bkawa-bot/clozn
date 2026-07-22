# Phase 4.2C — Native protocol advertisement

Roadmap item: Phase 4.2, versioned hook/capture and intervention contracts.

Completed:

- `clozn.engine_protocol.v1` health advertisement schema.
- Exact engine protocol version checking.
- Exact intervention-contract schema and SHA-256 checking.
- Explicit `verified`, `unadvertised`, and `incompatible` compatibility states.
- Fail-closed compatibility enforcement; endpoint presence is no longer treated as proof.
- No new capture or intervention operation.

Required engine health fragment:

```json
{
  "protocol": {
    "schema": "clozn.engine_protocol.v1",
    "version": "1.0",
    "intervention_contract": {
      "schema": "clozn.intervention_contract.v1",
      "sha256": "<exact client contract digest>"
    }
  }
}
```

Remaining:

- Implement this advertisement in the native worker itself.
- Qualify the residual seam on the selected reference model and artifact.
- Define compatibility rules for a future v2 protocol or contract.
- Bind verified protocol identity into gateway receipt exports.
