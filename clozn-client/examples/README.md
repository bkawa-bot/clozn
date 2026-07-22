# Worked examples

All examples keep the public gateway and private native engine explicit.

```bash
python examples/inspect_run.py CLOZN_RUN_ID --gateway http://127.0.0.1:8080
python examples/replay_manifest.py examples/capital-knockout.manifest.json \
  --engine http://127.0.0.1:8091
python examples/knockout_scan.py --engine http://127.0.0.1:8091 \
  --save-manifest /tmp/scan.json
```

The knockout examples do not infer a provenance verdict. They report measured log-probability
changes and preserve candidate/control labels supplied by the researcher. Token positions are only
valid for the exact prompt and tokenizer from which they were obtained.

- `patch_sweep.py` — harvest once and compare identity/amplification residual writes (requires the `arrays` extra).
