"""Unified export receipt bundle.

This module is intentionally model-free: it reshapes data that is already on the
stored run record plus the pure M1 explain object. Generating causal receipts
still belongs to receipts.py and requires an explicit substrate-backed endpoint.
"""
from __future__ import annotations

SCHEMA_VERSION = "receipt_bundle.v1"

REPRO_META_KEYS = (
    "model_id",
    "model_file",
    "quant",
    "mode",
    "sampler_mode",
    "sampling",
    "temperature",
    "top_p",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "max_tokens",
    "seed",
    "n_ctx",
    "device",
    "gpu_layers",
    "build_git_commit",
    "finish_reason_source",
    "finish_reason_fallback",
    "capture_tier",
)


def _dict(x) -> dict:
    return dict(x) if isinstance(x, dict) else {}


def _list(x) -> list:
    return list(x) if isinstance(x, list) else []


def _stored_receipts(run: dict) -> dict | None:
    rid = run.get("id")
    raw = run.get("receipts")
    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("run_id", rid)
        out.setdefault("receipts", [])
        out.setdefault("skipped", [])
        out.setdefault("redundant_pairs", [])
        return out
    if isinstance(raw, list):
        return {"run_id": rid, "receipts": list(raw), "skipped": [], "redundant_pairs": []}

    one = run.get("receipt")
    if isinstance(one, dict):
        return {"run_id": rid, "receipts": [dict(one)], "skipped": [], "redundant_pairs": []}
    return None


def _repro(run: dict) -> dict:
    meta = _dict(run.get("meta"))
    out = {k: meta.get(k) for k in REPRO_META_KEYS}
    out.update({
        "run_id": run.get("id"),
        "created_at": run.get("created_at"),
        "created_ts": run.get("created_ts"),
        "source": run.get("source"),
        "client": run.get("client"),
        "model": run.get("model"),
        "substrate": run.get("substrate"),
        "finish_reason": run.get("finish_reason"),
        "parent_run_id": run.get("parent_run_id"),
        "changes_applied": run.get("changes_applied"),
        "timing": _dict(run.get("timing")),
        "meta": meta,
    })
    return out


def _concepts(run: dict, trace: dict, explain: dict | None):
    concepts = trace.get("concepts")
    if concepts:
        return concepts
    concepts = run.get("concepts")
    if concepts:
        return concepts
    concepts = _dict(explain).get("concepts")
    if isinstance(concepts, dict) and concepts.get("available"):
        return concepts
    return None


def _tiny_tests(run: dict):
    tests = run.get("tiny_tests")
    return list(tests) if isinstance(tests, list) else None


def build(run: dict | None, explain: dict | None = None, receipts=None) -> dict:
    """Build the versioned export bundle from existing run/explain data."""
    run = run if isinstance(run, dict) else {}
    explain = explain if isinstance(explain, dict) else None
    trace = _dict(run.get("trace"))
    memory = _dict(run.get("memory"))
    receipt_obj = receipts if isinstance(receipts, dict) else None
    if receipt_obj is None:
        receipt_obj = _stored_receipts(run)
    workspace_readouts = _list(trace.get("workspace_readouts"))

    return {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "repro": _repro(run),
        "trace": trace,
        "memory": memory,
        "explain": explain,
        "receipts": receipt_obj,
        "workspace_readouts": workspace_readouts or None,
        "concepts": _concepts(run, trace, explain),
        "tiny_tests": _tiny_tests(run),
    }


def to_markdown(bundle: dict | None) -> str:
    """Render the readable export receipt from the same object returned as JSON."""
    bundle = bundle if isinstance(bundle, dict) else build(None)
    run = _dict(bundle.get("run"))
    repro = _dict(bundle.get("repro"))
    xr = _dict(bundle.get("explain"))
    mem = _dict(bundle.get("memory"))

    lines = [f"# Run {run.get('id') or '?'}"]
    head = " - ".join(str(x) for x in [
        run.get("created_at"),
        f"{run.get('source', '?')}/{run.get('client', '?')}",
        run.get("model"),
    ] if x)
    if head:
        lines.append(f"\n_{head}_")

    meta_bits = [f"{k}={repro[k]}" for k in REPRO_META_KEYS if repro.get(k) is not None]
    if meta_bits:
        lines.append("`" + " - ".join(meta_bits) + "`")

    finish_reason = repro.get("finish_reason")
    if finish_reason:
        suffix = " - WARNING: truncated (hit the token cap)" if finish_reason == "length" else ""
        lines.append(f"\n**stop:** {finish_reason}{suffix}")

    lines.append("\n## Conversation")
    msgs = run.get("messages") or []
    for msg in msgs:
        if isinstance(msg, dict):
            lines.append(f"\n**{msg.get('role', '?')}:** {str(msg.get('content', '')).strip()}")
    if run.get("response") and not (msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant"):
        lines.append(f"\n**assistant:** {str(run.get('response')).strip()}")

    assembled = run.get("assembled_messages")
    if mem.get("mode") == "prompt" and isinstance(assembled, list):
        lines.append("\n## Assembled prompt/messages")
        for msg in assembled:
            if isinstance(msg, dict):
                lines.append(f"\n**{msg.get('role', '?')}:** {str(msg.get('content', '')).strip()}")
        if mem.get("prompt_block"):
            lines.append("\n### Memory-injected section")
            lines.append(str(mem.get("prompt_block")).strip())
    elif mem.get("mode") == "internalized" and mem.get("has_prefix"):
        lines.append("\n## Assembled prompt/messages")
        lines.append("Memory injected as soft prefix; no literal prompt string.")

    cards = mem.get("cards_applied") or []
    if cards:
        lines.append("\n## Memory applied")
        rels = mem.get("relevance") or []
        for i, card in enumerate(cards):
            rel = rels[i] if i < len(rels) else None
            suffix = f"  _(relevance {rel:.2f})_" if isinstance(rel, (int, float)) else ""
            lines.append(f"- {card}{suffix}")
        bits = []
        if isinstance(mem.get("strength"), (int, float)):
            bits.append(f"strength {float(mem['strength']):.2f}")
        if isinstance(mem.get("gate"), (int, float)):
            bits.append(f"gate {float(mem['gate']):.2f}")
        if mem.get("mode"):
            bits.append(f"{mem['mode']} mode")
        if bits:
            lines.append("\n_" + " - ".join(bits) + "_")

    dials = _dict(_dict(run.get("behavior")).get("active_dials"))
    if dials:
        lines.append("\n## Behavior dials")
        for key, val in dials.items():
            lines.append(f"- {key}: {val}")

    confidence = _dict(xr.get("confidence"))
    if confidence.get("available"):
        lines.append("\n## Token trace")
        lines.append(f"{confidence.get('n_tokens', '?')} tokens - {confidence.get('summary', '')}")

    readouts = _list(bundle.get("workspace_readouts"))
    if readouts:
        latest = readouts[-1]
        provider = latest.get("provider") if isinstance(latest, dict) else None
        lines.append("\n## Workspace readouts")
        lines.append(f"{len(readouts)} readouts" + (f" - provider {provider}" if provider else ""))

    return "\n".join(lines) + "\n"
