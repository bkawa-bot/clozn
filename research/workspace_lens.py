"""Workspace Lens readout helpers.

The preferred path adapts Clozn's existing live concept readouts into
`workspace_readout` events. Mock readouts remain only for sample traces and
offline fallback demos; they are not auto-attached to real runs.
"""

from __future__ import annotations

import hashlib

LABELS = (
    "code_error",
    "uncertainty",
    "memory_reference",
    "instruction_following",
    "hallucination_risk",
)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _jitter(run_id: str, token_index: int, label: str) -> float:
    key = f"{run_id}:{token_index}:{label}".encode("utf-8", "ignore")
    return int(hashlib.sha1(key).hexdigest()[:6], 16) / 0xFFFFFF


def _trace_tokens(trace: dict) -> list[str]:
    return [str(t) for t in ((trace or {}).get("tokens") or [])]


def _trace_confidence(trace: dict) -> list[float]:
    out = []
    for c in ((trace or {}).get("confidence") or []):
        try:
            out.append(float(c))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _concept_items(concepts: dict, limit: int = 5) -> list[dict]:
    """Normalize BrainReadout concept payloads to [{label, score}]."""
    rows = []
    for c in (concepts or {}).get("considered") or []:
        if not isinstance(c, dict):
            continue
        label = str(c.get("label") or c.get("name") or "").strip()
        if not label:
            continue
        try:
            score = float(c.get("rel", c.get("score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        rows.append({"label": label, "score": _clamp(score / 1.5 if score > 1.0 else score)})
    if not rows:
        for c in (concepts or {}).get("concepts") or []:
            label = c.get("name") if isinstance(c, dict) else c
            label = str(label or "").strip()
            if label:
                rows.append({"label": label, "score": 0.5})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


def _token_scores(run_id: str, token_index: int, token_text: str, confidence: float) -> dict[str, float]:
    raw = (token_text or "").strip()
    text = raw.lower()
    fog = 1.0 - _clamp(confidence)
    scores = {
        "code_error": 0.10,
        "uncertainty": 0.18 + 0.68 * fog,
        "memory_reference": 0.12,
        "instruction_following": 0.16,
        "hallucination_risk": 0.16 + 0.35 * fog,
    }
    if any(w in text for w in ("error", "exception", "traceback", "bug", "fail", "undefined")):
        scores["code_error"] += 0.62
    if any(w in text for w in ("maybe", "perhaps", "likely", "roughly", "approx")):
        scores["uncertainty"] += 0.20
    if any(w in text for w in ("remember", "memory", "user", "preference", "profile")):
        scores["memory_reference"] += 0.58
    if any(w in text for w in ("must", "should", "please", "instruction", "follow", "do")):
        scores["instruction_following"] += 0.50
    if raw and (raw[:1].isupper() or any(w in text for w in ("citation", "source", "study", "fact"))):
        scores["hallucination_risk"] += 0.22
    for label in LABELS:
        scores[label] = _clamp(scores[label] + 0.08 * (_jitter(run_id, token_index, label) - 0.5))
    return scores


def mock_readouts_for_trace(run_id: str, trace: dict, *, layer: int = 12) -> list[dict]:
    """Return sample `workspace_readout` payloads for fixture/offline demo traces."""
    tokens = _trace_tokens(trace)
    if not tokens:
        return []
    confidence = _trace_confidence(trace)
    out = []
    for i, token in enumerate(tokens):
        conf = confidence[i] if i < len(confidence) else 1.0
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        scores = _token_scores(run_id, i, str(token), conf)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        entropy = _clamp((1.0 - _clamp(conf)) * 0.78 + 0.22 * (1.0 - top[0][1]))
        out.append({
            "t": i,
            "type": "workspace_readout",
            "run_id": run_id,
            "token_index": i,
            "token_text": str(token),
            "layer": layer,
            "position": i,
            "top_readouts": [{"label": label, "score": round(score, 4)} for label, score in top],
            "entropy": round(entropy, 4),
            "provider": "mock",
        })
    return out


def readouts_from_concepts(run_id: str, trace: dict, concepts: dict, *,
                           provider: str, layer: int | None = None) -> list[dict]:
    """Project an existing concept/SAE readout onto a token trace.

    BrainReadout produces run-level concept activations (`considered`) rather
    than a per-token decomposition. Until the engine emits per-token concept
    rows directly, we attach the same measured top concepts to each generated
    token and use the token confidence as the per-token fogginess signal.
    """
    tokens = _trace_tokens(trace)
    top = _concept_items(concepts)
    if not tokens or not top:
        return []
    confidence = _trace_confidence(trace)
    layer_val = layer if layer is not None else (concepts or {}).get("layer")
    try:
        layer_val = int(layer_val)
    except (TypeError, ValueError):
        layer_val = -1
    out = []
    for i, token in enumerate(tokens):
        conf = confidence[i] if i < len(confidence) else 1.0
        entropy = _clamp(1.0 - _clamp(conf))
        out.append({
            "t": i,
            "type": "workspace_readout",
            "run_id": run_id,
            "token_index": i,
            "token_text": token,
            "layer": layer_val,
            "position": i,
            "top_readouts": [{"label": r["label"], "score": round(r["score"], 4)} for r in top],
            "entropy": round(entropy, 4),
            "provider": provider,
        })
    return out
