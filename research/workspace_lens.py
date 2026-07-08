"""Mock Workspace Lens provider.

This is intentionally a tiny, deterministic placeholder. It gives the Studio a
visible latent-workspace readout without claiming real interpretability. Real
providers can later implement the same event payload with logit lens, Jacobian
Lens, SAE probes, or linear probes.
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
    """Return `workspace_readout` event payloads for a normalized trace dict."""
    tokens = list((trace or {}).get("tokens") or [])
    if not tokens:
        return []
    confidence = list((trace or {}).get("confidence") or [])
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
