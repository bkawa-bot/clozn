"""Qwen/Hugging Face substrate helpers."""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class RecordingLogitsProcessor(LogitsProcessor):
    """Observe-only: record the top-k token probabilities per decoding step."""

    def __init__(self, topk: int = 6):
        self.topk = int(topk)
        self.records: list[dict] = []

    def __call__(self, input_ids, scores):
        try:
            row = scores[0].detach().float()
            probs = torch.softmax(row, dim=-1)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-45))).sum().item())
            k = min(self.topk, probs.shape[-1])
            top = torch.topk(probs, k)
            self.records.append({
                "ids": [int(i) for i in top.indices.tolist()],
                "probs": [float(p) for p in top.values.tolist()],
                "entropy": entropy,
            })
        except Exception:
            self.records.append({"ids": [], "probs": []})
        return scores


def steps_from_records(records: list[dict], gen_ids: list[int], tok) -> list[dict]:
    """Align recorded top-k rows to the tokens actually emitted."""
    steps: list[dict] = []
    n = min(len(records), len(gen_ids))
    for i in range(n):
        try:
            rec = records[i] or {}
            ids = rec.get("ids", []) or []
            probs = rec.get("probs", []) or []
            tid = int(gen_ids[i])
            prob_by_id = {int(a): float(b) for a, b in zip(ids, probs)}
            conf = float(prob_by_id.get(tid, 0.0))
            alts = [{"token_id": int(a), "piece": tok.decode([a]), "prob": round(float(b), 4)}
                    for a, b in zip(ids, probs) if int(a) != tid][:3]
            step = {"index": i, "token_id": tid, "piece": tok.decode([tid]),
                    "conf": round(conf, 4), "alts": alts}
            if rec.get("entropy") is not None:
                step["entropy"] = float(rec["entropy"])
            steps.append(step)
        except Exception:
            continue
    return steps


def finish_reason_from_generated_ids(ids, eos_token_id, max_new) -> str | None:
    """Infer the HF generate stop cause from generated ids when observable."""
    try:
        gen_ids = [int(t) for t in (ids or [])]
    except Exception:
        gen_ids = []
    try:
        eos = int(eos_token_id) if eos_token_id is not None else None
    except Exception:
        eos = None
    if eos is not None and gen_ids and gen_ids[-1] == eos:
        return "stop"
    try:
        cap = int(max_new)
    except Exception:
        cap = 0
    if cap > 0 and len(gen_ids) >= cap:
        return "length"
    return None


def resolve_model_path(name: str) -> str:
    local = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
    return local if os.path.isfile(os.path.join(local, "config.json")) else name


def load_model_and_tokenizer(model_name: str, four_bit: bool = True):
    """Load the Qwen-compatible HF model/tokenizer pair for local substrate use."""
    path = resolve_model_path(model_name)
    print(f"loading {model_name} ({'4-bit nf4' if four_bit else 'bf16'}) on {DEV} from {path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if four_bit and DEV == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV)
    return tok, model
