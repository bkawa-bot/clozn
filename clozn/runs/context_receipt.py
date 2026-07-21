"""Context-delivery receipts and honest token-cutoff warnings.

The journal already retains three different views of an input turn:

* ``messages`` are what the compatibility/gateway layer delivered to generation;
* ``assembled_messages`` are the post-memory messages handed to the model template;
* ``final_prompt`` is the exact rendered string sent to the worker.

This module gives those views stable ``delivered`` / ``survived`` labels.  It never
infers token survival from text and never calls ``finish_reason == length`` prompt
truncation: the worker rejects an overlong prompt.  ``length`` only proves that the
generated reply stopped at its output/context budget.
"""
from __future__ import annotations

from copy import deepcopy


OUTPUT_TRUNCATED = "output_truncated"


def cutoff_warning(finish_reason, meta=None) -> dict | None:
    """Return one structured warning for a proven output cutoff, else ``None``."""
    if finish_reason != "length":
        return None
    meta = meta if isinstance(meta, dict) else {}
    warning = {
        "code": OUTPUT_TRUNCATED,
        "severity": "warning",
        "message": ("generation stopped at the output/context token budget; "
                    "the reply may be incomplete"),
    }
    maximum = meta.get("max_tokens")
    if isinstance(maximum, int) and maximum > 0:
        warning["requested_max_tokens"] = maximum
    return warning


def warnings_for(finish_reason, meta=None) -> list[dict]:
    warning = cutoff_warning(finish_reason, meta)
    return [warning] if warning else []


def build_context_receipt(*, messages=None, assembled_messages=None, final_prompt=None,
                          finish_reason=None, meta=None, trace=None) -> dict:
    """Build a no-inference receipt from the evidence captured for one run."""
    meta = meta if isinstance(meta, dict) else {}
    trace = trace if isinstance(trace, dict) else {}
    delivered = {
        "label": "delivered",
        "meaning": "messages accepted by the gateway and handed to prompt assembly",
        "messages": deepcopy(messages) if isinstance(messages, list) else [],
    }
    survived = {
        "label": "survived",
        "meaning": "post-assembly input retained as evidence of what reached generation",
        "assembled_messages": (deepcopy(assembled_messages)
                               if isinstance(assembled_messages, list) else None),
        "final_prompt": final_prompt if isinstance(final_prompt, str) else None,
    }
    prompt_tokens = meta.get("prompt_tokens")
    n_ctx = meta.get("n_ctx")
    maximum = meta.get("max_tokens")
    generated = len(trace.get("tokens") or []) if isinstance(trace.get("tokens"), list) else None
    limits = {
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "context_window_tokens": n_ctx if isinstance(n_ctx, int) else None,
        "requested_max_tokens": maximum if isinstance(maximum, int) else None,
        "generated_tokens": generated,
    }
    warning = cutoff_warning(finish_reason, meta)
    return {
        "schema": "clozn.context_receipt.v1",
        "delivered": delivered,
        "survived": survived,
        "input_truncated": False,
        "input_policy": "overlong prompts are rejected, not silently truncated",
        "output_cut_off": warning is not None,
        "limits": limits,
        "warnings": [warning] if warning else [],
    }
