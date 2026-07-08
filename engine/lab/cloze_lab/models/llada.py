"""LLaDA 8B adapter — second real family behind the seam (DESIGN §4.4 target #2).

torch/transformers imports are allowed here and only here (DESIGN invariant 1).
LLaDA exercises the ModelAdapter abstraction precisely where it differs from Dream:

* **In-place prediction head.** LLaDA is a true masked diffusion model, so the
  distribution for position p is raw row p — no AR-inherited shift (contrast Dream,
  which shifts by one). Family quirks stay behind the seam.
* **Structure goes in ``attention_bias``.** LLaDA takes the pairwise [1,1,n,n]
  additive bias for the block-causal/bidirectional structure; its ``attention_mask``
  is a separate padding mask. (Dream forwards a 4-D ``attention_mask`` instead.)
* **``mask_token_id`` is a convention (126336), absent from the tokenizer.**
* **No KV reuse.** The reference MDM modeling code asserts the cache off, so every
  pass is a full recompute and ``kv``/``recompute_kv`` raise. The prefix K/V *is*
  exactly cacheable under block-causal attention (the one-way law — verified: the
  prefix logits are bitwise-invariant to the active block), but exploiting that is
  the C++ runtime's job, not this reference-wrapping adapter's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

from cloze_lab.models.base import (
    BoolArray,
    Family,
    ForwardResult,
    IntArray,
    KVState,
    LoadConfig,
    ModelConfig,
    check_attn_mask,
    check_board,
    check_indices,
)

LLADA_8B_INSTRUCT = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_TOKEN_ID = 126336  # the LLaDA [MASK] id; a convention, not on the tokenizer

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class _NoReuseKV:
    """Placeholder KVState: LLaDA recomputes every pass, so this is never reusable."""

    seq_len: int


class LLaDAAdapter:
    """ModelAdapter for LLaDA-family checkpoints (LLaDA 8B Instruct/Base)."""

    def __init__(self, load: LoadConfig, *, quantization: str | None = None) -> None:
        if load.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {load.dtype!r}")
        self._tok = AutoTokenizer.from_pretrained(
            load.model_id, revision=load.revision, trust_remote_code=load.trust_remote_code
        )
        kwargs: dict = dict(
            revision=load.revision,
            torch_dtype=_DTYPES[load.dtype],
            trust_remote_code=load.trust_remote_code,
        )
        if quantization is not None:
            # bitsandbytes weight quant (GPU-only), so the 8B fits a 16 GB card (bf16 ~15 GB does
            # not); the product quantizes via GGUF/ggml. Mirrors DreamAdapter.
            from transformers import BitsAndBytesConfig

            if quantization == "nf4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_DTYPES[load.dtype],
                    bnb_4bit_quant_type="nf4",
                )
            elif quantization == "int8":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"quantization must be None|'nf4'|'int8', got {quantization!r}")
            kwargs["device_map"] = load.device  # bnb places the weights itself
            self._model = AutoModel.from_pretrained(load.model_id, **kwargs)
        else:
            self._model = AutoModel.from_pretrained(load.model_id, **kwargs)
            if load.device != "cpu":
                self._model = self._model.to(load.device)
        self._model.eval()
        self._device = load.device
        hf = self._model.config
        self._config = ModelConfig(
            family=Family.LLADA,
            vocab_size=hf.vocab_size,
            mask_token_id=LLADA_MASK_TOKEN_ID,
            eos_token_id=hf.eos_token_id,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    def forward(
        self,
        board: IntArray,
        attn_mask: BoolArray,
        *,
        kv: KVState | None = None,
        recompute_kv: Sequence[int] | None = None,
        logits_for: Sequence[int] | None = None,
    ) -> ForwardResult:
        if kv is not None or recompute_kv is not None:
            raise NotImplementedError(
                "LLaDAAdapter recomputes every pass: the reference MDM modeling code "
                "asserts the KV cache off. The prefix K/V is exactly cacheable under "
                "block-causal attention, but exploiting it belongs in the runtime."
            )
        board = check_board(board, self._config.vocab_size)
        n = board.shape[0]
        attn_mask = check_attn_mask(attn_mask, n)
        want = list(range(n)) if logits_for is None else check_indices(
            "logits_for", logits_for, n
        )

        ids = torch.from_numpy(board).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            out = self._model(input_ids=ids, attention_bias=self._attention_bias(attn_mask))
        # In-place head: the distribution for position p is raw row p (no shift).
        logits = out.logits[0, want].float().cpu().numpy()
        return ForwardResult(logits=logits, kv=_NoReuseKV(seq_len=n))

    def _attention_bias(self, attn_mask: BoolArray) -> torch.Tensor:
        """LLaDA's [1, 1, n, n] additive bias: 0 where attended, dtype-min where blocked."""
        n = attn_mask.shape[0]
        bias = torch.zeros((1, 1, n, n), dtype=torch.float32, device=self._device)
        bias.masked_fill_(
            torch.from_numpy(~attn_mask).to(self._device), torch.finfo(torch.float32).min
        )
        return bias

    def encode(self, text: str, *, chat: bool = False) -> list[int]:
        """Token ids for ``text``. ``chat=True`` wraps the prompt in the instruct chat template
        (LLaDA-Instruct otherwise emits its end token immediately on a raw prompt). Raw ``tok.encode``
        (the default) is the completion path the golden fixtures pin, so leaving ``chat`` off keeps
        those byte-identical."""
        if chat:
            return list(
                self._tok.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
                )
            )
        return self._tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids])
