"""Dream 7B adapter — first real checkpoint behind the seam (DESIGN §4.4 target #1).

torch/transformers imports are allowed here and only here (DESIGN invariant 1).

KV reuse (DESIGN §5.5) is **prefix-only**: the adapter reuses a frozen prefix from
a prior ``DreamKV`` (a transformers ``DynamicCache``) and recomputes a contiguous
suffix — Tier A/B exact caching, which under block-causal attention reproduces the
full-recompute picks (confidences within float epsilon). Tier C's scattered
mid-sequence recompute is not expressible via an append-only cache, so a
non-contiguous ``recompute_kv`` raises. Pass no ``kv`` for a full recompute
(cold start / cache off).

Facts about Dream's modeling code this adapter relies on (verified against the
cached ``modeling_dream.py`` / ``generation_utils.py``): attention is hard-coded
non-causal; the ``attention_mask`` argument is forwarded to the layers verbatim,
so it must be either None (full attention) or a ready-made additive 4-D float
mask [batch, 1, q, k] with 0 = attend and dtype-min = blocked. And crucially,
Dream keeps its AR parent's *shifted* prediction head: the distribution for the
token at position p comes from the hidden state at p-1 (their generation code
does ``cat([logits[:, :1], logits[:, :-1]])``). This adapter applies that shift,
so seam consumers always read row p as "the distribution for position p" —
in-place predictors like LLaDA will simply not shift. Family quirks stay here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache

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

DREAM_7B_INSTRUCT = "Dream-org/Dream-v0-Instruct-7B"
OPEN_DCODER_05B = "fredzzp/open-dcoder-0.5B"  # tiny Dream-family sibling for CPU CI

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class DreamKV:
    """Wraps a transformers ``DynamicCache`` (per-layer K/V) as an opaque KVState.

    The cache is append-only, so Dream reuses an exact *prefix* and recomputes a
    contiguous suffix (Tier A/B); Tier C's scattered mid-sequence recompute is not
    expressible here. ``seq_len`` is how many positions the cache currently covers.
    """

    cache: DynamicCache
    seq_len: int


class DreamAdapter:
    """ModelAdapter for Dream-family checkpoints (Dream 7B Instruct/Base)."""

    def __init__(
        self,
        load: LoadConfig,
        *,
        quantization: str | None = None,
        model_loader: type = AutoModel,
        mask_token_id: int | None = None,
    ) -> None:
        # ``model_loader``/``mask_token_id`` let tiny Dream-family siblings load behind
        # the same adapter: open-dCoder 0.5B is a plain ``Qwen2ForCausalLM`` (so
        # AutoModelForCausalLM, not AutoModel) whose ``<M>`` mask is absent from the
        # config. The shifted head and forward path are identical. See open_dcoder_adapter.
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
            # bitsandbytes weight quant (GPU-only) — a lab knob for the §9
            # quantization-vs-confidence study; the product quantizes via GGUF/ggml.
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
            kwargs["device_map"] = load.device  # bnb places weights itself
            self._model = model_loader.from_pretrained(load.model_id, **kwargs)
        else:
            self._model = model_loader.from_pretrained(load.model_id, **kwargs)
            if load.device != "cpu":
                self._model = self._model.to(load.device)
        self._model.eval()
        self._device = load.device
        hf = self._model.config
        mask = mask_token_id if mask_token_id is not None else getattr(hf, "mask_token_id", None)
        if mask is None:
            raise ValueError(
                "mask_token_id is absent from the checkpoint config; pass it explicitly "
                "(open-dCoder's <M> is 151665)."
            )
        self._config = ModelConfig(
            family=Family.DREAM,
            vocab_size=hf.vocab_size,
            mask_token_id=mask,
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
        board = check_board(board, self._config.vocab_size)
        n = board.shape[0]
        attn_mask = check_attn_mask(attn_mask, n)
        want = list(range(n)) if logits_for is None else check_indices(
            "logits_for", logits_for, n
        )

        rstart, cache = self._reuse_prefix(kv, recompute_kv, n, want)

        ids = torch.from_numpy(board[rstart:n]).unsqueeze(0).to(self._device)
        pos = torch.arange(rstart, n, device=self._device)
        mask4d = self._additive_mask(attn_mask[rstart:n, :])
        with torch.inference_mode():
            out = self._model(
                input_ids=ids,
                attention_mask=mask4d,
                past_key_values=cache,
                use_cache=True,
                position_ids=pos.unsqueeze(0),
                cache_position=pos,
            )
        # Dream's shifted head: the distribution for position p is raw row p-1 (row 0
        # is its own filler). out.logits covers rows [rstart, n); index relative to it.
        raw = out.logits[0]
        rows = [max(p - 1, 0) - rstart for p in want]
        logits = raw[rows].float().cpu().numpy()
        return ForwardResult(logits=logits, kv=DreamKV(cache=cache, seq_len=n))

    def _reuse_prefix(
        self,
        kv: KVState | None,
        recompute_kv: Sequence[int] | None,
        n: int,
        want: list[int],
    ) -> tuple[int, DynamicCache]:
        """Resolve (recompute start, cache) for this forward; crop reused drawers.

        Reuse is prefix-only: ``recompute_kv`` must be a contiguous suffix. The start
        is pulled back one position when needed so the shifted head has its source
        row for a masked first active slot (that extra slot is frozen-exact)."""
        if kv is None:
            if recompute_kv is not None:
                raise ValueError("recompute_kv given without kv: nothing to reuse")
            return 0, DynamicCache()
        if not isinstance(kv, DreamKV):
            raise TypeError(f"DreamAdapter got foreign KVState {type(kv).__name__}")
        recompute = list(range(n)) if recompute_kv is None else check_indices(
            "recompute_kv", recompute_kv, n
        )
        s = recompute[0] if recompute else n
        if recompute != list(range(s, n)):
            raise NotImplementedError(
                "DreamAdapter supports only contiguous-suffix KV reuse (Tier A/B "
                "prefix caching); Tier C scattered recompute is not expressible via "
                "an append-only KV cache. Use cache full_refresh_every=1 in block mode."
            )
        rstart = min([s] + [max(p - 1, 0) for p in want]) if want else s
        kv.cache.crop(rstart)
        return rstart, kv.cache

    def _additive_mask(self, rows: BoolArray) -> torch.Tensor:
        """4-D additive mask [1, 1, q, k] for the query rows over all keys: 0 where
        attended, dtype-min where blocked. Always explicit — never None.

        A None ``attention_mask`` is unsafe in the KV-reuse path for stock-Qwen2
        siblings (open-dCoder): with a non-empty ``DynamicCache``, Qwen2 falls back to
        *causal* masking over past+current, silently breaking the bidirectional
        one-way law and corrupting prefix reuse. An explicit all-visible (all-zero)
        mask forces true bidirectional attention; for Dream (which forwards the mask
        verbatim) it is identical to None. Verified bitwise-exact on both families."""
        dt = self._model.dtype
        m = torch.zeros((1, 1, *rows.shape), dtype=dt, device=self._device)
        m.masked_fill_(torch.from_numpy(~rows).to(self._device), torch.finfo(dt).min)
        return m

    def encode(self, text: str, *, chat: bool = False) -> list[int]:
        """Token ids for ``text``.

        ``chat=True`` wraps the prompt in the instruct chat template (system + user
        turns, assistant turn opened) before tokenizing — Dream-Instruct otherwise
        emits its end token immediately on a raw instruction. Raw ``tok.encode`` (the
        default) is the completion path the golden fixtures pin, so leaving ``chat``
        off keeps those byte-identical."""
        if chat:
            return list(
                self._tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
        return self._tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids])


def open_dcoder_adapter(load: LoadConfig) -> DreamAdapter:
    """open-dCoder 0.5B — the tiny CPU-CI checkpoint (DESIGN: "CI must run on CPU
    with a tiny checkpoint").

    It's a Dream-family masked diffusion model distilled into a stock
    ``Qwen2ForCausalLM`` (so it loads via ``AutoModelForCausalLM``, not ``AutoModel``),
    with the same shifted prediction head Dream uses. Its ``<M>`` mask token (151665)
    lives only in ``added_tokens.json``, not the config, so we pass it explicitly.
    """
    from transformers import AutoModelForCausalLM

    return DreamAdapter(load, model_loader=AutoModelForCausalLM, mask_token_id=151665)
