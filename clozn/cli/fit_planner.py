"""fit_planner -- answer "which model should I run / will it fit?" before a multi-GB download.

FRONTIER_BETS Sec.2: the metadata that decides fit -- architecture, quantization, layer/embedding
shapes, trained context length -- lives in a small, self-describing binary header at the very
START of a GGUF file (see the spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
The multi-GB tensor payload comes AFTER that header, so reading it needs only the first few MB:
a local file read is instant, and a remote HuggingFace URL only needs one HTTP Range request --
never the full download.

Stdlib + urllib only. CPU-only. Never loads the model, never touches a GPU -- this module only
ever reads and unpacks bytes.

    header = gguf_header_from_path("model.gguf")            # or gguf_header_from_url(url)
    report = fit_report(header, header["file_size_bytes"], vram_gb=16)

The four functions below are the whole surface: parse_gguf_header (pure, offline, takes bytes
already in hand), gguf_header_from_path / gguf_header_from_url (fetch just enough bytes and grow
on demand), and fit_report (the honest, approximate "will it fit" math).
"""
from __future__ import annotations

import os
import struct
import urllib.request
from collections import Counter

USER_AGENT = "clozn-fit-planner/0.1"

GGUF_MAGIC = b"GGUF"

# GGUF metadata value-type enum (the tag that precedes every KV's payload).
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = range(8)
_T_STRING, _T_ARRAY = 8, 9
_T_UINT64, _T_INT64, _T_FLOAT64 = 10, 11, 12

# ggml_type enum (ggml.h) -- what a *tensor's* dtype byte means. Only the ones that actually turn
# up in released GGUFs are named; anything else prints as "type<N>" rather than guessing.
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}
# Norm/bias-style tensors llama.cpp keeps at full precision even in a quantized model -- excluded
# when picking the "dominant" quant so a Q4_K_M model doesn't get misread as "mostly F32".
_HIGH_PRECISION_TYPES = {0, 1, 30}  # F32, F16, BF16

# general.file_type metadata (llama_ftype enum) -- the AUTHORITATIVE quant label when present: it's
# written by the quantizer itself and is the only way to tell a _S/_M/_L variant apart (they can
# share the exact same dominant tensor type and differ only in a handful of upgraded tensors).
LLAMA_FTYPE_LABELS = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 1024: "GUESSED",
}


class NeedMoreBytes(ValueError):
    """Raised by parse_gguf_header when the buffer it was given is truncated mid-header. Callers
    (gguf_header_from_path/url) catch this, grow the range/read, and retry from scratch."""

    def __init__(self, need_at_least: int, have: int):
        self.need_at_least = need_at_least
        self.have = have
        super().__init__(f"need more bytes: got {have}, need at least {need_at_least}")


class _Reader:
    """A bounds-checked little-endian cursor over an in-memory buffer. Any read past the end of
    the buffer raises NeedMoreBytes instead of a bare struct.error, naming exactly how many bytes
    would have been needed -- that's what lets the caller grow its HTTP Range/local read and retry."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def _take(self, fmt: str, size: int):
        end = self.pos + size
        if end > len(self.data):
            raise NeedMoreBytes(end, len(self.data))
        v = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos = end
        return v

    def bytes_(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise NeedMoreBytes(end, len(self.data))
        b = self.data[self.pos:end]
        self.pos = end
        return b

    def u8(self) -> int: return self._take("<B", 1)
    def i8(self) -> int: return self._take("<b", 1)
    def u16(self) -> int: return self._take("<H", 2)
    def i16(self) -> int: return self._take("<h", 2)
    def u32(self) -> int: return self._take("<I", 4)
    def i32(self) -> int: return self._take("<i", 4)
    def f32(self) -> float: return self._take("<f", 4)
    def bool_(self) -> bool: return self._take("<B", 1) != 0
    def u64(self) -> int: return self._take("<Q", 8)
    def i64(self) -> int: return self._take("<q", 8)
    def f64(self) -> float: return self._take("<d", 8)

    def str_(self) -> str:
        n = self.u64()
        return self.bytes_(n).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype == _T_UINT8: return self.u8()
        if vtype == _T_INT8: return self.i8()
        if vtype == _T_UINT16: return self.u16()
        if vtype == _T_INT16: return self.i16()
        if vtype == _T_UINT32: return self.u32()
        if vtype == _T_INT32: return self.i32()
        if vtype == _T_FLOAT32: return self.f32()
        if vtype == _T_BOOL: return self.bool_()
        if vtype == _T_STRING: return self.str_()
        if vtype == _T_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        if vtype == _T_UINT64: return self.u64()
        if vtype == _T_INT64: return self.i64()
        if vtype == _T_FLOAT64: return self.f64()
        raise ValueError(f"unknown GGUF metadata value type: {vtype}")


def _quant_label(metadata: dict, tensors: list[dict]) -> tuple[str, str]:
    """-> (label, source). Prefers general.file_type (exact, written by the quantizer); falls back
    to the dominant tensor ggml_type (can't tell _S/_M/_L apart -- flagged as approximate)."""
    ft = metadata.get("general.file_type")
    if isinstance(ft, int) and ft in LLAMA_FTYPE_LABELS:
        return LLAMA_FTYPE_LABELS[ft], "general.file_type"
    counts = Counter(t["ggml_type"] for t in tensors)
    if not counts:
        return "unknown", "no tensors in header"
    quantized = {t: c for t, c in counts.items() if t not in _HIGH_PRECISION_TYPES}
    pool = quantized or counts
    dominant = max(pool.items(), key=lambda kv: kv[1])[0]
    label = GGML_TYPE_NAMES.get(dominant, f"type{dominant}")
    return label, "dominant tensor type (approximate -- general.file_type metadata absent)"


def parse_gguf_header(data: bytes) -> dict:
    """Parse a GGUF binary header out of `data` (the first N bytes of a .gguf file/URL). Raises
    NeedMoreBytes if `data` is truncated before the header ends -- the caller should fetch more and
    retry. Returns a dict with the fields the fit planner needs (arch, context_length, n_layers,
    embedding_length, head_count, head_count_kv, quant, ...) plus the raw `metadata` dict and the
    parsed `tensors` list for anyone who wants more."""
    if len(data) < 4:
        raise NeedMoreBytes(4, len(data))
    if data[0:4] != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file: magic bytes {data[0:4]!r} != {GGUF_MAGIC!r}")

    r = _Reader(data)
    r.bytes_(4)                     # magic, already checked
    version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()

    metadata: dict = {}
    for _ in range(kv_count):
        key = r.str_()
        vtype = r.u32()
        metadata[key] = r.value(vtype)

    tensors: list[dict] = []
    for _ in range(tensor_count):
        name = r.str_()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ggml_type = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "shape": dims, "ggml_type": ggml_type, "offset": offset})

    arch = metadata.get("general.architecture", "") or ""

    def m(suffix, default=None):
        return metadata.get(f"{arch}.{suffix}", default)

    quant, quant_source = _quant_label(metadata, tensors)

    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": kv_count,
        "arch": arch,
        "name": metadata.get("general.name"),
        "context_length": m("context_length"),
        "n_layers": m("block_count"),
        "embedding_length": m("embedding_length"),
        "head_count": m("attention.head_count"),
        "head_count_kv": m("attention.head_count_kv") or m("attention.head_count"),
        "quant": quant,
        "quant_source": quant_source,
        "file_type": metadata.get("general.file_type"),
        "header_bytes": r.pos,          # how much of `data` the header actually used
        "metadata": metadata,
        "tensors": tensors,
    }


def gguf_header_from_path(path: str, initial_bytes: int = 1 << 20, max_bytes: int = 64 << 20) -> dict:
    """Read the first chunk of a local .gguf and parse it, growing the read if the header didn't
    fit (large tokenizer vocabularies routinely push real headers to several MB)."""
    size = os.path.getsize(path)
    n = min(initial_bytes, size) if size else initial_bytes
    data = b""
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            data = f.read(n)
            try:
                header = parse_gguf_header(data)
                break
            except NeedMoreBytes as e:
                if n >= size or n >= max_bytes:
                    raise
                n = min(max(e.need_at_least + 4096, n * 2), size, max_bytes)
    header["file_size_bytes"] = size
    header["path"] = os.path.abspath(path)
    header["bytes_read"] = len(data)
    return header


def _http_range_get(url: str, n: int, timeout: float):
    """GET bytes [0, n) via a Range request. Reads at most `n` bytes even if the server ignores
    the Range header and sends the whole file with a 200 -- this is what guarantees a fit-check
    never turns into an accidental multi-GB download."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes=0-{n - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(n)
        headers = dict(resp.headers.items())
    return data, headers


def _remote_size(url: str, timeout: float) -> int | None:
    """Total file size via HEAD; falls back to a 1-byte Range GET and its Content-Range total
    (some hosts, HuggingFace's CDN included, don't answer HEAD the same way as GET)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            if cl:
                return int(cl)
    except Exception:
        pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cr = resp.headers.get("Content-Range")   # "bytes 0-0/<total>"
            if cr and "/" in cr:
                return int(cr.rsplit("/", 1)[-1])
    except Exception:
        pass
    return None


def gguf_header_from_url(url: str, initial_bytes: int = 1 << 20, max_bytes: int = 64 << 20,
                          timeout: float = 30.0) -> dict:
    """HTTP Range the first chunk of a remote .gguf (start ~1 MiB, grow on NeedMoreBytes) and parse
    it, plus a separate size lookup for the full file. Works against HuggingFace
    `.../resolve/main/*.gguf` URLs, which honor Range without needing auth for public repos."""
    n = initial_bytes
    data, headers = b"", {}
    while True:
        data, headers = _http_range_get(url, n, timeout)
        try:
            header = parse_gguf_header(data)
            break
        except NeedMoreBytes as e:
            if n >= max_bytes:
                raise
            n = min(max(e.need_at_least + 4096, n * 2), max_bytes)

    size = _remote_size(url, timeout)
    if size is None:
        cr = headers.get("Content-Range")
        if cr and "/" in cr:
            try:
                size = int(cr.rsplit("/", 1)[-1])
            except ValueError:
                size = None
        elif headers.get("Content-Length") and headers.get("Content-Length").isdigit():
            # only trustworthy as a total if the server ignored our Range and sent everything
            if int(headers["Content-Length"]) > n:
                size = int(headers["Content-Length"])

    header["file_size_bytes"] = size
    header["url"] = url
    header["bytes_fetched"] = len(data)
    return header


def fit_report(header: dict, file_size_bytes: int, vram_gb: float, ctx_for_estimate: int = 8192) -> dict:
    """Rough "will it fit on this GPU" estimate: model file size (weights) + an approximate KV-cache
    term for an 8k-token context at KV-q8 + a rule-of-thumb runtime/compute-buffer overhead. This is
    a before-you-download sanity check, not a promise -- actual usage depends on the runtime,
    context actually used, and batch size."""
    file_size_bytes = file_size_bytes or 0
    n_layers = header.get("n_layers") or 0
    embedding_length = header.get("embedding_length") or 0
    head_count = header.get("head_count") or 0
    head_count_kv = header.get("head_count_kv") or head_count or 1
    head_dim = (embedding_length / head_count) if head_count else 0.0

    file_gb = file_size_bytes / 1e9
    # KV cache: 2 (K and V) * layers * kv-heads * head_dim * context, 1 byte/element at KV-q8.
    kv_bytes = 2 * n_layers * head_count_kv * head_dim * ctx_for_estimate
    kv_gb = kv_bytes / 1e9
    overhead_gb = max(0.3, file_gb * 0.08)   # compute buffers / dequant workspace, rule-of-thumb
    est_vram_gb = file_gb + kv_gb + overhead_gb

    fits = est_vram_gb <= vram_gb

    note = (f"APPROXIMATE: file size + KV-cache for a {ctx_for_estimate}-token context (KV-q8; "
            f"{n_layers} layers x {head_count_kv} kv-heads x {head_dim:.0f}-dim) + a rule-of-thumb "
            f"compute/workspace overhead. Real VRAM use depends on the runtime, batch size, and the "
            f"context actually used -- this is a before-you-download sanity check, not a promise.")

    offload_hint = None
    if not fits and n_layers:
        bytes_per_layer = file_size_bytes / n_layers   # rough: spreads embed/lm_head across layers too
        budget_bytes = max(0.0, vram_gb * 1e9 - kv_bytes - overhead_gb * 1e9)
        layers_fit = max(0, min(n_layers, int(budget_bytes // bytes_per_layer))) if bytes_per_layer else 0
        offload_hint = (f"~{layers_fit}/{n_layers} layers would fit on {vram_gb:g} GB VRAM; "
                         f"the rest would spill to CPU/RAM (slower).")

    return {
        "est_vram_gb": round(est_vram_gb, 2),
        "fits": fits,
        "note": note,
        "offload_hint": offload_hint,
    }
