"""throughput_predictor -- a model-free ROOFLINE estimate of decode tok/s for `clozn plan`.

The physics: autoregressive decode at batch size 1 is memory-BANDWIDTH bound, not compute bound. To
produce each new token, the engine must read essentially the ENTIRE quantized weight file plus the KV
cache built so far off (V)RAM once, then do a comparatively tiny amount of arithmetic on it. So:

    decode tok/s ~= effective_memory_bandwidth_bytes_s / bytes_read_per_token
    bytes_read_per_token ~= weight_bytes                              (whole model, read once/token)
                          + kv_bytes_per_token
    kv_bytes_per_token   = 2 (K and V) * n_layers * n_kv_heads * head_dim * ctx_position * kv_elem_bytes

Every input comes straight off the GGUF header fit_planner.parse_gguf_header() already parses (arch,
block count, embedding/head counts) -- plus, read here for the first time, the per-tensor {shape,
ggml_type} list that header carries as `tensors`. That list lets us compute the total parameter count
and total weight-byte count EXACTLY (sum of shape-product * bits-per-weight over every tensor) straight
from the header's tensor-info table -- no model load, no dominant-quant guessing, no GPU, no download
(the tensor DATA, which is the multi-GB part of a GGUF, is never read; only its declared shape+dtype).

This is a ROOFLINE: an upper-bound, order-of-magnitude estimate that ignores compute-bound prefill,
batching, and real kernel efficiency (actual hardware rarely sustains 100% of theoretical memory
bandwidth). It answers "is this ~3 tok/s or ~30 tok/s or ~300 tok/s?", not "is this exactly 24.7 tok/s?"
-- calibrate against a live run (see cmd_plan's --calibrate seam, deliberately deferred) for the real
number on real hardware.

    est = predict_throughput(header, bandwidth_gb_s=900.0)
    est["predicted_tok_s"]
"""
from __future__ import annotations

from collections import Counter

# Bits-per-weight for each ggml_type, computed directly from the block struct each quant format is
# defined as in ggml (ggml-quants.c/h) -- NOT an empirical "average bpw over a real model" number (those
# run a little higher because embedding/output/norm tensors are often kept at higher precision even in an
# otherwise-quantized file). That upgrade is already handled exactly here: every tensor in a GGUF carries
# its OWN ggml_type, so weight_bytes_and_params() below sums per-tensor bytes rather than assuming one
# quant format for the whole file.
#
#   ggml_type: bpw     # block=N: byte layout -> bytes/block*8/N
BPW_BY_GGML_TYPE = {
    0:  32.0,      # F32
    1:  16.0,      # F16
    2:  4.5,       # Q4_0     block=32:  2 (d,f16)                          + 16 (4-bit x32)               = 18B
    3:  5.0,       # Q4_1     block=32:  2 (d) + 2 (m)                      + 16 (4-bit x32)               = 20B
    6:  5.5,       # Q5_0     block=32:  2 (d)              + 4 (hi bits)   + 16 (lo 4-bit x32)            = 22B
    7:  6.0,       # Q5_1     block=32:  2 (d) + 2 (m)      + 4 (hi bits)   + 16 (lo 4-bit x32)            = 24B
    8:  8.5,       # Q8_0     block=32:  2 (d)                              + 32 (int8 x32)                = 34B
    9:  9.0,       # Q8_1     block=32:  4 (d+s, 2xf16)                     + 32 (int8 x32)                = 36B
    10: 2.625,     # Q2_K     block=256: 2 (d) + 2 (dmin) + 16 (scales)     + 64 (2-bit x256)              = 84B
    11: 3.4375,    # Q3_K     block=256: 2 (d)            + 12 (scales) + 32 (hmask) + 64 (2-bit lo)       = 110B
    12: 4.5,       # Q4_K     block=256: 2 (d) + 2 (dmin) + 12 (scales)                + 128 (4-bit x256)  = 144B
    13: 5.5,       # Q5_K     block=256: 2 (d) + 2 (dmin) + 12 (scales) + 32 (hi) + 128 (lo 4-bit)         = 176B
    14: 6.5625,    # Q6_K     block=256: 2 (d)            + 16 (scales) + 64 (hi 2-bit) + 128 (lo 4-bit)   = 210B
    15: 9.125,     # Q8_K     block=256: 4 (d,f32)        + 32 (bsums)  + 256 (int8 x256)                  = 292B
    16: 2.0625,    # IQ2_XXS  approximate -- llama.cpp's own published bpw (codebook format, not re-derived here)
    17: 2.3125,    # IQ2_XS   approximate
    18: 3.0625,    # IQ3_XXS  approximate
    19: 1.5625,    # IQ1_S    approximate
    20: 4.5,       # IQ4_NL   block=32: 2 (d) + 16 (4-bit nonlinear-codebook x32) = 18B
    21: 3.4375,    # IQ3_S    approximate
    22: 2.5,       # IQ2_S    approximate
    23: 4.25,      # IQ4_XS   approximate
    28: 64.0,      # F64
    29: 1.75,      # IQ1_M    approximate
    30: 16.0,      # BF16
    34: 1.6875,    # TQ1_0    ternary, approximate
    35: 2.0625,    # TQ2_0    ternary, approximate
}

DEFAULT_BANDWIDTH_GB_S = 900.0        # RTX 5080-class GDDR7 -- "~900 GB/s-ish", stated explicitly (this
                                       # drives the whole estimate; override with --bandwidth-gb-s)
DEFAULT_KV_BYTES_PER_ELEMENT = 2.0    # fp16 KV cache (llama.cpp's default; a q8_0-cache run would be 1.0)


def weight_bytes_and_params(tensors: list[dict]) -> dict:
    """Sum EXACT parameter count and weight bytes straight off a GGUF's tensor-info table (name, shape,
    ggml_type per tensor -- header metadata only, never tensor DATA). Any tensor whose ggml_type isn't in
    BPW_BY_GGML_TYPE is excluded from the byte total (never silently misestimated) and counted separately
    so callers can flag it honestly."""
    n_params = 0
    weight_bytes = 0.0
    unknown_type_tensor_counts: Counter = Counter()
    unknown_params = 0
    for t in tensors:
        n = 1
        for d in t["shape"]:
            n *= d
        n_params += n
        bpw = BPW_BY_GGML_TYPE.get(t["ggml_type"])
        if bpw is None:
            unknown_type_tensor_counts[t["ggml_type"]] += 1
            unknown_params += n
            continue
        weight_bytes += n * bpw / 8.0
    return {
        "n_params": n_params,
        "weight_bytes": weight_bytes,
        "unknown_type_tensor_counts": dict(unknown_type_tensor_counts),
        "unknown_params": unknown_params,
    }


def predict_throughput(header: dict, *, bandwidth_gb_s: float = DEFAULT_BANDWIDTH_GB_S,
                       kv_bytes_per_element: float = DEFAULT_KV_BYTES_PER_ELEMENT,
                       ctx_for_estimate: int = 8192) -> dict:
    """Model-free decode-throughput ROOFLINE from GGUF header fields alone (see module docstring for the
    physics). Never raises on missing header fields -- falls back to 0, which format_throughput() renders
    as an honest "cannot estimate" rather than a crash."""
    tensors = header.get("tensors") or []
    wp = weight_bytes_and_params(tensors)

    n_layers = header.get("n_layers") or 0
    embedding_length = header.get("embedding_length") or 0
    head_count = header.get("head_count") or 0
    head_count_kv = header.get("head_count_kv") or head_count or 1
    head_dim = (embedding_length / head_count) if head_count else 0.0

    weight_bytes = wp["weight_bytes"]
    kv_bytes_per_token = 2 * n_layers * head_count_kv * head_dim * ctx_for_estimate * kv_bytes_per_element
    total_bytes_per_token = weight_bytes + kv_bytes_per_token

    bandwidth_bytes_s = bandwidth_gb_s * 1e9
    predicted_tok_s = (bandwidth_bytes_s / total_bytes_per_token) if total_bytes_per_token > 0 else 0.0

    return {
        "n_params": wp["n_params"],
        "weight_bytes": weight_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "total_bytes_per_token": total_bytes_per_token,
        "bandwidth_gb_s": bandwidth_gb_s,
        "kv_bytes_per_element": kv_bytes_per_element,
        "ctx_for_estimate": ctx_for_estimate,
        "predicted_tok_s": predicted_tok_s,
        "unknown_type_tensor_counts": wp["unknown_type_tensor_counts"],
        "unknown_params": wp["unknown_params"],
    }
