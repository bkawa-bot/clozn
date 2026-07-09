"""test_fit_planner -- model-free tests for clozn/fit_planner.py (FRONTIER_BETS Sec.2: "will it fit?"
answered from a GGUF's header alone -- no download of the multi-GB tensor payload, no model load, no GPU).

Layout:
  * a tiny hand-built GGUF byte-string (magic + version + counts + a handful of metadata KVs, including
    a string array, + two tensor infos) drives parse_gguf_header() in complete isolation -- deterministic,
    no real model file needed -- and exercises the NeedMoreBytes growth contract directly (truncated
    buffers at several cut points) plus the bad-magic error path.
  * gguf_header_from_path() is driven against that same synthetic bytes written to a real temp file, with
    a deliberately tiny initial read size, so the "start small, grow on NeedMoreBytes" path actually runs
    through more than one iteration.
  * the two REAL local GGUFs named in FRONTIER_BETS (Qwen2.5-7B-Instruct-Q4_K_M.gguf, arch qwen2, 28 layers,
    32k ctx; Llama-3.2-1B-Instruct-Q4_K_M.gguf, arch llama) are parsed and checked against their known
    facts -- skipped (not failed) if this machine doesn't have them under ~/.clozn/models.
  * fit_report()'s fit/no-fit math and its offload hint, plus the mandatory "APPROXIMATE" honesty flag.
  * gguf_header_from_url() against one live HuggingFace GGUF -- best-effort: any network failure skips
    rather than fails, so the offline suite never depends on connectivity.
  * clozn/cli.py wiring: `plan` is registered in build_parser() with the expected flags, and format_plan()
    (the pure header+report -> terminal text renderer cli.py factors out) is driven with a canned dict.
"""
from __future__ import annotations

import os
import struct
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))          # tests/
REPO = os.path.dirname(HERE)                                # repo root
sys.path.insert(0, REPO)

from clozn import fit_planner                               # noqa: E402
from clozn import cli as clozn_cli                           # noqa: E402


# ---------------------------------------------------------------------------------- synthetic GGUF bytes
# Hand-rolled per the spec (github ggml-org/ggml docs/gguf.md): magic "GGUF" + version u32 + tensor_count
# u64 + metadata_kv_count u64, then that many KVs, then that many tensor infos. No real model needed --
# this is the same binary shape a real quantizer would write, just tiny.

def _u32(v: int) -> bytes: return struct.pack("<I", v)
def _u64(v: int) -> bytes: return struct.pack("<Q", v)
def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv_u32(key: str, v: int) -> bytes:
    return _str(key) + _u32(4) + _u32(v)                   # vtype 4 = uint32


def _kv_str(key: str, v: str) -> bytes:
    return _str(key) + _u32(8) + _str(v)                    # vtype 8 = string


def _kv_arr_str(key: str, items: list[str]) -> bytes:
    body = _u32(8) + _u64(len(items))                       # elem_type=string, count
    for it in items:
        body += _str(it)
    return _str(key) + _u32(9) + body                        # vtype 9 = array


def _tensor_info(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    out = _str(name) + _u32(len(dims))
    for d in dims:
        out += _u64(d)
    out += _u32(ggml_type) + _u64(offset)
    return out


def _build_gguf_bytes(*, arch="testarch", file_type=15, block_count=4, context_length=2048,
                      embedding_length=64, head_count=8, head_count_kv=8,
                      tensors=None) -> bytes:
    if tensors is None:
        # a Q4_K attention tensor + an F32 norm tensor -- like a real quantized checkpoint, norms stay F32
        tensors = [("blk.0.attn_q.weight", [64, 64], 12, 0),
                   ("blk.0.attn_norm.weight", [64], 0, 16384)]

    kvs = b"".join([
        _kv_str("general.architecture", arch),
        _kv_u32("general.file_type", file_type),
        _kv_u32(f"{arch}.block_count", block_count),
        _kv_u32(f"{arch}.context_length", context_length),
        _kv_u32(f"{arch}.embedding_length", embedding_length),
        _kv_u32(f"{arch}.attention.head_count", head_count),
        _kv_u32(f"{arch}.attention.head_count_kv", head_count_kv),
        _kv_arr_str("general.tags", ["chat", "text-generation"]),
    ])
    kv_count = 8

    tinfo = b"".join(_tensor_info(*t) for t in tensors)
    body = _u32(3) + _u64(len(tensors)) + _u64(kv_count) + kvs + tinfo
    return b"GGUF" + body


SYNTH = _build_gguf_bytes()


# --------------------------------------------------------------------------------------- parse_gguf_header

def test_parse_synthetic_header_fields():
    h = fit_planner.parse_gguf_header(SYNTH)
    assert h["arch"] == "testarch"
    assert h["n_layers"] == 4
    assert h["context_length"] == 2048
    assert h["embedding_length"] == 64
    assert h["head_count"] == 8
    assert h["head_count_kv"] == 8
    assert h["quant"] == "Q4_K_M"                # from general.file_type=15, not the tensor histogram
    assert h["quant_source"] == "general.file_type"
    assert h["tensor_count"] == 2
    assert h["metadata"]["general.tags"] == ["chat", "text-generation"]
    assert h["header_bytes"] <= len(SYNTH)


def test_parse_falls_back_to_dominant_tensor_type_without_file_type():
    # Drop general.file_type from the KV block entirely -- quant must fall back to the tensor histogram
    # and say so, since it can no longer tell a _M from a _S/_L variant.
    tensors = [("blk.0.a", [8, 8], 12, 0), ("blk.0.b", [8, 8], 12, 256), ("blk.0.norm", [8], 0, 512)]
    kvs = b"".join([
        _kv_str("general.architecture", "testarch"),
        _kv_u32("testarch.block_count", 1),
        _kv_u32("testarch.context_length", 512),
        _kv_u32("testarch.embedding_length", 8),
        _kv_u32("testarch.attention.head_count", 1),
    ])
    tinfo = b"".join(_tensor_info(*t) for t in tensors)
    data = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(5) + kvs + tinfo
    h = fit_planner.parse_gguf_header(data)
    assert h["quant"] == "Q4_K"                    # dominant non-F32 tensor type (2 of 3 tensors)
    assert "approximate" in h["quant_source"]
    assert h["file_type"] is None


def test_bad_magic_raises_value_error_not_need_more_bytes():
    data = b"OOPS" + SYNTH[4:]
    with pytest.raises(ValueError) as ei:
        fit_planner.parse_gguf_header(data)
    assert not isinstance(ei.value, fit_planner.NeedMoreBytes)
    assert "not a GGUF file" in str(ei.value)


@pytest.mark.parametrize("cut", [0, 3, 10, 24, 40, len(SYNTH) - 5])
def test_truncated_buffer_raises_need_more_bytes_with_clear_message(cut):
    with pytest.raises(fit_planner.NeedMoreBytes) as ei:
        fit_planner.parse_gguf_header(SYNTH[:cut])
    msg = str(ei.value)
    assert msg.startswith("need more bytes: got ")
    assert str(cut) in msg                          # "got <cut>" -- the caller can see exactly what it had
    assert ei.value.need_at_least > cut              # and exactly how much more it needs to try next


# ----------------------------------------------------------------------------------- gguf_header_from_path

def test_gguf_header_from_path_grows_the_read_until_it_fits(tmp_path):
    path = tmp_path / "synthetic.gguf"
    path.write_bytes(SYNTH)
    # initial_bytes deliberately smaller than the full header -- forces >1 grow-and-retry iteration
    h = fit_planner.gguf_header_from_path(str(path), initial_bytes=16, max_bytes=1 << 20)
    assert h["arch"] == "testarch"
    assert h["n_layers"] == 4
    assert h["quant"] == "Q4_K_M"
    assert h["file_size_bytes"] == len(SYNTH)
    assert h["path"] == os.path.abspath(str(path))


def test_gguf_header_from_path_raises_if_max_bytes_too_small(tmp_path):
    path = tmp_path / "synthetic.gguf"
    path.write_bytes(SYNTH)
    with pytest.raises(fit_planner.NeedMoreBytes):
        fit_planner.gguf_header_from_path(str(path), initial_bytes=8, max_bytes=8)


# ------------------------------------------------------------------------- ground truth: the two real GGUFs
# FRONTIER_BETS's own ground truth. Skipped (not failed) on a machine without these under ~/.clozn/models --
# they're multi-GB local models, not something the offline suite can assume or fetch.

_MODELS_DIR = os.path.expanduser("~/.clozn/models")
QWEN_PATH = os.path.join(_MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
LLAMA_PATH = os.path.join(_MODELS_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")


@pytest.mark.skipif(not os.path.isfile(QWEN_PATH), reason="ground-truth Qwen2.5-7B GGUF not on this machine")
def test_local_qwen_header_matches_ground_truth():
    h = fit_planner.gguf_header_from_path(QWEN_PATH)
    assert h["arch"] == "qwen2"
    assert h["quant"] == "Q4_K_M"
    assert h["quant_source"] == "general.file_type"
    assert h["context_length"] == 32768                     # 32k trained context
    assert h["n_layers"] == 28
    assert h["embedding_length"] == 3584
    assert h["head_count"] == 28
    assert h["file_size_bytes"] == os.path.getsize(QWEN_PATH)


@pytest.mark.skipif(not os.path.isfile(LLAMA_PATH), reason="ground-truth Llama-3.2-1B GGUF not on this machine")
def test_local_llama_header_matches_ground_truth():
    h = fit_planner.gguf_header_from_path(LLAMA_PATH)
    assert h["arch"] == "llama"
    assert h["quant"] == "Q4_K_M"
    assert h["n_layers"] == 16
    assert h["file_size_bytes"] == os.path.getsize(LLAMA_PATH)


# --------------------------------------------------------------------------------------------- fit_report

QWEN_LIKE_HEADER = {"n_layers": 28, "embedding_length": 3584, "head_count": 28, "head_count_kv": 4}
QWEN_LIKE_SIZE = 4_683_074_240   # the real file size of the Q4_K_M GGUF on disk


def test_fit_report_fits_on_ample_vram_and_flags_approximate():
    r = fit_planner.fit_report(QWEN_LIKE_HEADER, QWEN_LIKE_SIZE, vram_gb=16)
    assert r["fits"] is True
    assert r["offload_hint"] is None
    assert "APPROXIMATE" in r["note"]
    assert r["est_vram_gb"] > QWEN_LIKE_SIZE / 1e9            # file size alone is a lower bound


def test_fit_report_does_not_fit_on_tight_vram_and_gives_offload_hint():
    r = fit_planner.fit_report(QWEN_LIKE_HEADER, QWEN_LIKE_SIZE, vram_gb=2)
    assert r["fits"] is False
    assert r["offload_hint"] is not None
    assert "/28" in r["offload_hint"]


def test_fit_report_never_promises_the_estimate():
    r = fit_planner.fit_report(QWEN_LIKE_HEADER, QWEN_LIKE_SIZE, vram_gb=16)
    assert "promise" in r["note"].lower()


# ------------------------------------------------------------------------------- gguf_header_from_url (live)
# Best-effort only: the offline suite must stay green (and fast) with no network, so any failure OR slowness
# here is a skip, never a failure. Validates that a real HuggingFace resolve/main/*.gguf URL answers Range
# and that our growth loop (starting ~1 MiB) lands on a correctly parsed header without downloading the
# whole (few-hundred-MB) file.
#
# Run off a daemon thread with a hard join timeout rather than relying on gguf_header_from_url's own
# `timeout=` alone: some networks (this development sandbox included -- verified directly with raw
# sockets) hand back IPv6 addresses first for huggingface.co but silently blackhole IPv6 egress, and
# stdlib's http.client tries every resolved address in order before falling back to IPv4 -- so a single
# request can serially re-time-out across several v6 candidates and take minutes even though it would
# eventually succeed. A daemon thread means an abandoned slow attempt can't block pytest's own exit either.

LIVE_URL = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q8_0.gguf"


def test_gguf_header_from_url_live_optional():
    import threading

    box: dict = {}

    def _run():
        try:
            box["header"] = fit_planner.gguf_header_from_url(LIVE_URL, timeout=3.0)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=10.0)
    if t.is_alive():
        pytest.skip("network too slow (or blocked) within the 10s test budget -- skipping live HF Range check")
    if "error" in box:
        pytest.skip(f"network unavailable/blocked, skipping live HF Range check: {box['error']}")

    h = box["header"]
    assert h["arch"] == "qwen2"
    assert h["quant"]
    assert h["file_size_bytes"] and h["file_size_bytes"] > 0
    assert h["bytes_fetched"] < h["file_size_bytes"]          # proof we did NOT download the whole file


# ------------------------------------------------------------------------------------------ cli.py wiring

def test_cli_plan_subcommand_registered():
    p = clozn_cli.build_parser()
    args = p.parse_args(["plan", "qwen", "--vram", "8"])
    assert args.fn is clozn_cli.cmd_plan
    assert args.model == "qwen"
    assert args.vram == 8.0


def test_cli_plan_default_vram_is_none_until_detected():
    p = clozn_cli.build_parser()
    args = p.parse_args(["plan", "qwen"])
    assert args.vram is None


def test_format_plan_pure_render_from_canned_dicts():
    header = dict(QWEN_LIKE_HEADER, quant="Q4_K_M", quant_source="general.file_type",
                  context_length=32768)
    report = fit_planner.fit_report(header, QWEN_LIKE_SIZE, vram_gb=16)
    text = clozn_cli.format_plan("Qwen2.5-7B-Instruct", header, QWEN_LIKE_SIZE, report, 16.0)
    assert "Q4_K_M" in text
    assert "28 layers" in text
    assert "32k ctx" in text
    assert "FITS" in text
    assert "APPROXIMATE" in text


def test_format_plan_shows_wont_fit_and_offload_hint():
    header = dict(QWEN_LIKE_HEADER, quant="Q4_K_M", quant_source="general.file_type",
                  context_length=32768)
    report = fit_planner.fit_report(header, QWEN_LIKE_SIZE, vram_gb=2)
    text = clozn_cli.format_plan("Qwen2.5-7B-Instruct", header, QWEN_LIKE_SIZE, report, 2.0)
    assert "WON'T FIT" in text
    assert "/28" in text


def test_format_plan_flags_dominant_tensor_type_guess():
    header = dict(QWEN_LIKE_HEADER, quant="Q4_K", quant_source="dominant tensor type (approximate -- "
                  "general.file_type metadata absent)", context_length=32768)
    report = fit_planner.fit_report(header, QWEN_LIKE_SIZE, vram_gb=16)
    text = clozn_cli.format_plan("mystery-model", header, QWEN_LIKE_SIZE, report, 16.0)
    assert "guess from the dominant tensor type" in text
