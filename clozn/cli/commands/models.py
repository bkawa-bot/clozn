"""commands.models -- model discovery (`clozn models`), fetch (`clozn pull`), and the fit planner
(`clozn plan`): "will it fit?" answered by reading only a GGUF's header -- a few MB at the front of the
file -- never the multi-GB tensor payload, never a model load, never the GPU.

resolve_model()/_flags_for()/_friendly() are also used by the run/serve/explain command modules to turn a
model name/path argument into a resolved GGUF + its launch flags.

HOME/CloznError live on `clozn.cli.main`; every function here that needs either does
`from clozn.cli import main as ctx` INSIDE the function body (never at module level -- main.py imports
this module at its own module level, so a module-level back-reference would deadlock the first time
something imports clozn.cli.commands.models before clozn.cli.main; see engine_process.py's docstring for
the full trace).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.engine_process import ENGINE_CORE, REPO, find_engine

# Known models: a filename fragment -> friendly name + launch flags. mask/eos => diffusion; chat => wrap the
# prompt in the chat template; AR models need no special flags (the engine auto-detects mode from the GGUF).
KNOWN = [
    ("qwen2.5-7b-instruct",   "qwen",      {"chat": True}),
    ("qwen2.5-0.5b-instruct", "qwen-0.5b", {"chat": True}),
    ("dream-v0-instruct",     "dream",     {"chat": True, "mask": 151666}),
    ("llada-8b-instruct",     "llada",     {"chat": True, "mask": 126336, "eos": 126081}),
    ("open-dcoder",           "dcoder",    {"mask": 151666}),
    ("mistral-7b-instruct",   "mistral",   {"chat": True, "tmpl": "mistral"}),
    ("llama-3.2-1b-instruct", "llama-1b",  {"chat": True, "tmpl": "llama3"}),
    ("llama-3.2-3b-instruct", "llama-3b",  {"chat": True, "tmpl": "llama3"}),
    ("gemma-2-2b-it",         "gemma-2b",  {"chat": True, "tmpl": "gemma"}),
]

# Models `clozn pull` knows how to fetch: name -> (HF repo, file). Verified ungated single-file GGUFs.
# Anything else: `clozn pull owner/repo/file.gguf`.
PULLABLE = {
    "qwen-0.5b": ("bartowski/Qwen2.5-0.5B-Instruct-GGUF",    "Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
    "qwen":      ("bartowski/Qwen2.5-7B-Instruct-GGUF",      "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "mistral":   ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
    "llama-1b":  ("bartowski/Llama-3.2-1B-Instruct-GGUF",    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "llama-3b":  ("bartowski/Llama-3.2-3B-Instruct-GGUF",    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "gemma-2b":  ("bartowski/gemma-2-2b-it-GGUF",            "gemma-2-2b-it-Q4_K_M.gguf"),
}


# ----------------------------------------------------------------------------- discovery

def _model_dirs() -> list[str]:
    from clozn.cli import main as ctx
    dirs = []
    if os.environ.get("CLOZN_MODELS"):
        dirs += os.environ["CLOZN_MODELS"].split(os.pathsep)
    cfg = os.path.join(ctx.HOME, "config.json")
    if os.path.isfile(cfg):
        try:
            dirs += json.load(open(cfg)).get("model_dirs", [])
        except Exception:
            pass
    dirs += [os.path.join(ctx.HOME, "models"), os.path.join(REPO, "models"),
             os.path.join(ENGINE_CORE, "models")]
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def _scan_models() -> list[str]:
    found = []
    for d in _model_dirs():
        found += glob.glob(os.path.join(d, "*.gguf"))
    return sorted(set(found))


def _flags_for(path: str) -> dict:
    base = os.path.basename(path).lower()
    for frag, _name, flags in KNOWN:
        if frag in base:
            return dict(flags)
    # Unknown GGUF: assume autoregressive instruct (the common case); engine still auto-detects mode.
    return {"chat": "instruct" in base or "chat" in base}


def _friendly(path: str) -> str:
    base = os.path.basename(path).lower()
    for frag, name, _ in KNOWN:
        if frag in base:
            return name
    return os.path.splitext(os.path.basename(path))[0]


def resolve_model(arg: str) -> str:
    """A path, a known short name, or a fuzzy filename fragment -> an absolute GGUF path."""
    from clozn.cli import main as ctx
    if arg.lower().endswith(".gguf") and os.path.isfile(arg):
        return os.path.abspath(arg)
    models = _scan_models()
    if not models:
        raise ctx.CloznError("no GGUF models found. Put .gguf files in ~/.clozn/models or set CLOZN_MODELS=<dir>.")
    # exact known short-name
    for frag, name, _ in KNOWN:
        if arg.lower() == name:
            for m in models:
                if frag in os.path.basename(m).lower():
                    return m
    # fuzzy: filename contains the arg
    hits = [m for m in models if arg.lower() in os.path.basename(m).lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ctx.CloznError(f"'{arg}' is ambiguous: {', '.join(_friendly(h) for h in hits)}. Be more specific.")
    avail = ", ".join(sorted({_friendly(m) for m in models}))
    raise ctx.CloznError(f"model '{arg}' not found. Available: {avail}.")


def cmd_models(_args):
    from clozn.cli import main as ctx
    models = _scan_models()
    try:
        _, _, gpu = find_engine()
        eng = f"{fmt.BOLD}{'GPU' if gpu else 'CPU'} build{fmt.RST} found"
    except ctx.CloznError:
        eng = f"{fmt.BOLD}no engine built{fmt.RST} (run: cd engine/core && build_gpu.bat)"
    print(f"engine: {eng}")
    if not models:
        print(f"\nno models found. dirs searched: {', '.join(_model_dirs()) or '(none)'}")
        print("put .gguf files in ~/.clozn/models, or set CLOZN_MODELS=<dir>.")
        return
    print(f"\n{'NAME':<14} {'SIZE':>7}  {'KIND':<11} PATH")
    for m in models:
        size = f"{os.path.getsize(m)/1e9:.1f}G"
        flags = _flags_for(m)
        kind = "diffusion" if "mask" in flags else "autoregress"
        print(f"{_friendly(m):<14} {size:>7}  {kind:<11} {m}")
    print(f"\nrun one:  clozn run {_friendly(models[0])} \"your prompt\"")


def _rm(p):
    try:
        os.remove(p)
    except Exception:
        pass


def cmd_pull(args):
    from clozn.cli import main as ctx
    spec = args.model
    if spec in PULLABLE:
        repo, file = PULLABLE[spec]
    elif spec.endswith(".gguf") and spec.count("/") >= 2:
        parts = spec.split("/"); repo, file = "/".join(parts[:-1]), parts[-1]
    else:
        raise ctx.CloznError(f"don't know how to pull '{spec}'. Known: {', '.join(PULLABLE)}. "
                             f"Or give an explicit  owner/repo/file.gguf")
    dest_dir = os.path.join(ctx.HOME, "models"); os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, file)
    if os.path.isfile(dest):
        print(f"already have {file} ({os.path.getsize(dest) / 1e9:.1f}G)"); return
    url = f"https://huggingface.co/{repo}/resolve/main/{file}?download=true"
    print(f"{fmt.DIM}pulling{fmt.RST} {file}  {fmt.DIM}from {repo}{fmt.RST}", file=sys.stderr)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clozn/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0; t0 = time.time(); last = 0.0
            with open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); done += len(b)
                    now = time.time()
                    if now - last > 0.4 or done == total:
                        last = now
                        sp = done / 1e6 / max(0.1, now - t0)
                        head = (f"{fmt._confbar(done / total)} {done / total * 100:5.1f}%  "
                                f"{done / 1e9:.2f}/{total / 1e9:.2f} GB") if total else f"{done / 1e9:.2f} GB"
                        sys.stderr.write(f"\r  {head}  {sp:4.0f} MB/s   "); sys.stderr.flush()
        sys.stderr.write("\n")
        os.replace(tmp, dest)
    except urllib.error.HTTPError as e:
        _rm(tmp)
        raise ctx.CloznError(f"{repo}/{file} not found on HuggingFace (404)." if e.code == 404
                             else f"download failed (HTTP {e.code}).")
    except Exception as e:
        _rm(tmp)
        raise ctx.CloznError(f"download failed: {e}")
    print(f"saved {_friendly(dest)} ({os.path.getsize(dest) / 1e9:.1f}G).  "
          f"run it:  clozn run {_friendly(dest)} \"hello\"")


# ------------------------------------------------------------------ plan (fit check, no download, no load)
# `clozn plan <name|path|url>` answers "will it fit?" by reading only a GGUF's header -- a few MB at the
# front of the file -- never the multi-GB tensor payload, never a model load, never the GPU. Local models
# read straight off disk; a name `clozn pull` knows but hasn't fetched yet is read remotely over HTTP Range
# against HuggingFace, so the fit check happens BEFORE the download it's trying to save you from.

def _fmt_ctx(n) -> str:
    if not n:
        return "?"
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def _detect_vram_gb():
    """Best-effort local VRAM budget via `nvidia-smi` -- a driver metadata query, not a CUDA context: it
    doesn't allocate anything or run compute, so it's consistent with this being a CPU-only, no-GPU-use
    planner. Returns None (caller falls back to a default) if nvidia-smi isn't there or times out."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return round(float(out.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:
        pass
    return None


def format_plan(name: str, header: dict, file_size_bytes: int, report: dict, vram_gb: float,
                ctx_for_estimate: int = 8192, source_note: str = "") -> str:
    """Pure dict(s)->text render of a fit_planner report (no I/O -- testable with canned dicts), e.g.:
    'Qwen2.5 7B Instruct Q4_K_M -- 28 layers, 32k ctx, 4.4 GB file
     on 16 GB VRAM: FITS (~est 5.3 GB at 8k ctx, KV-q8)'"""
    quant = header.get("quant", "?")
    n_layers = header.get("n_layers")
    size_gb = (file_size_bytes or 0) / 1e9
    verdict = f"{fmt.BOLD}FITS{fmt.RST}" if report["fits"] else f"{fmt.BOLD}WON'T FIT{fmt.RST}"
    lines = [
        f"{name} {quant} — {n_layers if n_layers is not None else '?'} layers, "
        f"{_fmt_ctx(header.get('context_length'))} ctx, {size_gb:.1f} GB file"
        + (f"  {fmt.DIM}{source_note}{fmt.RST}" if source_note else ""),
        f"on {vram_gb:g} GB VRAM: {verdict}  (~est {report['est_vram_gb']:.1f} GB at "
        f"{_fmt_ctx(ctx_for_estimate)} ctx, KV-q8)",
    ]
    if report.get("offload_hint"):
        lines.append(f"  {report['offload_hint']}")
    lines.append(f"{fmt.DIM}{report['note']}{fmt.RST}")
    if header.get("quant_source") and header["quant_source"] != "general.file_type":
        lines.append(f"{fmt.DIM}quant is a guess from the dominant tensor type ({header['quant_source']}).{fmt.RST}")
    return "\n".join(lines)


def cmd_plan(args):
    from clozn.cli import fit_planner   # stdlib+urllib only; imported lazily like the other clozn.* siblings
    from clozn.cli import main as ctx

    vram_gb = args.vram if args.vram is not None else (_detect_vram_gb() or 16.0)
    spec = args.model

    # gguf_header_from_path/url raise ValueError ("not a GGUF file: ...") on a malformed header, or its
    # subclass NeedMoreBytes on a truncated one (the header didn't fit even at max_bytes) -- both are
    # facts about the FILE, not a bug in this command, so they get the same clean one-line CloznError
    # exit every other bad-input path here already uses; main() only catches CloznError, so anything
    # else here would otherwise surface as a raw traceback.
    try:
        if spec.startswith("http://") or spec.startswith("https://"):
            header = fit_planner.gguf_header_from_url(spec)
            size = header.get("file_size_bytes") or 0
            name = header.get("name") or spec.rsplit("/", 1)[-1]
            source_note = f"remote, not downloaded: {spec}"
        else:
            path = spec if (spec.lower().endswith(".gguf") and os.path.isfile(spec)) else None
            if path is None:
                try:
                    path = resolve_model(spec)
                except ctx.CloznError:
                    if spec not in PULLABLE:
                        raise
                    repo, file = PULLABLE[spec]
                    url = f"https://huggingface.co/{repo}/resolve/main/{file}"
                    print(f"{fmt.DIM}- '{spec}' isn't downloaded yet -- reading its header straight off "
                         f"HuggingFace (no download){fmt.RST}", file=sys.stderr)
                    header = fit_planner.gguf_header_from_url(url)
                    size = header.get("file_size_bytes") or 0
                    name = spec
                    source_note = f"not downloaded -- `clozn pull {spec}` fetches it: {url}"
                    path = None
            if path is not None:
                header = fit_planner.gguf_header_from_path(path)
                size = header["file_size_bytes"]
                name = _friendly(path)
                source_note = path
    except ValueError as e:
        raise ctx.CloznError(f"couldn't read '{spec}' as a GGUF: {e}")

    report = fit_planner.fit_report(header, size, vram_gb)
    print(format_plan(name, header, size, report, vram_gb, source_note=source_note))
