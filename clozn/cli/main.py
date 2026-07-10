#!/usr/bin/env python3
"""clozn -- a boring, reliable front door to the local model engine.

The fast runtime is the C++ engine (cloze-server.exe). This wraps it so the daily path is one command:

    clozn run   <model> "<prompt>"     one-shot, streams tokens to the terminal
    clozn serve <model> [--port 8080]  bring up the OpenAI-compatible endpoint, print the base URL
    clozn models                       discover local GGUFs and the backend that would run them

Stdlib only (urllib/subprocess/json) -- no torch, no pip install -- so it stays dependency-free and quick.
It finds the engine build (GPU preferred), puts the right DLLs on PATH, picks per-model flags (diffusion
mask tokens, etc.), reports honestly what it's running on, and fails with one actionable line instead of a
stack trace. Model dirs: $CLOZN_MODELS, ~/.clozn/models, <repo>/models, ~/.clozn/config.json["model_dirs"].
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (parent of clozn/)
HOME = os.path.expanduser("~/.clozn")
ENGINE_CORE = os.path.join(REPO, "engine", "core")

# Engine builds, most-preferred first. (subdir, is_gpu); the exe sits at <subdir>/ or <subdir>/Release/.
BUILDS = [("build-gpu", True), ("build-cuda", True),
          ("build-ggml-cpu", False), ("build-serve", False), ("build-cpu", False)]

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

DIM = BOLD = RST = ""           # set by _setup_console() when the terminal supports ANSI
COLOR = False                   # truecolor confidence painting (denoise-style heatmap); set by _setup_console()


def _setup_console():
    """UTF-8 stdout (so model tokens print right on Windows), ANSI enabled where supported, plain otherwise."""
    global DIM, BOLD, RST, COLOR
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    use = sys.stderr.isatty() or bool(os.environ.get("CLICOLOR_FORCE"))   # CLICOLOR_FORCE: color even when piped
    if os.name == "nt" and sys.stderr.isatty():                          # only a real console needs VT enabling
        try:
            import ctypes
            k = ctypes.windll.kernel32
            for h in (-11, -12):                               # stdout, stderr handles
                hd = k.GetStdHandle(h); m = ctypes.c_uint32()
                if k.GetConsoleMode(hd, ctypes.byref(m)):
                    k.SetConsoleMode(hd, m.value | 0x0004)     # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            use = bool(os.environ.get("CLICOLOR_FORCE"))
    if use:
        DIM, BOLD, RST = "\033[2m", "\033[1m", "\033[0m"
        COLOR = not os.environ.get("NO_COLOR")   # truecolor heatmaps unless the user opts out (NO_COLOR is a std)


class CloznError(Exception):
    """A clean, user-facing failure -- printed as one line, no traceback."""


# ----------------------------------------------------------------------------- discovery

def _model_dirs() -> list[str]:
    dirs = []
    if os.environ.get("CLOZN_MODELS"):
        dirs += os.environ["CLOZN_MODELS"].split(os.pathsep)
    cfg = os.path.join(HOME, "config.json")
    if os.path.isfile(cfg):
        try:
            dirs += json.load(open(cfg)).get("model_dirs", [])
        except Exception:
            pass
    dirs += [os.path.join(HOME, "models"), os.path.join(REPO, "models"),
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
    if arg.lower().endswith(".gguf") and os.path.isfile(arg):
        return os.path.abspath(arg)
    models = _scan_models()
    if not models:
        raise CloznError("no GGUF models found. Put .gguf files in ~/.clozn/models or set CLOZN_MODELS=<dir>.")
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
        raise CloznError(f"'{arg}' is ambiguous: {', '.join(_friendly(h) for h in hits)}. Be more specific.")
    avail = ", ".join(sorted({_friendly(m) for m in models}))
    raise CloznError(f"model '{arg}' not found. Available: {avail}.")


def find_engine(prefer_gpu=True) -> tuple[str, list[str], bool]:
    """-> (exe_path, dll_dirs, is_gpu). Raises if no build exists."""
    cands = []
    for sub, gpu in BUILDS:
        root = os.path.join(ENGINE_CORE, sub)
        for exe in (os.path.join(root, "cloze-server.exe"),
                    os.path.join(root, "Release", "cloze-server.exe"),
                    os.path.join(root, "cloze-server")):       # posix
            if os.path.isfile(exe):
                bins = [d for d in (root, os.path.join(root, "Release"),
                                    os.path.join(root, "bin"), os.path.join(root, "bin", "Release"))
                        if os.path.isdir(d)]
                cands.append((exe, bins, gpu))
                break
    if not cands:
        raise CloznError("no engine found. Build it:  cd engine/core  then  build_gpu.bat (GPU) "
                         "or build_serve.bat (CPU).")
    cands.sort(key=lambda c: (0 if c[2] else 1) if prefer_gpu else (1 if c[2] else 0))
    return cands[0]


# ----------------------------------------------------------------------------- engine process

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _env_with_dlls(dll_dirs: list[str], gpu: bool) -> dict:
    env = dict(os.environ)
    extra = list(dll_dirs)
    if gpu:
        for c in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
                  r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"):
            if os.path.isdir(c):
                extra.append(c)
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _launch_args(exe: str, model: str, port: int, flags: dict, gpu: bool) -> list[str]:
    args = [exe, model, "--port", str(port), "--host", "127.0.0.1"]
    if gpu:
        args += ["--gpu-layers", "99"]
    if "mask" in flags:
        args += ["--mask-token", str(flags["mask"])]
    if "eos" in flags:
        args += ["--eos", str(flags["eos"])]
    if "sae" in flags:                        # passthrough only: dims must match, server refuses politely
        args += ["--sae", flags["sae"]]
        if "sae_k" in flags:
            args += ["--sae-k", str(flags["sae_k"])]
    return args


def _health(port: int, timeout=3.0):
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return json.loads(r.read())
    except Exception:
        return None


def spawn_engine(model: str, port: int, flags: dict, *, prefer_gpu=True, logf=None, boot_timeout=180):
    """Start an engine on `port`, wait until /health is ok. Returns (proc, health, is_gpu)."""
    exe, dll_dirs, gpu = find_engine(prefer_gpu)
    args = _launch_args(exe, model, port, flags, gpu)
    proc = subprocess.Popen(args, env=_env_with_dlls(dll_dirs, gpu),
                            stdout=logf or subprocess.DEVNULL, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < boot_timeout:
        if proc.poll() is not None:                            # died before healthy
            raise CloznError(f"engine exited (code {proc.returncode}). {_log_tail(logf)}")
        h = _health(port)
        if h and h.get("status") == "ok":
            return proc, h, gpu
        time.sleep(0.3)
    proc.terminate()
    raise CloznError(f"engine did not become healthy within {boot_timeout}s. {_log_tail(logf)}")


def _log_tail(logf, n=400):
    if not logf:
        return ""
    try:
        logf.flush()
        with open(logf.name, "r", errors="replace") as f:
            return "last output: " + f.read()[-n:].strip().replace("\n", " ")
    except Exception:
        return ""


# --------------------------------------------------------------- warm-daemon registry (clozn serve <-> run)
# `clozn serve` records {port -> model/gpu/mode} here; `clozn run` reuses a live one instead of reloading.
# Stale entries self-heal: a dead serve fails the /health check in _find_warm and is ignored (then pruned).

REG = os.path.join(HOME, "daemons.json")


def _reg_read() -> dict:
    try:
        return json.load(open(REG))
    except Exception:
        return {}


def _reg_write(d: dict):
    os.makedirs(HOME, exist_ok=True)
    try:
        json.dump(d, open(REG, "w"))
    except Exception:
        pass


def _register(model: str, port: int, gpu: bool, mode: str, pid: int):
    d = _reg_read(); d[str(port)] = {"model": model, "gpu": gpu, "mode": mode, "pid": pid}; _reg_write(d)


def _kill(pid: int):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, 15)
    except Exception:
        pass


def _unregister(port: int):
    d = _reg_read()
    if d.pop(str(port), None) is not None:
        _reg_write(d)


def _find_warm(model: str):
    """A live `clozn serve` for this exact model -> (port, gpu, mode), pruning dead entries. Else None."""
    d = _reg_read(); hit = None; dirty = False
    for port, ent in list(d.items()):
        h = _health(int(port), timeout=1.0)
        if not (h and h.get("status") == "ok"):
            d.pop(port, None); dirty = True; continue           # prune the dead
        if ent.get("model") == model and hit is None:
            hit = (int(port), bool(ent.get("gpu")), ent.get("mode", h.get("mode", "?")))
    if dirty:
        _reg_write(d)
    return hit


# ----------------------------------------------------------------------------- prompting

SYS = "You are a helpful assistant."


def _chat_session(history, new_user: str, family: str = "qwen") -> str:
    """Build the whole conversation in the family's chat template. BOS is left to the engine (add_bos).
    history: list of (user, assistant) pairs already exchanged; new_user: the current turn."""
    if family == "mistral":
        s = "".join(f"[INST] {u} [/INST] {a}</s>" for u, a in history)
        return s + f"[INST] {new_user} [/INST]"
    if family == "llama3":
        s = f"<|start_header_id|>system<|end_header_id|>\n\n{SYS}<|eot_id|>"
        for u, a in history:
            s += (f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
                  f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>")
        return s + (f"<|start_header_id|>user<|end_header_id|>\n\n{new_user}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    if family == "gemma":
        s = "".join(f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>\n"
                    for u, a in history)
        return s + f"<start_of_turn>user\n{new_user}<end_of_turn>\n<start_of_turn>model\n"
    s = f"<|im_start|>system\n{SYS}<|im_end|>\n"
    for u, a in history:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    return s + f"<|im_start|>user\n{new_user}<|im_end|>\n<|im_start|>assistant\n"


def _chat_wrap(prompt: str, family: str = "qwen") -> str:
    return _chat_session([], prompt, family)


def stream_ar(port: int, prompt: str, max_tokens: int, heat: bool = False):
    """POST /v1/completions (stream); print each committed token. -> (count, trace steps w/ conf + alts).

    heat=True paints each token as it lands by its confidence (the denoise heatmap, live); False (default)
    is the plain, byte-for-byte-unchanged stream. Painting also no-ops when color is off (piped/NO_COLOR),
    so `--heat | cat` is still clean text.

    The engine emits, per token, a `tokens_committed` frame (the chosen piece + its confidence) and a
    `step_lens` frame (the top-k it weighed). We pair them by position into a replayable trace -- the raw
    material for the timeline: where was it uncertain, and what did it almost say."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    n = 0
    frames = []                                             # every parsed SSE frame, for the shared accumulator
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            frames.append(obj)
            if obj.get("type") == "tokens_committed":       # print live as tokens land
                for it in obj.get("items", []):
                    sys.stdout.write(_stream_token(it.get("piece", ""), it.get("conf"), heat))
                    sys.stdout.flush()
                    n += 1
    # Accumulation (pair tokens_committed with step_lens by position) lives in runlog so the CLI and the
    # engine-chat capture share ONE tested implementation. Fall back to a local pairing if the import fails
    # -- the stdlib CLI must never break on a missing sibling.
    try:
        import clozn.runs.store as runlog
        steps = runlog.accumulate_ar_events(frames)
    except Exception:
        by_pos: dict = {}
        for obj in frames:
            if obj.get("type") == "tokens_committed":
                for it in obj.get("items", []):
                    by_pos[it.get("pos")] = {"pos": it.get("pos"), "index": it.get("pos"),
                                             "token_id": it.get("id"), "piece": it.get("piece", ""),
                                             "prob": round(float(it.get("conf", 0.0)), 3),
                                             "conf": round(float(it.get("conf", 0.0)), 3), "alts": []}
            elif obj.get("type") == "step_lens":
                step = by_pos.get((obj.get("positions") or [None])[0])
                if step:
                    pieces, probs = obj.get("pieces", []), obj.get("probs", [])
                    ids = obj.get("ids") or [None] * len(pieces)
                    chosen_id = step.get("token_id")
                    step["alts"] = [{"token_id": tid, "piece": p, "prob": round(float(pr), 3)}
                                    for tid, p, pr in zip(ids, pieces, probs)
                                    if (chosen_id is None or tid != chosen_id) and p != step["piece"]][:3]
        steps = [by_pos[p] for p in sorted(by_pos, key=lambda x: (x is None, x))]
    return n, steps


def complete_once(port: int, prompt: str, max_tokens: int) -> str:
    """Non-streaming /v1/completions (used for diffusion, which commits out of reading order)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read())
    return (r.get("choices") or [{}])[0].get("text", "")


# ----------------------------------------------------------------------------- trace (the debugger seam)
# Every run is persisted to ~/.clozn/runs through research/runlog.py; Studio and `clozn trace` read that
# same journal. AR runs also keep writing the older ~/.clozn/traces/<id>.json cache for branch/back-compat;
# `clozn trace --legacy-cache` can inspect it, but the shared runlog is the default source of truth.

def _runid() -> str:
    return f"{int(time.time())}-{os.getpid() % 1000:03d}"


def _save_trace(meta: dict, steps: list) -> str:
    d = os.path.join(HOME, "traces")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, meta["id"] + ".json")
    try:
        json.dump({"meta": meta, "steps": steps}, open(path, "w"))
    except Exception:
        return ""
    return path


def _confbar(c: float, width=8) -> str:
    full = int(round(max(0.0, min(1.0, c)) * width))
    return "█" * full + "░" * (width - full)


# --- confidence as color: a terminal heatmap over the tokens, echoing the denoise UI (brightness = --------
#     confidence). A warm pink flags where the model wavered; a cool blue where it was sure. The endpoints
#     are denoise.html's exact palette (#e07a96 / #c3cde0 / #7aa7ff), so the terminal reads like the web
#     "watch it denoise" board. Painting is a no-op when color is off (non-tty, NO_COLOR, or no ANSI), so
#     pipes/tests/plain terminals see clean text.
_C_HOT = (231, 90, 110)     # conf 0.0  -- most uncertain: a hot rose (denoise's "changed its mind" red-flash)
_C_PINK = (224, 122, 150)   # conf ~0.5 -- the hesitation line: warm pink  (#e07a96, denoise's uncertainty flag)
_C_PALE = (195, 205, 224)   # conf 0.5+ -- just cleared the bar: pale blue-gray (#c3cde0, denoise low end)
_C_BLUE = (122, 167, 255)   # conf 1.0  -- fully sure: vivid blue          (#7aa7ff, denoise high end)


def _lerp_rgb(a, b, t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _conf_rgb(c: float):
    """Confidence in [0,1] -> (r,g,b) on the denoise ramp: hot-rose(0) -> pink(0.5) | pale(0.5) -> blue(1).
    The step at the 0.5 hesitation threshold is deliberate -- it's the same line cmd_trace marks with '?', so
    'flagged uncertain' (warm) reads as a different family from 'confident' (cool), not a smooth blur."""
    c = _num(c)
    c = 0.0 if c < 0 else (1.0 if c > 1 else c)
    if c < 0.5:
        return _lerp_rgb(_C_HOT, _C_PINK, c / 0.5)
    return _lerp_rgb(_C_PALE, _C_BLUE, (c - 0.5) / 0.5)


def _paint(text: str, c: float) -> str:
    """Wrap `text` in a truecolor SGR for confidence `c`. No-op when COLOR is off, so every existing
    (color-off) test keeps passing byte-for-byte and piped output stays clean."""
    if not COLOR or not text:
        return text
    r, g, b = _conf_rgb(c)
    return f"\033[38;2;{r};{g};{b}m{text}{RST}"


def _stream_token(piece: str, conf, heat: bool) -> str:
    """The exact string written for one streamed token in `clozn run --heat`: painted by its confidence when
    heat is on (and color is available), the raw piece otherwise. Factored out so the live paint is
    unit-testable without a running engine -- heat off is byte-identical to the plain stream."""
    return _paint(piece, _num(conf)) if heat else piece


def _term_width(default=76) -> int:
    try:
        return max(30, min(shutil.get_terminal_size().columns - 2, 118))
    except Exception:
        return default


def _heatmap_lines(pieces_confs, width=None) -> list:
    """Reconstruct a reply from (piece, conf) pairs, each token painted by its confidence, wrapped to the
    terminal width. A newline inside a piece breaks the line (the model's own line breaks); wrapping counts
    VISIBLE characters only (SGR escapes are zero-width). Plain text when color is off."""
    width = width or _term_width()
    lines, line, col = [], "", 0
    for piece, conf in pieces_confs:
        for j, part in enumerate(str(piece if piece is not None else "").split("\n")):
            if j > 0:
                lines.append(line); line, col = "", 0
            if not part:
                continue
            if col > 0 and col + len(part) > width:
                lines.append(line); line, col = "", 0
                part = part.lstrip() or part
            line += _paint(part, conf)
            col += len(part)
    if line:
        lines.append(line)
    return lines


def _conf_legend() -> str:
    """One line explaining color = confidence, itself painted in the colors it names (echoes the denoise
    legend). Plain sentence when color is off."""
    if not COLOR:
        return f"{DIM}color = per-token confidence: low -> high; a warm flag marks where it wavered{RST}"
    ramp = _paint("low", 0.15) + _paint(" ->", 0.5) + _paint(" ->", 0.72) + _paint(" high", 0.98)
    return f"{DIM}color = per-token confidence{RST}  {ramp}   {_paint('wavered', 0.18)}"


# ----------------------------------------------------------------------------- commands

def cmd_models(_args):
    models = _scan_models()
    try:
        _, _, gpu = find_engine()
        eng = f"{BOLD}{'GPU' if gpu else 'CPU'} build{RST} found"
    except CloznError:
        eng = f"{BOLD}no engine built{RST} (run: cd engine/core && build_gpu.bat)"
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
    spec = args.model
    if spec in PULLABLE:
        repo, file = PULLABLE[spec]
    elif spec.endswith(".gguf") and spec.count("/") >= 2:
        parts = spec.split("/"); repo, file = "/".join(parts[:-1]), parts[-1]
    else:
        raise CloznError(f"don't know how to pull '{spec}'. Known: {', '.join(PULLABLE)}. "
                         f"Or give an explicit  owner/repo/file.gguf")
    dest_dir = os.path.join(HOME, "models"); os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, file)
    if os.path.isfile(dest):
        print(f"already have {file} ({os.path.getsize(dest) / 1e9:.1f}G)"); return
    url = f"https://huggingface.co/{repo}/resolve/main/{file}?download=true"
    print(f"{DIM}pulling{RST} {file}  {DIM}from {repo}{RST}", file=sys.stderr)
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
                        head = (f"{_confbar(done / total)} {done / total * 100:5.1f}%  "
                                f"{done / 1e9:.2f}/{total / 1e9:.2f} GB") if total else f"{done / 1e9:.2f} GB"
                        sys.stderr.write(f"\r  {head}  {sp:4.0f} MB/s   "); sys.stderr.flush()
        sys.stderr.write("\n")
        os.replace(tmp, dest)
    except urllib.error.HTTPError as e:
        _rm(tmp)
        raise CloznError(f"{repo}/{file} not found on HuggingFace (404)." if e.code == 404
                         else f"download failed (HTTP {e.code}).")
    except Exception as e:
        _rm(tmp)
        raise CloznError(f"download failed: {e}")
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
    verdict = f"{BOLD}FITS{RST}" if report["fits"] else f"{BOLD}WON'T FIT{RST}"
    lines = [
        f"{name} {quant} — {n_layers if n_layers is not None else '?'} layers, "
        f"{_fmt_ctx(header.get('context_length'))} ctx, {size_gb:.1f} GB file"
        + (f"  {DIM}{source_note}{RST}" if source_note else ""),
        f"on {vram_gb:g} GB VRAM: {verdict}  (~est {report['est_vram_gb']:.1f} GB at "
        f"{_fmt_ctx(ctx_for_estimate)} ctx, KV-q8)",
    ]
    if report.get("offload_hint"):
        lines.append(f"  {report['offload_hint']}")
    lines.append(f"{DIM}{report['note']}{RST}")
    if header.get("quant_source") and header["quant_source"] != "general.file_type":
        lines.append(f"{DIM}quant is a guess from the dominant tensor type ({header['quant_source']}).{RST}")
    return "\n".join(lines)


def cmd_plan(args):
    from clozn import fit_planner   # stdlib+urllib only; imported lazily like the other clozn.* siblings

    vram_gb = args.vram if args.vram is not None else (_detect_vram_gb() or 16.0)
    spec = args.model

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
            except CloznError:
                if spec not in PULLABLE:
                    raise
                repo, file = PULLABLE[spec]
                url = f"https://huggingface.co/{repo}/resolve/main/{file}"
                print(f"{DIM}- '{spec}' isn't downloaded yet -- reading its header straight off "
                     f"HuggingFace (no download){RST}", file=sys.stderr)
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

    report = fit_planner.fit_report(header, size, vram_gb)
    print(format_plan(name, header, size, report, vram_gb, source_note=source_note))


def _open_browser(url):
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def _studio_health(port):
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/substrate", timeout=3)
        return json.loads(r.read())
    except Exception:
        return None


def _studio_python():
    """The Studio backend needs PyTorch (HF model + SAE), so it can't use the stdlib CLI python. Prefer
    $CLOZN_STUDIO_PYTHON, then ~/.clozn/config.json['studio_python'], then this process's own python."""
    if os.environ.get("CLOZN_STUDIO_PYTHON"):
        return os.environ["CLOZN_STUDIO_PYTHON"]
    cfg = os.path.join(HOME, "config.json")
    if os.path.isfile(cfg):
        try:
            sp = json.load(open(cfg)).get("studio_python")
            if sp:
                return sp
        except Exception:
            pass
    return sys.executable


def _studio_banner(base, health):
    print(f"  Studio UI:        {BOLD}{base}/studio.html{RST}")
    print(f"  OpenAI endpoint:  {BOLD}{base}/v1{RST}   (point Open WebUI / Cursor / any client here)")
    print(f"  Lab / brain viz:  {base}/brain.html")
    if health:
        print(f"  Substrate:        {health.get('active')}   (available: {', '.join(health.get('available', []))})")


def cmd_studio(args):
    """Launch Clozn Studio -- the glass-box UI + the local OpenAI endpoint your other tools connect to."""
    port = args.port or 8090
    base = f"http://127.0.0.1:{port}"
    h = _studio_health(port)
    if h:                                                  # already up -> just point at it
        print(f"{BOLD}Clozn Studio{RST} already running:")
        _studio_banner(base, h)
        if args.open:
            _open_browser(f"{base}/studio.html")
        return
    server = os.path.join(REPO, "clozn", "clozn_server.py")
    if not os.path.isfile(server):
        raise CloznError(f"studio backend not found at {server}")
    cmd = [_studio_python(), "-m", "clozn.clozn_server", "--port", str(port)]
    if args.substrate:
        cmd += ["--substrate", args.substrate]
    os.makedirs(HOME, exist_ok=True)
    logf = open(os.path.join(HOME, "studio.log"), "w")
    print(f"{DIM}- starting Clozn Studio (loading the model, ~30s) …{RST}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            tail = _log_tail(logf); logf.close()
            raise CloznError(f"studio exited (code {proc.returncode}). {tail} "
                             "(the backend needs a PyTorch python -- set CLOZN_STUDIO_PYTHON if import failed)")
        h = _studio_health(port)
        if h:
            break
        time.sleep(0.5)
    else:
        proc.terminate(); logf.close()
        raise CloznError("studio did not come up within 300s.")
    print(f"  {BOLD}Clozn Studio{RST} ready in {time.time()-t0:.0f}s\n")
    _studio_banner(base, h)
    if args.open:
        _open_browser(f"{base}/studio.html")
    print(f"\n  {DIM}your local model now runs under Clozn -- point any client at {base}/v1   -   Ctrl-C to stop{RST}\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n{DIM}- stopping studio{RST}", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
    finally:
        logf.close()


def _run_turn(port, mode, text, max_tokens, gpu, model_name, prompt_for_trace, heat=False):
    """One generation: stream (AR, auto-saving the trace) or denoise (diffusion). Prints stats. -> response."""
    g0 = time.time()
    steps = []
    if mode == "autoregressive":
        n, steps = stream_ar(port, text, max_tokens, heat=heat)
        sys.stdout.write("\n")
        if heat and COLOR:                                 # a legend + how many tokens wavered, after the reply
            lows = sum(1 for s in steps if _num(s.get("conf")) < 0.5)
            print(f"{_conf_legend()}   {DIM}{lows} wavered{RST}", file=sys.stderr)
        _save_trace({"id": _runid(), "model": model_name, "prompt": prompt_for_trace,
                     "backend": "GPU" if gpu else "CPU", "mode": mode, "n": n, "t": time.time()}, steps)
        resp = "".join(s["piece"] for s in steps).strip()
    else:
        resp = complete_once(port, text, max_tokens).strip()
        print(resp); n = len(resp.split())
    dt = time.time() - g0
    rate = f", ~{n/dt:.0f} tok/s" if dt > 0 and mode == "autoregressive" else ""
    print(f"{DIM}- {n} tok in {dt:.1f}s{rate} - {'GPU' if gpu else 'CPU'}{RST}", file=sys.stderr)
    _log_run_cli(model_name, prompt_for_trace, resp, steps, g0)   # every CLI turn becomes an inspectable run
    return resp


def _log_run_cli(model_name, prompt, resp, steps, started):
    """Write this CLI turn to the Run Log so `clozn run`/REPL turns show up in the Studio alongside chats.
    runlog.py lives in research/ (a sibling of this stdlib-only CLI) and is itself stdlib-only, so we add
    its dir to sys.path and import it. Logging must NEVER break a run -- swallow everything."""
    try:
        import clozn.runs.store as runlog
        # stream_ar hands us per-token steps ({piece, conf, alts}); runlog owns the steps->trace mapping so
        # the on-disk trace schema stays one contract shared with the engine-chat capture (issue B3).
        trace = runlog.steps_to_trace(steps)
        runlog.record(source="cli", client="cli", model=model_name, substrate="engine",
                      messages=[{"role": "user", "content": prompt}], response=resp,
                      trace=trace, started=started)
    except Exception:
        pass


def _repl(port, mode, flags, fam, gpu, model, max_tokens, heat=False):
    """Interactive chat loop on a warm engine (Ollama-style). /reset clears, /bye quits."""
    name = _friendly(model)
    is_chat = flags.get("chat") and mode == "autoregressive"
    tty = sys.stdin.isatty()
    print(f"{DIM}chat with {name}  -  /reset clears history, /bye quits{RST}", file=sys.stderr)
    history = []
    while True:
        if tty:
            try:
                msg = input("\nyou> ").strip()
            except EOFError:
                break
        else:
            line = sys.stdin.readline()
            if not line:
                break
            msg = line.strip()
        if not msg:
            continue
        if msg in ("/bye", "/exit", "/quit"):
            break
        if msg == "/reset":
            history = []; print(f"{DIM}(history cleared){RST}", file=sys.stderr); continue
        text = (_chat_session(history, msg, fam) if is_chat
                else "".join(f"{u}\n{a}\n" for u, a in history) + msg)
        sys.stdout.write(f"{BOLD}{name}>{RST} ")
        resp = _run_turn(port, mode, text, max_tokens, gpu, name, msg, heat=heat)
        history.append((msg, resp))
    print(f"{DIM}bye{RST}", file=sys.stderr)


def cmd_run(args):
    model = resolve_model(args.model)
    flags = _flags_for(model)
    if args.mask is not None:
        flags["mask"] = args.mask
    if args.eos is not None:
        flags["eos"] = args.eos
    fam = flags.get("tmpl", "qwen")
    warm = None if args.cpu else _find_warm(model)     # reuse a live `clozn serve` instead of reloading
    proc = logf = None
    if warm:
        port, gpu, mode = warm
        print(f"{DIM}- {_friendly(model)} warm on port {port} ({'GPU' if gpu else 'CPU'}, {mode}){RST}",
              file=sys.stderr, flush=True)
    else:
        os.makedirs(HOME, exist_ok=True)
        logf = open(os.path.join(HOME, "engine-run.log"), "w")
        port = args.port or _free_port()
        print(f"{DIM}- loading {_friendly(model)} …{RST}", file=sys.stderr, flush=True)
        t0 = time.time()
        proc, health, gpu = spawn_engine(model, port, flags, prefer_gpu=not args.cpu, logf=logf)
        mode = health.get("mode", "?")
        print(f"{DIM}- {_friendly(model)} on {'GPU' if gpu else 'CPU'} build ({mode}), "
              f"ready in {time.time()-t0:.1f}s{RST}", file=sys.stderr, flush=True)
    try:
        if args.prompt is not None:                            # one-shot
            is_chat = flags.get("chat") and mode == "autoregressive"
            text = _chat_wrap(args.prompt, fam) if is_chat else args.prompt
            _run_turn(port, mode, text, args.max, gpu, _friendly(model), args.prompt, heat=args.heat)
            if mode == "autoregressive":
                print(f"{DIM}  clozn trace   inspect where it was uncertain + what it almost said{RST}",
                      file=sys.stderr)
        else:                                                  # interactive REPL (Ollama-style)
            _repl(port, mode, flags, fam, gpu, model, args.max, heat=args.heat)
    finally:
        if proc:                                       # only tear down an engine WE spawned; leave warm ones up
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if logf:
            logf.close()


def cmd_serve(args):
    model = resolve_model(args.model)
    flags = _flags_for(model)
    if args.mask is not None:
        flags["mask"] = args.mask
    if args.eos is not None:
        flags["eos"] = args.eos
    if args.sae is not None:
        flags["sae"] = args.sae
        if args.sae_k is not None:
            flags["sae_k"] = args.sae_k
    port = args.port or 8080
    if _health(port):
        raise CloznError(f"port {port} already serving something. Pick another with --port.")
    print(f"{DIM}- loading {_friendly(model)} …{RST}", file=sys.stderr, flush=True)
    t0 = time.time()
    proc, health, gpu = spawn_engine(model, port, flags, prefer_gpu=not args.cpu, logf=None)
    base = f"http://127.0.0.1:{port}"
    print(f"\n  {BOLD}{_friendly(model)}{RST} ready on {'GPU' if gpu else 'CPU'} build "
          f"({health.get('mode')}) in {time.time()-t0:.1f}s")
    print(f"  OpenAI-compatible endpoint:  {BOLD}{base}/v1{RST}")
    print(f"  text completions:            POST {base}/v1/completions")
    print(f"  live viz / health:           {base}/   -   {base}/health")
    print(f"\n  {DIM}point any OpenAI client at {base}/v1  -  `clozn run {_friendly(model)} ...` reuses this  -  Ctrl-C to stop{RST}\n")
    _register(model, port, gpu, health.get("mode", "?"), proc.pid)   # so `clozn run`/`ps`/`stop` can find it
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n{DIM}- stopping{RST}", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    finally:
        _unregister(port)


def cmd_ps(_args):
    d = _reg_read(); live = []; dirty = False
    for port, ent in list(d.items()):
        h = _health(int(port), timeout=1.0)
        if h and h.get("status") == "ok":
            live.append((port, ent))
        else:
            d.pop(port, None); dirty = True
    if dirty:
        _reg_write(d)
    if not live:
        print("no clozn serve daemons running."); return
    print(f"{'MODEL':<14} {'PORT':>6}  {'BACKEND':<8} MODE")
    for port, ent in live:
        print(f"{_friendly(ent.get('model', '?')):<14} {port:>6}  "
              f"{('GPU' if ent.get('gpu') else 'CPU'):<8} {ent.get('mode', '?')}")


def cmd_stop(args):
    d = _reg_read()
    targets = [(port, ent) for port, ent in d.items()
               if args.which in ("all", str(port)) or _friendly(ent.get("model", "")) == args.which]
    if not targets:
        raise CloznError(f"no running daemon matches '{args.which}'. See: clozn ps")
    for port, ent in targets:
        if ent.get("pid"):
            _kill(int(ent["pid"]))
        d.pop(port, None)
        print(f"stopped {_friendly(ent.get('model', '?'))} on port {port}")
    _reg_write(d)


def _trace_cache_files() -> list[str]:
    d = os.path.join(HOME, "traces")
    return sorted(glob.glob(os.path.join(d, "*.json")))


def _render_trace(meta: dict, steps: list):
    m = meta or {}
    steps = [s for s in (steps or []) if isinstance(s, dict)]
    print(f"{BOLD}{m.get('model', '?')}{RST}  \"{m.get('prompt', '')[:64]}\"")
    print(f"{DIM}{m.get('n', len(steps))} tokens - {m.get('backend', '?')} - short bar = less sure; "
          f"'almost' = what it nearly said{RST}")
    if not steps:
        print("-" * 62)
        print(f"{DIM}no per-token trace recorded on this run{RST}")
        return
    # the reply reconstructed from the trace, each token painted by its confidence -- the denoise board, in
    # the terminal. Then the per-token detail below (piece also painted, so the two views share one palette).
    hm = _heatmap_lines([(s.get("piece", s.get("text", "")),
                          _num(s.get("prob", s.get("conf", s.get("confidence"))))) for s in steps])
    if hm:
        print("-" * 62)
        for ln in hm:
            print(ln)
        print(_conf_legend())
    print("-" * 62)
    for i, s in enumerate(steps):
        piece = (s.get("piece", s.get("text", "")) or "").replace("\n", "\\n").replace("\t", "\\t")
        conf = _num(s.get("prob", s.get("conf", s.get("confidence"))))
        mark = " " if conf >= 0.5 else "?"
        shown = piece[:16]
        cell = _paint(shown, conf) + " " * max(0, 16 - len(shown))
        idx = s.get("index", s.get("pos", i))
        meta = []
        if s.get("token_id") is not None:
            meta.append(f"id {s.get('token_id')}")
        if s.get("logprob") is not None:                    # derived: log(confidence), never a separate signal
            meta.append(f"logp {_num(s.get('logprob')):.3f}")
        if s.get("entropy") is not None:                    # true full-distribution entropy (HF/Qwen path only)
            meta.append(f"H {_num(s.get('entropy')):.3f}")
        if s.get("topk_entropy") is not None:                # TOP-K APPROXIMATION only (engine path) -- say so
            meta.append(f"H@k(approx) {_num(s.get('topk_entropy')):.3f}")
        line = f" {mark} {str(idx):>3} {cell} {_confbar(conf)} {conf:.2f}"
        if meta:
            line += f"   {DIM}{' '.join(meta)}{RST}"
        if conf < 0.5 and s.get("alts"):
            alts = "  ".join(f"{(a.get('piece', a.get('text', '')) or '').strip() or '_'} "
                              f"{_num(a.get('prob')):.2f}" +
                              (f" id {a.get('token_id')}" if a.get("token_id") is not None else "")
                              for a in s["alts"][:3] if isinstance(a, dict))
            if alts:
                line += f"   {DIM}almost: {alts}{RST}"
        print(line)
    lows = [s for s in steps if _num(s.get("prob", s.get("conf", s.get("confidence"))), 1.0) < 0.5]
    print("-" * 62)
    tail = " -> " + ", ".join((s.get("piece", s.get("text", "")) or "").strip() for s in lows[:6]) if lows else ""
    print(f"{DIM}{len(lows)} uncertain moment(s){tail}{RST}")


def _cmd_trace_legacy(args):
    files = _trace_cache_files()
    if not files:
        print('no legacy trace cache entries yet -- run something first:  clozn run qwen "..."'); return
    if args.list:
        print(f"{'WHEN':<18} {'MODEL':<11} {'TOK':>4}  PROMPT")
        for f in files[-12:]:
            m = json.load(open(f)).get("meta", {})
            print(f"{m.get('id', ''):<18} {m.get('model', ''):<11} {m.get('n', 0):>4}  {m.get('prompt', '')[:46]}")
        return
    tr = json.load(open(files[-1]))
    m, steps = tr.get("meta", {}), tr.get("steps", [])
    _render_trace(m, steps)


def _import_runlog():
    import clozn.runs.store as runlog
    return runlog


def _run_prompt(run: dict) -> str:
    for msg in reversed(run.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(run.get("prompt_summary", ""))


def _runlog_trace_steps(run: dict) -> list[dict]:
    trace = run.get("trace") if isinstance(run, dict) else {}
    trace = trace if isinstance(trace, dict) else {}
    rich = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    if rich:
        out = []
        for i, s in enumerate(rich):
            if not isinstance(s, dict):
                continue
            piece = s.get("piece", s.get("text", ""))
            conf = s.get("prob", s.get("conf", s.get("confidence")))
            alts = s.get("alts", s.get("alternatives", []))
            step = {"index": s.get("index", s.get("pos", i)), "piece": str(piece),
                    "conf": _num(conf, 1.0), "alts": alts if isinstance(alts, list) else []}
            for k in ("token_id", "logprob", "entropy", "topk_entropy", "wall_ms", "dt_ms"):
                if s.get(k) is not None:
                    step[k] = s.get(k)
            out.append(step)
        return out
    # legacy v1-array reconstruction (no `steps` on this trace) -- also folds in the v2 parallel arrays
    # (token_ids/logprobs/topk_entropy) when a trace happens to carry them without a `steps` list; absent
    # on any trace persisted before this change, so every lookup below is a guarded .get()/length check.
    tokens = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    confidence = trace.get("confidence") if isinstance(trace.get("confidence"), list) else []
    alternatives = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    logprobs = trace.get("logprobs") if isinstance(trace.get("logprobs"), list) else []
    topk_entropy = trace.get("topk_entropy") if isinstance(trace.get("topk_entropy"), list) else []
    out = []
    for i, piece in enumerate(tokens):
        alts = alternatives[i] if i < len(alternatives) and isinstance(alternatives[i], list) else []
        step = {"pos": i, "piece": str(piece),
                "conf": _num(confidence[i], 1.0) if i < len(confidence) else 1.0,
                "alts": alts}
        if i < len(token_ids) and token_ids[i] is not None:
            step["token_id"] = token_ids[i]
        if i < len(logprobs) and logprobs[i] is not None:
            step["logprob"] = logprobs[i]
        if i < len(topk_entropy) and topk_entropy[i] is not None:
            step["topk_entropy"] = topk_entropy[i]
        out.append(step)
    return out


def _runlog_trace_meta(run: dict, steps: list[dict]) -> dict:
    backend = run.get("substrate") or run.get("source") or run.get("client") or "runlog"
    return {"id": run.get("id", ""), "model": run.get("model", ""), "prompt": _run_prompt(run),
            "backend": backend, "n": len(steps), "t": run.get("created_ts")}


def _list_runlog_traces(runlog, limit=12):
    rows = runlog.list_runs(limit=limit)
    print(f"{'WHEN':<19} {'MODEL':<11} {'TOK':>4}  PROMPT")
    for row in reversed(rows):
        run = runlog.get_run(row.get("id", "")) or {}
        trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
        toks = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
        when = str(row.get("created_at") or row.get("id", ""))[:19]
        print(f"{when:<19} {str(row.get('model', ''))[:11]:<11} {len(toks):>4}  "
              f"{str(row.get('prompt_summary', ''))[:46]}")


def cmd_trace(args):
    if getattr(args, "legacy_cache", False):
        return _cmd_trace_legacy(args)
    try:
        runlog = _import_runlog()
        rows = runlog.list_runs(limit=12 if args.list else 1)
    except Exception as e:
        raise CloznError(f"could not read the run journal (~/.clozn/runs): {e}")
    if not rows:
        hint = ""
        if _trace_cache_files():
            hint = "  Legacy trace cache entries exist; use: clozn trace --legacy-cache"
        print(f'no runs yet -- run something first:  clozn run qwen "..."{hint}')
        return
    if args.list:
        _list_runlog_traces(runlog, limit=12)
        return
    run = runlog.get_run(rows[0].get("id", "")) or {}
    if not run:
        raise CloznError("latest run disappeared from the run journal")
    steps = _runlog_trace_steps(run)
    _render_trace(_runlog_trace_meta(run, steps), steps)


def cmd_branch(args):
    """Take the road not taken: re-run from an uncertain point with the alternative the model nearly chose.

    Text-level (re-runs prompt + kept tokens + the alt through /v1/completions), so token boundaries can
    shift a hair -- but it shows, concretely, 'what if it had said X instead'. The seed of branch-a-bad-answer."""
    files = sorted(glob.glob(os.path.join(HOME, "traces", "*.json")))
    if not files:
        raise CloznError('no trace yet -- run something first:  clozn run qwen "..."')
    tr = json.load(open(files[-1])); meta = tr.get("meta", {})
    steps = [s for s in tr.get("steps", []) if (s.get("piece", s.get("text", "")) or "").strip()]   # branch on real tokens
    if not steps:
        raise CloznError("the last trace has no branchable tokens.")
    idx = (max(0, min(args.at, len(steps) - 1)) if args.at is not None
           else min(range(len(steps)),
                    key=lambda i: _num(steps[i].get("prob", steps[i].get("conf", steps[i].get("confidence"))), 1.0)))
    step = steps[idx]; alts = step.get("alts", [])
    if not alts:
        raise CloznError(f"no recorded alternative at '{step.get('piece', step.get('text', '')).strip()}' "
                         f"(conf {_num(step.get('prob', step.get('conf', step.get('confidence')))):.2f}).")
    alt = alts[max(0, min(args.pick, len(alts) - 1))]
    model = resolve_model(meta.get("model", "")); flags = _flags_for(model)
    head = _chat_wrap(meta.get("prompt", "")) if flags.get("chat") else meta.get("prompt", "")
    kept = "".join(s.get("piece", s.get("text", "")) for s in steps[:idx])
    alt_piece = alt.get("piece", alt.get("text", ""))
    prefix = head + kept + alt_piece
    print(f"{BOLD}branch{RST} of \"{meta.get('prompt', '')[:54]}\"")
    print(f"  fork at token {idx}: it chose {BOLD}{step.get('piece', step.get('text', '')).strip()!r}{RST} "
          f"({_num(step.get('prob', step.get('conf', step.get('confidence')))):.2f})"
          f"  ->  branch on {BOLD}{alt_piece.strip()!r}{RST} ({_num(alt.get('prob')):.2f})")
    print(f"  {DIM}original:{RST} {''.join(s.get('piece', s.get('text', '')) for s in steps).strip()[:130]}")
    warm = _find_warm(model); proc = logf = None
    if warm:
        port = warm[0]
    else:
        os.makedirs(HOME, exist_ok=True); logf = open(os.path.join(HOME, "engine-run.log"), "w")
        port = _free_port()
        print(f"{DIM}  - loading {_friendly(model)} …{RST}", file=sys.stderr, flush=True)
        proc, _h, _g = spawn_engine(model, port, flags, prefer_gpu=not args.cpu, logf=logf)
    try:
        cont = complete_once(port, prefix, args.max).strip()
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if logf:
            logf.close()
    print(f"  {BOLD}branch:{RST}   {(kept + alt['piece'] + cont).strip()[:160]}")


# ----------------------------------------------------------------------------- explain (M5 display; M1 assembles)
# `clozn explain <run_id>` (EXPLAIN_THIS_ANSWER_SPEC.md) renders the Studio's already-shipped, zero-generation
# /runs/<id>/explain (research/explain.py) as a terminal view: the confidence hesitations, the influences that
# were active, and the concepts note. DISPLAY ONLY by default -- this command generates nothing; it POSTs to
# the endpoint (the same "any client" bridge the Run Inspector's own Explain tab uses) and renders whatever
# comes back. The one opt-in exception is `--why` (below): it additionally POSTs to /runs/<id>/narrate, which
# DOES generate (M4's accountable-self narration, two model calls) -- opt-in for exactly that reason.
#
# format_explain() is factored out as a pure function (JSON in, text out) specifically so it's testable with a
# canned /explain dict -- no server, no model, no GPU -- mirroring cmd_trace's confidence-bar language.

_SPARK = "▁▂▃▄▅▆▇█"      # 8 heights for a compact per-token confidence shape (never a synthesized aggregate)
_SPARK_MAX = 400          # a defensive cap so a pathologically long run can't blow up one decorative line


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _num(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _spark_char(c: float) -> str:
    c = max(0.0, min(1.0, c))
    return _SPARK[min(len(_SPARK) - 1, int(c * len(_SPARK)))]


def _sparkline(n_tokens, moments) -> str:
    """One glyph per token position -- height-scaled at the recorded confidence for a hesitation; full
    height elsewhere (all that's known there is 'not flagged as a hesitation', i.e. it either cleared the
    threshold or had no confidence recorded at all -- a uniform glyph reports exactly that, not a fabricated
    precise value). A shape of where the model wavered across the reply, never a synthesized score."""
    by_idx = {}
    for m in _as_list(moments):
        if not isinstance(m, dict):
            continue
        i = m.get("index")
        if isinstance(i, int):
            by_idx[i] = _num(m.get("confidence"))
    n = int(n_tokens) if isinstance(n_tokens, (int, float)) else (max(by_idx) + 1 if by_idx else 0)
    n = max(0, min(n, _SPARK_MAX))
    return "".join(_spark_char(by_idx[i]) if i in by_idx else _SPARK[-1] for i in range(n))


def _paint_sparkline(n_tokens, moments) -> str:
    """`_sparkline`, but each glyph painted by its confidence (denoise-style) -- a mostly-cool bar with warm
    dips exactly where the model wavered. Byte-identical to `_sparkline` when color is off (tests unaffected)."""
    if not COLOR:
        return _sparkline(n_tokens, moments)
    by_idx = {}
    for m in _as_list(moments):
        if isinstance(m, dict) and isinstance(m.get("index"), int):
            by_idx[m["index"]] = _num(m.get("confidence"))
    n = int(n_tokens) if isinstance(n_tokens, (int, float)) else (max(by_idx) + 1 if by_idx else 0)
    n = max(0, min(n, _SPARK_MAX))
    return "".join(_paint(_spark_char(by_idx[i]), by_idx[i]) if i in by_idx else _paint(_SPARK[-1], 1.0)
                   for i in range(n))


def _verified_tag(v) -> str:
    """causal_verified -> a label that never overclaims. M1 (this command's only data source) tags every
    influence None ("active, not proven" -- the spec's own wording); True/False can only ever come from
    M2's on-demand ablation receipt, once it exists -- handled here so the CLI needs no change that day."""
    return "proven" if v is True else "ruled out" if v is False else "was active"


def _format_confidence(conf: dict) -> list[str]:
    out = [f"{BOLD}confidence{RST}  {DIM}measured per token -- never an overall score{RST}"]
    if not conf.get("available"):
        out.append(f"  {DIM}not available -- {conf.get('note', 'no trace on this run')}{RST}")
        return out
    moments = [m for m in _as_list(conf.get("uncertain_moments")) if isinstance(m, dict)]
    spark = _paint_sparkline(conf.get("n_tokens"), moments)
    if spark:
        out.append(f"  {spark}")
        if COLOR:
            out.append(f"  {_conf_legend()}")
    out.append(f"  {DIM}{conf.get('summary', '')} of {conf.get('n_tokens', 0)} tokens"
               f" (threshold {conf.get('threshold')}){RST}")
    for m in moments:
        piece = str(m.get("token") or "").replace("\n", "\\n").replace("\t", "\\t")
        c = _num(m.get("confidence"))
        shown = piece[:16]
        cell = _paint(shown, c) + " " * max(0, 16 - len(shown))
        line = f"   ? {cell} {_confbar(c)} {c:.2f}"
        alts = [a for a in _as_list(m.get("alternatives")) if isinstance(a, dict)]
        if alts:
            altxt = "  ".join(f"{(a.get('piece') or '').strip() or '_'} {_num(a.get('prob')):.2f}" for a in alts[:3])
            line += f"   {DIM}almost: {altxt}{RST}"
        out.append(line)
    return out


def _format_influences(inf: dict) -> list[str]:
    out = [f"{BOLD}influences active{RST}  {DIM}active this turn -- not yet proven causal{RST}"]
    gate, mode = inf.get("gate"), inf.get("mode")
    if gate is not None or mode:
        gate_s = f"{gate:.2f}" if isinstance(gate, (int, float)) else str(gate)
        out.append(f"  {DIM}gate {gate_s}{(' · ' + str(mode)) if mode else ''}{RST}")
    cards = [c for c in _as_list(inf.get("cards")) if isinstance(c, dict)]
    dials = [d for d in _as_list(inf.get("dials")) if isinstance(d, dict)]
    if cards:
        for c in cards:
            out.append(f"  [{_verified_tag(c.get('causal_verified'))}] {c.get('text', '')}")
            quote = c.get("quoted_span")
            if quote:
                out.append(f"      {DIM}“{quote}”{RST}")
            elif c.get("note"):
                out.append(f"      {DIM}{c['note']}{RST}")
    else:
        out.append(f"  {DIM}{inf.get('note', 'no memory applied')}{RST}")
    if dials:
        for d in dials:
            val = d.get("value")
            val_s = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
            out.append(f"  [{_verified_tag(d.get('causal_verified'))}] dial {d.get('name')} = {val_s}")
    else:
        out.append(f"  {DIM}no dials active{RST}")
    return out


def _format_concepts(conc: dict) -> list[str]:
    out = [f"{BOLD}concepts{RST}"]
    if not conc.get("available"):
        out.append(f"  {DIM}not available -- {conc.get('note', 'concept readout needs the engine')}{RST}")
        return out
    spans = [s for s in _as_list(conc.get("spans")) if isinstance(s, dict)]
    if not spans:
        out.append(f"  {DIM}(no spans recorded){RST}")
        return out
    for span in spans:
        piece = span.get("piece")
        head = f"  {piece!r} " if piece is not None else "  "
        feats = [f for f in _as_list(span.get("features")) if isinstance(f, dict)]
        feat_s = ", ".join(f"{f.get('label') or f.get('id') or '?'} {_num(f.get('score')):.2f}" for f in feats)
        out.append(f"{head}{feat_s}")
    return out


def format_explain(expl: dict) -> str:
    """The M1 explanation object (POST /runs/<id>/explain's JSON body) -> the terminal render. Pure: no
    I/O, no server, no model -- a canned fixture dict renders identically to a live response, which is
    exactly what makes this testable without either. Mirrors cmd_trace's confidence-bar language.

    Honesty is enforced HERE too, not just trusted from the server: never synthesizes an aggregate
    confidence number (only the per-hesitation values/bars explain.py already measured, or plain counts);
    an {"available": false, "note": ...} panel always prints its note (never silently skipped); every
    influence is labeled "was active", never "caused" (see _verified_tag). Never raises: a malformed panel
    degrades to a one-line notice instead of losing the ones that DID render, same discipline as
    research/explain.py's own explain()."""
    expl = expl if isinstance(expl, dict) else {}
    lines = [f"{BOLD}explain{RST}  run {expl.get('run_id') or '?'}", "-" * 62]
    try:
        lines += _format_confidence(_as_dict(expl.get("confidence")))
    except Exception:
        lines += [f"{BOLD}confidence{RST}", f"  {DIM}couldn't render this panel{RST}"]
    lines.append("")
    try:
        lines += _format_influences(_as_dict(expl.get("influences_active")))
    except Exception:
        lines += [f"{BOLD}influences active{RST}", f"  {DIM}couldn't render this panel{RST}"]
    lines.append("")
    try:
        lines += _format_concepts(_as_dict(expl.get("concepts")))
    except Exception:
        lines += [f"{BOLD}concepts{RST}", f"  {DIM}couldn't render this panel{RST}"]
    lines.append("-" * 62)
    return "\n".join(lines)


def _last_run_id():
    """The most recent run in the shared Studio run log (~/.clozn/runs), read directly -- mirrors
    _log_run_cli's own direct `import runlog` (research/ is a stdlib-only sibling). `clozn run`/`serve`
    write every turn straight into this log whether or not the Studio HTTP server is up, so --last
    resolves even while it's down; only the actual /explain fetch below needs the server."""
    try:
        import clozn.runs.store as runlog
        runs = runlog.list_runs(limit=1)
        return runs[0]["id"] if runs else None
    except Exception:
        return None


def _fetch_explain(port: int, run_id: str) -> dict:
    """POST /runs/<id>/explain on the Studio backend -- M1's assembly (research/explain.py), zero
    generation. A clean CloznError (one line, no traceback) when the Studio isn't up or the run doesn't
    resolve, matching the rest of this CLI's error style."""
    url = f"http://127.0.0.1:{port}/runs/{run_id}/explain"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise CloznError(f"explain failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise CloznError(f"couldn't reach the Studio on port {port} ({e.reason}). Start it first:  clozn studio")
    except Exception as e:
        raise CloznError(f"explain failed: {e}")


# --------------------------------------------------------------------- narrate (M4 display; narrate.py assembles)
# `clozn explain --why` additionally renders the Studio's POST /runs/<id>/narrate (research/narrate.py): the
# accountable-self narration -- a receipt-CONSTRAINED "why" diffed against an independent judge, with every
# unsupported claim it catches shown as a warning. Opt-in (--why), unlike the rest of `explain`, because this
# one GENERATES: two model calls (the constrained narration, and the unconstrained confabulation sample it is
# diffed against -- the latter is never returned by the endpoint at all, per narrate.py's trap guard, so there
# is nothing here that could render it even by accident).
#
# format_narrate() is factored out as a pure function (JSON in, text out), exactly like format_explain(), so
# it is testable with a canned /narrate dict -- no server, no model, no GPU.

def _fetch_narrate(port: int, run_id: str) -> dict:
    """POST /runs/<id>/narrate on the Studio backend -- M4's accountable-self narration (research/narrate.py).
    Unlike _fetch_explain (M1, free), this generates -- two model calls -- so it gets a longer timeout. A
    clean CloznError (one line, no traceback) when the Studio isn't up, the run doesn't resolve, or the qwen
    substrate isn't loaded (503), matching _fetch_explain's error style exactly."""
    url = f"http://127.0.0.1:{port}/runs/{run_id}/narrate"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise CloznError(f"narrate failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise CloznError(f"couldn't reach the Studio on port {port} ({e.reason}). Start it first:  clozn studio")
    except Exception as e:
        raise CloznError(f"narrate failed: {e}")


def _format_narration(cn: dict) -> list[str]:
    out = [f"{BOLD}why did it say this?{RST}  {DIM}receipt-constrained -- never the raw self-report (M4){RST}"]
    narration = cn.get("narration")
    narration = narration.strip() if isinstance(narration, str) else ""
    if narration:
        out.append(f"  {narration}")
    else:
        out.append(f"  {DIM}no receipt-backed narration was produced for this reply -- with a thin or empty "
                   f"record, that's a complete and honest answer, not a failure.{RST}")
    return out


def _format_flags(flags: list) -> list[str]:
    flags = [f for f in flags if isinstance(f, str) and f]
    out = [f"{BOLD}caught in the diff{RST}  {DIM}claimed with no receipt to back it{RST}"]
    if not flags:
        out.append(f"  {DIM}no unsupported claims flagged this time.{RST}")
        return out
    for f in flags:
        # reuse the file's own warm/confidence palette (the "wavered" end of the denoise ramp) for each
        # flag -- a flagged claim IS the same kind of thing a low-confidence token is: a place the honest
        # record does not back up. No-op (plain text) when COLOR is off, exactly like every other _paint call.
        out.append(f"  {_paint('⚠ ' + f, 0.0)}")
    return out


def format_narrate(obj: dict) -> str:
    """The M4 /narrate response object (POST /runs/<id>/narrate's JSON body -- exactly narrate.narrate()'s
    four keys: constrained_narration, flags, unsupported_claims, note) -> the terminal render. Pure: no I/O,
    no server, no model -- mirrors format_explain()'s contract exactly, so a canned dict renders identically
    to a live response and is testable without either.

    Renders ONLY what the endpoint can return: the constrained narration prose (the "why"), each flag as a
    visible warning line (the caught confabulations -- a claim the model was about to make that no receipt
    backs), and the note (which matcher ran + its honesty caveat), always shown so the reader knows the
    honesty level. `unsupported_claims` is not re-rendered separately -- every one of its entries is already
    represented, verbatim, inside `flags`. An empty/thin narration is rendered as an honest first-class
    result ("no receipt-backed narration..."), never as an error. Never raises: a malformed section degrades
    to a one-line notice instead of losing the rest, same discipline as format_explain."""
    obj = obj if isinstance(obj, dict) else {}
    lines = [f"{BOLD}narrate{RST}  {DIM}the accountable-self narration -- opt-in, generates (--why){RST}", "-" * 62]
    try:
        lines += _format_narration(_as_dict(obj.get("constrained_narration")))
    except Exception:
        lines += [f"{BOLD}why did it say this?{RST}", f"  {DIM}couldn't render the narration{RST}"]
    lines.append("")
    try:
        lines += _format_flags(_as_list(obj.get("flags")))
    except Exception:
        lines += [f"{BOLD}caught in the diff{RST}", f"  {DIM}couldn't render the flags{RST}"]
    lines.append("")
    try:
        note = obj.get("note")
        if isinstance(note, str) and note:
            lines.append(f"{DIM}{note}{RST}")
    except Exception:
        pass
    lines.append("-" * 62)
    return "\n".join(lines)


def cmd_explain(args):
    rid = _last_run_id() if args.last else args.run_id
    if not rid:
        raise CloznError("give a run id, or pass --last for the most recent one "
                         "(see ids in the Studio's Runs list, or run something first:  clozn run qwen \"...\")")
    port = args.port or 8090
    print(format_explain(_fetch_explain(port, rid)))
    if args.why:
        print()
        print(format_narrate(_fetch_narrate(port, rid)))


# ------------------------------------------------------------------ preferences (propose-and-review surface)
# `clozn preferences` renders the Studio's POST /preferences (research/preferences.py): the learned-preference
# suggestions the model proposes from your Run Inspector quick-repairs ("Too verbose" x3 -> "make concise your
# default?"). Zero generation -- it reads the accumulated pattern. --approve/--dismiss POST /preferences/resolve
# (approve persists the dial; the ONLY place a dial changes). Terminal-reachable, like `clozn explain`, so the
# learning loop isn't studio-only. format_preferences is a pure JSON->text function, testable with a canned dict.

def _fetch_preferences(port: int) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/preferences", data=b'{"threshold":3}',
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise CloznError(f"couldn't reach the Studio on port {port} ({getattr(e, 'reason', e)}). "
                         f"Start it first:  clozn studio")
    except Exception as e:
        raise CloznError(f"preferences failed: {e}")


def _resolve_preference(port: int, pid: str, action: str) -> dict:
    body = json.dumps({"id": pid, "action": action}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/preferences/resolve", data=body,
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise CloznError(f"{action} failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise CloznError(f"couldn't reach the Studio on port {port} ({getattr(e, 'reason', e)}). "
                         f"Start it first:  clozn studio")


def format_preferences(data: dict) -> str:
    """Pure JSON->text render of the pending proposals (no server -- testable with a canned dict). Lists what
    the model is asking to make a default; nothing here changes a dial (that's `--approve <id>`)."""
    pend = (data or {}).get("pending") or []
    if not pend:
        return (f"{DIM}no suggestions yet -- use the Run Inspector's \"Too verbose / Too cold\" buttons a few "
                f"times and a pattern will surface here.{RST}")
    out = [f"{BOLD}learned-preference suggestions{RST}  {DIM}approve to make one a default, or dismiss{RST}"]
    for p in pend:
        n = len(p.get("evidence") or [])
        ev = f"   {DIM}from {n} repl{'y' if n == 1 else 'ies'}{RST}" if n else ""
        out.append(f"  {p.get('label', '(preference)')}{ev}")
        out.append(f"    {DIM}approve:{RST} clozn preferences --approve {p.get('id')}"
                   f"   {DIM}dismiss:{RST} --dismiss {p.get('id')}")
    return "\n".join(out)


def cmd_preferences(args):
    port = args.port or 8090
    pid = args.approve or args.dismiss
    if pid:
        action = "approve" if args.approve else "dismiss"
        r = _resolve_preference(port, pid, action)
        pr = (r or {}).get("proposal") or {}
        if action == "approve":
            ap = (r or {}).get("applied") or {}
            if ap.get("error"):
                print(f"approved, but the dial couldn't be applied: {ap['error']}")
            else:
                print(f"approved -- {BOLD}{pr.get('dial', '?')}{RST} set to {ap.get('value', '?')} "
                      f"{DIM}(now your default){RST}")
        else:
            print(f"dismissed -- {pr.get('dial', '?')} {DIM}won't resurface unless the pattern gets much "
                  f"stronger{RST}")
        return
    print(format_preferences(_fetch_preferences(port)))


# --------------------------------------------------------------------------------------- test (tiny-test harness)
# `clozn test <file>` (clozn/testkit.py, backlog/tiny-test-harness) -- a minimal, JSON-authored, run-level
# assertion harness on top of the receipt/replay seams: static checks read the stored run record alone
# (zero generation); the one causal check (`leans_on`) runs receipts.py's leave-one-out ablation and is
# honestly SKIPPED (never a silent pass) unless --live is given. testkit.run_suite/evaluate do all the
# work and never print; this section is purely the CLI's load/dispatch/render/exit-code shell around it.

def _import_testkit():
    from clozn import testkit
    return testkit


def _load_test_spec(path: str):
    """Load + shallow-validate a tiny-test JSON spec file. Returns (spec, error) -- exactly one is None. A
    problem here is `clozn test`'s EXIT-2 case (unreadable file, invalid JSON, or no non-empty 'tests'
    list) -- distinct from a per-assertion "error" status (a bad/unknown check inside an otherwise
    well-formed spec), which testkit.run_suite handles itself and only ever costs exit 1."""
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except OSError as e:
        return None, f"could not read {path}: {e}"
    except json.JSONDecodeError as e:
        return None, f"{path} is not valid JSON: {e}"
    if not isinstance(spec, dict) or not isinstance(spec.get("tests"), list) or not spec["tests"]:
        return None, f"{path}: spec must be a JSON object with a non-empty 'tests' list"
    return spec, None


def _fetch_live_receipt(port: int, run_id: str, influence: dict):
    """POST /runs/<id>/receipt on a running `clozn studio` -- the SAME rigorous, both-arms-greedy causal
    receipt receipts.receipt() computes in-process there (clozn_server.py), fetched over the loopback HTTP
    bridge instead of needing an in-process substrate inside this CLI. Returns None on ANY failure (Studio
    not up, run not found, no substrate loaded, bad influence spec) -- never raises: testkit's honesty rule
    (judge_receipt) turns a None receipt into an honest 'skipped' assertion, not a crashed `clozn test` run."""
    url = f"http://127.0.0.1:{port}/runs/{run_id}/receipt"
    body = json.dumps({"influence": influence}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


_STATUS_MARK = {"pass": "✓", "fail": "✗", "skip": "○", "error": "!"}
# Mirrors testkit.CAUSAL_CHECKS -- duplicated (not imported) so this render-only helper stays a plain
# JSON-in/text-out function, exactly like format_explain; if the causal check vocabulary ever grows, change
# both (same convention run_timeline.py's LOW_CONF comment documents for explain.py's copy of it).
_CAUSAL_CHECKS = {"leans_on"}


def format_test_report(suite: dict) -> str:
    """Pure JSON(testkit.run_suite() result) -> text render: one line per test, one indented line per
    assertion, each tagged [static] or [causal] so the two classes never blur together in the output. The
    expected/actual delta only prints on a fail/error (a pass or an honest skip needs no diff). No I/O --
    testable with a canned suite dict, exactly like format_explain."""
    lines = []
    for t in suite.get("tests") or []:
        mark = _STATUS_MARK.get(t.get("status"), "?")
        lines.append(f"{BOLD}{mark} {t.get('name', '(unnamed test)')}{RST}  "
                     f"{DIM}[{t.get('run_id') or '?'}]{RST}")
        for a in t.get("assertions") or []:
            amark = _STATUS_MARK.get(a.get("status"), "?")
            check = a.get("check") or "?"
            kind = "causal" if check in _CAUSAL_CHECKS else "static"
            lines.append(f"    {amark} {DIM}[{kind}]{RST} {check}  {DIM}{a.get('target', '')}{RST}")
            if a.get("status") in ("fail", "error"):
                lines.append(f"        {DIM}expected:{RST} {a.get('expected')!r}")
                lines.append(f"        {DIM}actual:  {RST} {a.get('actual')!r}")
            if a.get("note"):
                lines.append(f"        {DIM}note:{RST} {a['note']}")
    counts = suite.get("counts") or {}
    summary = ", ".join(f"{counts.get(s, 0)} {s}" for s in ("pass", "fail", "skip", "error") if counts.get(s))
    lines.append("")
    lines.append(f"{BOLD}{summary or 'no assertions'}{RST}")
    return "\n".join(lines)


def cmd_test(args):
    """`clozn test <file>`: load a JSON tiny-test spec, resolve each test's run (an id, or "latest"),
    evaluate every assertion, render a report (or --json), optionally --attach the results into each
    touched run's tiny_tests field, and return the process exit code (main() propagates it):
        0 -- every assertion passed (skips allowed)
        1 -- at least one assertion failed or errored
        2 -- the spec file itself couldn't be loaded (bad path / invalid JSON / no 'tests' list)
    """
    testkit = _import_testkit()
    runlog = _import_runlog()
    spec, err = _load_test_spec(args.file)
    if err:
        print(f"{BOLD}clozn test:{RST} {err}", file=sys.stderr)
        return 2

    fetch_receipt = None
    if args.live:
        port = args.port or 8090
        fetch_receipt = lambda run, influence: _fetch_live_receipt(port, run.get("id"), influence)

    suite = testkit.run_suite(spec, get_run=testkit.default_get_run, sub=None, fetch_receipt=fetch_receipt)

    if args.attach:
        for rid, assertions in testkit.results_by_run(suite).items():
            runlog.update_tiny_tests(rid, assertions)

    if args.json:
        print(json.dumps(suite, indent=2, default=str))
    else:
        print(format_test_report(suite))

    return 0 if suite["status"] in ("pass", "skip") else 1


def build_parser():
    """The full argparse tree, factored out of main() so tests can introspect flags without dispatching."""
    p = argparse.ArgumentParser(prog="clozn", description="a reliable front door to the local model engine")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="one-shot: stream a completion to the terminal")
    pr.add_argument("model"); pr.add_argument("prompt", nargs="?", default=None)
    pr.add_argument("--max", type=int, default=256, help="max new tokens (default 256)")
    pr.add_argument("--cpu", action="store_true", help="force the CPU build")
    pr.add_argument("--port", type=int, default=0); pr.add_argument("--mask", type=int, default=None)
    pr.add_argument("--eos", type=int, default=None)
    pr.add_argument("--heat", action="store_true", help="paint each token as it streams by the model's "
                    "confidence (warm = wavered, cool = sure) -- the denoise heatmap, live (AR models)")
    pr.set_defaults(fn=cmd_run)

    ps = sub.add_parser("serve", help="bring up the OpenAI-compatible endpoint")
    ps.add_argument("model"); ps.add_argument("--port", type=int, default=0)
    ps.add_argument("--cpu", action="store_true"); ps.add_argument("--mask", type=int, default=None)
    ps.add_argument("--eos", type=int, default=None)
    ps.add_argument("--sae", default=None, help="on-device SAE readout dir (dims must match the model; "
                    "server refuses politely on mismatch)")
    ps.add_argument("--sae-k", type=int, default=None, help="SAE features kept per position (default 16)")
    ps.set_defaults(fn=cmd_serve)

    sub.add_parser("models", help="list local models + the engine backend").set_defaults(fn=cmd_models)
    pp = sub.add_parser("pull", help="download a model GGUF (by name, or owner/repo/file.gguf)")
    pp.add_argument("model"); pp.set_defaults(fn=cmd_pull)
    ppl = sub.add_parser("plan", help="will it fit? read a GGUF's header (no download, no load, no GPU) "
                         "before you commit to a multi-GB pull")
    ppl.add_argument("model", help="a known model name, a local .gguf path, or a HF resolve/... .gguf URL")
    ppl.add_argument("--vram", type=float, default=None,
                     help="VRAM budget in GB (default: detect via nvidia-smi, else 16)")
    ppl.set_defaults(fn=cmd_plan)
    pst = sub.add_parser("studio", help="launch Clozn Studio (the glass-box UI + the endpoint your tools connect to)")
    pst.add_argument("substrate", nargs="?", default=None, help="qwen (default) | dream | engine")
    pst.add_argument("--port", type=int, default=0); pst.add_argument("--open", action="store_true", help="open the UI in your browser")
    pst.set_defaults(fn=cmd_studio)
    sub.add_parser("ps", help="list running serve daemons").set_defaults(fn=cmd_ps)
    pstop = sub.add_parser("stop", help="stop a serve daemon (by model name, port, or 'all')")
    pstop.add_argument("which"); pstop.set_defaults(fn=cmd_stop)
    pt = sub.add_parser("trace", help="inspect the last run journal entry's confidence timeline")
    pt.add_argument("--list", action="store_true", help="list recent run journal entries instead of showing the last")
    pt.add_argument("--legacy-cache", action="store_true",
                    help="read the old ~/.clozn/traces cache instead of the shared ~/.clozn/runs journal")
    pt.set_defaults(fn=cmd_trace)
    pb = sub.add_parser("branch", help="re-run from an uncertain point on the alternative (the road not taken)")
    pb.add_argument("--at", type=int, default=None, help="token index to fork at (default: the most uncertain)")
    pb.add_argument("--pick", type=int, default=0, help="which alternative to take (0 = the runner-up)")
    pb.add_argument("--max", type=int, default=80); pb.add_argument("--cpu", action="store_true")
    pb.set_defaults(fn=cmd_branch)
    pe = sub.add_parser("explain", help="explain a run: hesitations, active influences, concepts "
                        "(needs `clozn studio` running)")
    pe.add_argument("run_id", nargs="?", default=None, help="run id, as shown in the Studio's Runs list")
    pe.add_argument("--last", action="store_true", help="use the most recently recorded run")
    pe.add_argument("--port", type=int, default=0, help="Studio port (default 8090)")
    pe.add_argument("--why", action="store_true", help="also generate the accountable-self narration (M4): "
                    "a receipt-constrained \"why\", diffed against an independent judge and flagged wherever "
                    "it overclaims. Opt-in -- unlike the rest of `explain`, this GENERATES (two model calls; "
                    "needs the qwen substrate loaded in `clozn studio`)")
    pe.set_defaults(fn=cmd_explain)
    ppref = sub.add_parser("preferences", help="review learned-preference suggestions the model proposes "
                           "from your quick-repairs (needs `clozn studio` running)")
    ppref.add_argument("--approve", metavar="ID", default=None, help="approve a proposal by id (persists the dial)")
    ppref.add_argument("--dismiss", metavar="ID", default=None, help="dismiss a proposal by id")
    ppref.add_argument("--port", type=int, default=0, help="Studio port (default 8090)")
    ppref.set_defaults(fn=cmd_preferences)
    pte = sub.add_parser("test", help="run tiny-test assertions against a stored run (the receipt/replay seams)")
    pte.add_argument("file", help="path to a JSON tiny-test spec (see clozn/testkit.py's module docstring)")
    pte.add_argument("--json", action="store_true",
                     help="print the machine-readable suite result instead of the report")
    pte.add_argument("--attach", action="store_true",
                     help="write results into each touched run's tiny_tests field (rides the receipt_bundle export)")
    pte.add_argument("--live", action="store_true",
                     help="permit causal (leans_on) assertions to run against a live Studio substrate; "
                          "without it they're honestly skipped ('needs --live'), never silently passed")
    pte.add_argument("--port", type=int, default=0, help="Studio port for --live (default 8090)")
    pte.set_defaults(fn=cmd_test)
    return p


def main(argv=None):
    _setup_console()
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help(); return 2
    try:
        rc = args.fn(args)
        return rc if isinstance(rc, int) else 0
    except CloznError as e:
        print(f"{BOLD}clozn:{RST} {e}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
