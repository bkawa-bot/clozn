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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
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


def _setup_console():
    """UTF-8 stdout (so model tokens print right on Windows), ANSI enabled where supported, plain otherwise."""
    global DIM, BOLD, RST
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    use = sys.stderr.isatty()
    if os.name == "nt" and use:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            for h in (-11, -12):                               # stdout, stderr handles
                hd = k.GetStdHandle(h); m = ctypes.c_uint32()
                if k.GetConsoleMode(hd, ctypes.byref(m)):
                    k.SetConsoleMode(hd, m.value | 0x0004)     # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            use = False
    if use:
        DIM, BOLD, RST = "\033[2m", "\033[1m", "\033[0m"


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


def stream_ar(port: int, prompt: str, max_tokens: int):
    """POST /v1/completions (stream); print each committed token. -> (count, trace steps w/ conf + alts).

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
            if obj.get("type") == "tokens_committed":       # print live as tokens land (unchanged UX)
                for it in obj.get("items", []):
                    sys.stdout.write(it.get("piece", "")); sys.stdout.flush()
                    n += 1
    # Accumulation (pair tokens_committed with step_lens by position) lives in runlog so the CLI and the
    # engine-chat capture share ONE tested implementation. Fall back to a local pairing if the import fails
    # -- the stdlib CLI must never break on a missing sibling.
    try:
        sys.path.insert(0, os.path.join(REPO, "research"))
        import runlog
        steps = runlog.accumulate_ar_events(frames)
    except Exception:
        by_pos: dict = {}
        for obj in frames:
            if obj.get("type") == "tokens_committed":
                for it in obj.get("items", []):
                    by_pos[it.get("pos")] = {"pos": it.get("pos"), "piece": it.get("piece", ""),
                                             "conf": round(float(it.get("conf", 0.0)), 3), "alts": []}
            elif obj.get("type") == "step_lens":
                step = by_pos.get((obj.get("positions") or [None])[0])
                if step:
                    step["alts"] = [{"piece": p, "prob": round(float(pr), 3)}
                                    for p, pr in zip(obj.get("pieces", []), obj.get("probs", []))
                                    if p != step["piece"]][:3]
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
# Every AR run auto-saves its timeline to ~/.clozn/traces/<id>.json so a run is debuggable after the fact:
# `clozn trace` shows the per-token confidence + what the model almost said -- the seed of branch-a-bad-answer.

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
    for old in sorted(glob.glob(os.path.join(d, "*.json")))[:-25]:   # keep the last 25 runs
        try:
            os.remove(old)
        except Exception:
            pass
    return path


def _confbar(c: float, width=8) -> str:
    full = int(round(max(0.0, min(1.0, c)) * width))
    return "█" * full + "░" * (width - full)


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
    server = os.path.join(REPO, "research", "clozn_server.py")
    if not os.path.isfile(server):
        raise CloznError(f"studio backend not found at {server}")
    cmd = [_studio_python(), server, "--port", str(port)]
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


def _run_turn(port, mode, text, max_tokens, gpu, model_name, prompt_for_trace):
    """One generation: stream (AR, auto-saving the trace) or denoise (diffusion). Prints stats. -> response."""
    g0 = time.time()
    steps = []
    if mode == "autoregressive":
        n, steps = stream_ar(port, text, max_tokens)
        sys.stdout.write("\n")
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
        sys.path.insert(0, os.path.join(REPO, "research"))
        import runlog
        # stream_ar hands us per-token steps ({piece, conf, alts}); runlog owns the steps->trace mapping so
        # the on-disk trace schema stays one contract shared with the engine-chat capture (issue B3).
        trace = runlog.steps_to_trace(steps)
        runlog.record(source="cli", client="cli", model=model_name, substrate="engine",
                      messages=[{"role": "user", "content": prompt}], response=resp,
                      trace=trace, started=started)
    except Exception:
        pass


def _repl(port, mode, flags, fam, gpu, model, max_tokens):
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
        resp = _run_turn(port, mode, text, max_tokens, gpu, name, msg)
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
            _run_turn(port, mode, text, args.max, gpu, _friendly(model), args.prompt)
            if mode == "autoregressive":
                print(f"{DIM}  clozn trace   inspect where it was uncertain + what it almost said{RST}",
                      file=sys.stderr)
        else:                                                  # interactive REPL (Ollama-style)
            _repl(port, mode, flags, fam, gpu, model, args.max)
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


def cmd_trace(args):
    d = os.path.join(HOME, "traces")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        print('no traces yet -- run something first:  clozn run qwen "..."'); return
    if args.list:
        print(f"{'WHEN':<18} {'MODEL':<11} {'TOK':>4}  PROMPT")
        for f in files[-12:]:
            m = json.load(open(f)).get("meta", {})
            print(f"{m.get('id', ''):<18} {m.get('model', ''):<11} {m.get('n', 0):>4}  {m.get('prompt', '')[:46]}")
        return
    tr = json.load(open(files[-1]))
    m, steps = tr.get("meta", {}), tr.get("steps", [])
    print(f"{BOLD}{m.get('model', '?')}{RST}  \"{m.get('prompt', '')[:64]}\"")
    print(f"{DIM}{m.get('n', 0)} tokens - {m.get('backend', '?')} - short bar = less sure; "
          f"'almost' = what it nearly said{RST}")
    print("-" * 62)
    for s in steps:
        piece = (s.get("piece", "") or "").replace("\n", "\\n").replace("\t", "\\t")
        conf = s.get("conf", 0.0)
        mark = " " if conf >= 0.5 else "?"
        line = f" {mark} {piece[:16]:<16} {_confbar(conf)} {conf:.2f}"
        if conf < 0.5 and s.get("alts"):
            alts = "  ".join(f"{(a['piece'] or '').strip() or '_'} {a['prob']:.2f}" for a in s["alts"][:3])
            line += f"   {DIM}almost: {alts}{RST}"
        print(line)
    lows = [s for s in steps if s.get("conf", 1) < 0.5]
    print("-" * 62)
    tail = " -> " + ", ".join((s.get("piece", "") or "").strip() for s in lows[:6]) if lows else ""
    print(f"{DIM}{len(lows)} uncertain moment(s){tail}{RST}")


def cmd_branch(args):
    """Take the road not taken: re-run from an uncertain point with the alternative the model nearly chose.

    Text-level (re-runs prompt + kept tokens + the alt through /v1/completions), so token boundaries can
    shift a hair -- but it shows, concretely, 'what if it had said X instead'. The seed of branch-a-bad-answer."""
    files = sorted(glob.glob(os.path.join(HOME, "traces", "*.json")))
    if not files:
        raise CloznError('no trace yet -- run something first:  clozn run qwen "..."')
    tr = json.load(open(files[-1])); meta = tr.get("meta", {})
    steps = [s for s in tr.get("steps", []) if (s.get("piece", "") or "").strip()]   # branch on real tokens
    if not steps:
        raise CloznError("the last trace has no branchable tokens.")
    idx = (max(0, min(args.at, len(steps) - 1)) if args.at is not None
           else min(range(len(steps)), key=lambda i: steps[i].get("conf", 1.0)))
    step = steps[idx]; alts = step.get("alts", [])
    if not alts:
        raise CloznError(f"no recorded alternative at '{step['piece'].strip()}' (conf {step.get('conf', 0):.2f}).")
    alt = alts[max(0, min(args.pick, len(alts) - 1))]
    model = resolve_model(meta.get("model", "")); flags = _flags_for(model)
    head = _chat_wrap(meta.get("prompt", "")) if flags.get("chat") else meta.get("prompt", "")
    kept = "".join(s["piece"] for s in steps[:idx])
    prefix = head + kept + alt["piece"]
    print(f"{BOLD}branch{RST} of \"{meta.get('prompt', '')[:54]}\"")
    print(f"  fork at token {idx}: it chose {BOLD}{step['piece'].strip()!r}{RST} ({step.get('conf', 0):.2f})"
          f"  ->  branch on {BOLD}{alt['piece'].strip()!r}{RST} ({alt['prob']:.2f})")
    print(f"  {DIM}original:{RST} {''.join(s['piece'] for s in steps).strip()[:130]}")
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


def main(argv=None):
    _setup_console()
    p = argparse.ArgumentParser(prog="clozn", description="a reliable front door to the local model engine")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="one-shot: stream a completion to the terminal")
    pr.add_argument("model"); pr.add_argument("prompt", nargs="?", default=None)
    pr.add_argument("--max", type=int, default=256, help="max new tokens (default 256)")
    pr.add_argument("--cpu", action="store_true", help="force the CPU build")
    pr.add_argument("--port", type=int, default=0); pr.add_argument("--mask", type=int, default=None)
    pr.add_argument("--eos", type=int, default=None)
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
    pst = sub.add_parser("studio", help="launch Clozn Studio (the glass-box UI + the endpoint your tools connect to)")
    pst.add_argument("substrate", nargs="?", default=None, help="qwen (default) | dream | engine")
    pst.add_argument("--port", type=int, default=0); pst.add_argument("--open", action="store_true", help="open the UI in your browser")
    pst.set_defaults(fn=cmd_studio)
    sub.add_parser("ps", help="list running serve daemons").set_defaults(fn=cmd_ps)
    pstop = sub.add_parser("stop", help="stop a serve daemon (by model name, port, or 'all')")
    pstop.add_argument("which"); pstop.set_defaults(fn=cmd_stop)
    pt = sub.add_parser("trace", help="inspect the last run's confidence timeline")
    pt.add_argument("--list", action="store_true", help="list recent runs instead of showing the last")
    pt.set_defaults(fn=cmd_trace)
    pb = sub.add_parser("branch", help="re-run from an uncertain point on the alternative (the road not taken)")
    pb.add_argument("--at", type=int, default=None, help="token index to fork at (default: the most uncertain)")
    pb.add_argument("--pick", type=int, default=0, help="which alternative to take (0 = the runner-up)")
    pb.add_argument("--max", type=int, default=80); pb.add_argument("--cpu", action="store_true")
    pb.set_defaults(fn=cmd_branch)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help(); return 2
    try:
        args.fn(args)
        return 0
    except CloznError as e:
        print(f"{BOLD}clozn:{RST} {e}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
