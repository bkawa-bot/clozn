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
    ("qwen2.5-7b-instruct",  "qwen",      {"chat": True}),
    ("qwen2.5-0.5b-instruct", "qwen-0.5b", {"chat": True}),
    ("dream-v0-instruct",    "dream",     {"chat": True, "mask": 151666}),
    ("llada-8b-instruct",    "llada",     {"chat": True, "mask": 126336, "eos": 126081}),
    ("open-dcoder",          "dcoder",    {"mask": 151666}),
]

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


def _chat_wrap(prompt: str) -> str:
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")


def stream_ar(port: int, prompt: str, max_tokens: int) -> int:
    """POST /v1/completions with stream:true; print each committed token's piece. -> token count."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    n = 0
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
            if obj.get("type") == "tokens_committed":
                for it in obj.get("items", []):
                    sys.stdout.write(it.get("piece", "")); sys.stdout.flush()
                    n += 1
    return n


def complete_once(port: int, prompt: str, max_tokens: int) -> str:
    """Non-streaming /v1/completions (used for diffusion, which commits out of reading order)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read())
    return (r.get("choices") or [{}])[0].get("text", "")


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


def cmd_run(args):
    model = resolve_model(args.model)
    prompt = args.prompt if args.prompt is not None else (sys.stdin.read() if not sys.stdin.isatty() else None)
    if not prompt:
        raise CloznError('no prompt. Usage: clozn run <model> "your prompt"')
    flags = _flags_for(model)
    if args.mask is not None:
        flags["mask"] = args.mask
    if args.eos is not None:
        flags["eos"] = args.eos
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
        is_chat = flags.get("chat") and mode == "autoregressive"
        text = _chat_wrap(prompt) if is_chat else prompt
        g0 = time.time()
        if mode == "autoregressive":
            n = stream_ar(port, text, args.max)
            sys.stdout.write("\n")
        else:                                                  # diffusion: print the final ordered text
            out = complete_once(port, text, args.max).strip()
            print(out); n = len(out.split())
        dt = time.time() - g0
        rate = f", ~{n/dt:.0f} tok/s" if dt > 0 and mode == "autoregressive" else ""
        print(f"{DIM}- {n} tokens in {dt:.1f}s{rate} - {'GPU' if gpu else 'CPU'}{RST}", file=sys.stderr)
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
    ps.set_defaults(fn=cmd_serve)

    sub.add_parser("models", help="list local models + the engine backend").set_defaults(fn=cmd_models)
    sub.add_parser("ps", help="list running serve daemons").set_defaults(fn=cmd_ps)
    pstop = sub.add_parser("stop", help="stop a serve daemon (by model name, port, or 'all')")
    pstop.add_argument("which"); pstop.set_defaults(fn=cmd_stop)

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
