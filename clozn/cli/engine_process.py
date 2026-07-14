"""engine_process -- find the engine build, launch it as a subprocess with the right DLLs on PATH, wait for
/health, and track the `clozn serve` <-> `clozn run` warm-daemon registry (~/.clozn/daemons.json).

HOME/CloznError live on `clozn.cli.main` (the CLI's shared-state owner, mirroring the server's app.py);
every function here that needs either does `from clozn.cli import main as ctx` INSIDE the function body
(never at module level) and reads `ctx.HOME` / raises `ctx.CloznError(...)` at call time. This is
deliberately lazy, not just a style preference: main.py imports THIS module (for _free_port etc.) at its
own module level, so a module-level `from clozn.cli import main as ctx` here would deadlock the first time
anything imports clozn.cli.engine_process before clozn.cli.main has been touched (a real circular import,
not a theoretical one -- caught by directly `import clozn.cli.engine_process` in isolation). Deferring the
import to call time sidesteps it entirely: by the time any of these functions actually run, module loading
has long finished.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (parent of clozn/)
ENGINE_CORE = os.path.join(REPO, "engine", "core")

# Engine builds, most-preferred first. (subdir, is_gpu); the exe sits at <subdir>/ or <subdir>/Release/.
BUILDS = [("build-gpu", True), ("build-cuda", True),
          ("build-ggml-cpu", False), ("build-serve", False), ("build-cpu", False)]


def find_engine(prefer_gpu=True) -> tuple[str, list[str], bool]:
    """-> (exe_path, dll_dirs, is_gpu). Raises if no build exists."""
    from clozn.cli import main as ctx
    override = os.environ.get("CLOZN_ENGINE_BIN")
    if override:
        exe = os.path.abspath(os.path.expanduser(override))
        if not os.path.isfile(exe):
            raise ctx.CloznError(f"CLOZN_ENGINE_BIN does not point to a file: {exe}")
        root = os.path.dirname(exe)
        bins = [path for path in (root, os.path.join(root, "bin")) if os.path.isdir(path)]
        gpu = os.environ.get("CLOZN_ENGINE_GPU", "").strip().lower() in ("1", "true", "yes", "on")
        return exe, bins, gpu
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
        raise ctx.CloznError("no engine found. See docs/DEVELOPMENT.md, or set CLOZN_ENGINE_BIN.")
    cands.sort(key=lambda c: (0 if c[2] else 1) if prefer_gpu else (1 if c[2] else 0))
    return cands[0]


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
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _terminate_process(proc, timeout: float = 5.0) -> None:
    """Best-effort child cleanup, including interrupted startup."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=timeout)
        except Exception:
            pass


def spawn_engine(model: str, port: int, flags: dict, *, prefer_gpu=True, logf=None, boot_timeout=180):
    """Start an engine on `port`, wait until /health is ok. Returns (proc, health, is_gpu)."""
    from clozn.cli import main as ctx
    exe, dll_dirs, gpu = find_engine(prefer_gpu)
    args = _launch_args(exe, model, port, flags, gpu)
    proc = subprocess.Popen(args, env=_env_with_dlls(dll_dirs, gpu),
                            stdout=logf or subprocess.DEVNULL, stderr=subprocess.STDOUT)
    started = time.monotonic()
    try:
        while time.monotonic() - started < boot_timeout:
            if proc.poll() is not None:                        # died before healthy
                raise ctx.CloznError(f"engine exited (code {proc.returncode}). {_log_tail(logf)}")
            h = _health(port)
            if h and h.get("status") == "ok":
                return proc, h, gpu
            time.sleep(0.3)
        raise ctx.CloznError(f"engine did not become healthy within {boot_timeout}s. {_log_tail(logf)}")
    except BaseException:
        _terminate_process(proc)
        raise


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
# Stale entries self-heal: a dead gateway fails /readyz in _find_warm and is ignored (then pruned).

def _reg_path() -> str:
    from clozn.cli import main as ctx
    return os.path.join(ctx.HOME, "daemons.json")


def _reg_read() -> dict:
    try:
        with open(_reg_path(), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _reg_write(d: dict):
    from clozn.cli import main as ctx
    from clozn._io import atomic_write_json
    os.makedirs(ctx.HOME, exist_ok=True)
    try:
        atomic_write_json(_reg_path(), d)
    except Exception:
        pass


def _register(model: str, port: int, gpu: bool, mode: str, pid: int, **runtime_fields):
    d = _reg_read()
    d[str(port)] = {"model": model, "gpu": gpu, "mode": mode, "pid": pid, **runtime_fields}
    _reg_write(d)


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
    """A live product gateway for this exact model -> (public_port, gpu, mode), else None."""
    from clozn.cli.runtime_process import gateway_health

    d = _reg_read(); hit = None; dirty = False
    for port, ent in list(d.items()):
        h = gateway_health(int(port), timeout=1.0)
        if not h:
            d.pop(port, None); dirty = True; continue           # prune the dead
        if ent.get("model") == model and hit is None:
            hit = (int(port), bool(ent.get("gpu")), ent.get("mode", h.get("mode", "?")))
    if dirty:
        _reg_write(d)
    return hit
