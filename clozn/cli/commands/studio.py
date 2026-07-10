"""commands.studio -- `clozn studio`: launch Clozn Studio (the glass-box UI + the local OpenAI endpoint
your other tools connect to), or just point at it if it's already up.

HOME/CloznError live on `clozn.cli.main`; imported INSIDE the functions that need them (not at module
level) for the same circular-import reason documented in engine_process.py's module docstring.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.engine_process import REPO, _log_tail


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
    from clozn.cli import main as ctx
    if os.environ.get("CLOZN_STUDIO_PYTHON"):
        return os.environ["CLOZN_STUDIO_PYTHON"]
    cfg = os.path.join(ctx.HOME, "config.json")
    if os.path.isfile(cfg):
        try:
            sp = json.load(open(cfg)).get("studio_python")
            if sp:
                return sp
        except Exception:
            pass
    return sys.executable


def _studio_banner(base, health):
    print(f"  Studio UI:        {fmt.BOLD}{base}/studio.html{fmt.RST}")
    print(f"  OpenAI endpoint:  {fmt.BOLD}{base}/v1{fmt.RST}   (point Open WebUI / Cursor / any client here)")
    print(f"  Lab / brain viz:  {base}/brain.html")
    if health:
        print(f"  Substrate:        {health.get('active')}   (available: {', '.join(health.get('available', []))})")


def cmd_studio(args):
    """Launch Clozn Studio -- the glass-box UI + the local OpenAI endpoint your other tools connect to."""
    from clozn.cli import main as ctx
    port = args.port or 8090
    base = f"http://127.0.0.1:{port}"
    h = _studio_health(port)
    if h:                                                  # already up -> just point at it
        print(f"{fmt.BOLD}Clozn Studio{fmt.RST} already running:")
        _studio_banner(base, h)
        if args.open:
            _open_browser(f"{base}/studio.html")
        return
    server = os.path.join(REPO, "clozn", "server", "app.py")
    if not os.path.isfile(server):
        raise ctx.CloznError(f"studio backend not found at {server}")
    cmd = [_studio_python(), "-m", "clozn.server.app", "--port", str(port)]
    if args.substrate:
        cmd += ["--substrate", args.substrate]
    os.makedirs(ctx.HOME, exist_ok=True)
    logf = open(os.path.join(ctx.HOME, "studio.log"), "w")
    print(f"{fmt.DIM}- starting Clozn Studio (loading the model, ~30s) …{fmt.RST}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            tail = _log_tail(logf); logf.close()
            raise ctx.CloznError(f"studio exited (code {proc.returncode}). {tail} "
                                 "(the backend needs a PyTorch python -- set CLOZN_STUDIO_PYTHON if import failed)")
        h = _studio_health(port)
        if h:
            break
        time.sleep(0.5)
    else:
        proc.terminate(); logf.close()
        raise ctx.CloznError("studio did not come up within 300s.")
    print(f"  {fmt.BOLD}Clozn Studio{fmt.RST} ready in {time.time()-t0:.0f}s\n")
    _studio_banner(base, h)
    if args.open:
        _open_browser(f"{base}/studio.html")
    print(f"\n  {fmt.DIM}your local model now runs under Clozn -- point any client at {base}/v1   -   "
          f"Ctrl-C to stop{fmt.RST}\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n{fmt.DIM}- stopping studio{fmt.RST}", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
    finally:
        logf.close()
