"""Launch the explicitly non-product PyTorch workbench."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.engine_process import REPO, _log_tail


def _lab_python() -> str:
    from clozn.cli import main as ctx
    if os.environ.get("CLOZN_LAB_PYTHON"):
        return os.environ["CLOZN_LAB_PYTHON"]
    config = os.path.join(ctx.HOME, "config.json")
    try:
        with open(config, encoding="utf-8") as handle:
            value = json.load(handle).get("lab_python")
        if value:
            return str(value)
    except Exception:
        pass
    return sys.executable


def _health(port: int):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=2) as response:
            value = json.loads(response.read())
        return value if value.get("service") == "clozn-lab" else None
    except Exception:
        return None


def _open(url: str):
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def cmd_lab(args):
    from clozn.cli import main as ctx

    port = args.port or 8090
    os.makedirs(ctx.HOME, exist_ok=True)
    log = open(os.path.join(ctx.HOME, "lab.log"), "w", encoding="utf-8")
    command = [_lab_python(), "-m", "clozn.lab.app", args.substrate, "--port", str(port)]
    print(f"{fmt.DIM}- starting the {args.substrate} lab workbench …{fmt.RST}", file=sys.stderr, flush=True)
    env = dict(os.environ)
    env.pop("CLOZN_ENGINE_PORT", None)
    env["CLOZN_RUNTIME_KIND"] = "lab"
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        started = time.monotonic()
        state = None
        while time.monotonic() - started < 300:
            if process.poll() is not None:
                raise ctx.CloznError(
                    f"lab exited during startup (code {process.returncode}). {_log_tail(log)} "
                    "Set CLOZN_LAB_PYTHON to a Python with the lab dependencies."
                )
            state = _health(port)
            if state:
                break
            time.sleep(0.5)
        if not state:
            raise ctx.CloznError("lab did not become ready within 300s")
        base = f"http://127.0.0.1:{port}"
        print(f"  {fmt.BOLD}Clozn lab{fmt.RST} ({args.substrate}): {base}/")
        print(f"  {fmt.DIM}No /v1 API is exposed; this process is a workbench only.{fmt.RST}")
        if args.open:
            _open(base + "/")
        process.wait()
    except KeyboardInterrupt:
        print(f"\n{fmt.DIM}- stopping lab{fmt.RST}", file=sys.stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except Exception:
                process.kill()
        log.close()
