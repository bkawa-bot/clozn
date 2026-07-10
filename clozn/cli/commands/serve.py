"""commands.serve -- `clozn serve` (bring up the OpenAI-compatible endpoint as a background daemon) plus
its small companion commands `clozn ps` (list running daemons) and `clozn stop` (kill one), since all three
share the one daemon registry in engine_process.py.

CloznError lives on `clozn.cli.main`; it's imported INSIDE the functions that raise it (not at module
level) for the same circular-import reason documented in engine_process.py's module docstring.
"""
from __future__ import annotations

import sys
import time

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import _flags_for, _friendly, resolve_model
from clozn.cli.engine_process import _health, _kill, _reg_read, _reg_write, _register, _unregister, spawn_engine


def cmd_serve(args):
    from clozn.cli import main as ctx
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
        raise ctx.CloznError(f"port {port} already serving something. Pick another with --port.")
    print(f"{fmt.DIM}- loading {_friendly(model)} …{fmt.RST}", file=sys.stderr, flush=True)
    t0 = time.time()
    proc, health, gpu = spawn_engine(model, port, flags, prefer_gpu=not args.cpu, logf=None)
    base = f"http://127.0.0.1:{port}"
    print(f"\n  {fmt.BOLD}{_friendly(model)}{fmt.RST} ready on {'GPU' if gpu else 'CPU'} build "
          f"({health.get('mode')}) in {time.time()-t0:.1f}s")
    print(f"  OpenAI-compatible endpoint:  {fmt.BOLD}{base}/v1{fmt.RST}")
    print(f"  text completions:            POST {base}/v1/completions")
    print(f"  live viz / health:           {base}/   -   {base}/health")
    print(f"\n  {fmt.DIM}point any OpenAI client at {base}/v1  -  `clozn run {_friendly(model)} ...` reuses "
          f"this  -  Ctrl-C to stop{fmt.RST}\n")
    _register(model, port, gpu, health.get("mode", "?"), proc.pid)   # so `clozn run`/`ps`/`stop` can find it
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n{fmt.DIM}- stopping{fmt.RST}", file=sys.stderr)
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
    from clozn.cli import main as ctx
    d = _reg_read()
    targets = [(port, ent) for port, ent in d.items()
               if args.which in ("all", str(port)) or _friendly(ent.get("model", "")) == args.which]
    if not targets:
        raise ctx.CloznError(f"no running daemon matches '{args.which}'. See: clozn ps")
    for port, ent in targets:
        if ent.get("pid"):
            _kill(int(ent["pid"]))
        d.pop(port, None)
        print(f"stopped {_friendly(ent.get('model', '?'))} on port {port}")
    _reg_write(d)
