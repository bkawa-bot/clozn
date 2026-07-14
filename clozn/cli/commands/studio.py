"""``clozn studio`` attaches to the product gateway; it never launches a model."""
from __future__ import annotations

from clozn.cli import formatting as fmt
from clozn.cli.engine_process import _reg_read
from clozn.cli.runtime_process import gateway_health


def _open_browser(url):
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def _find_gateway(requested_port: int = 0):
    candidates = []
    if requested_port:
        candidates.append(int(requested_port))
    else:
        candidates.extend(int(p) for p in _reg_read())
        if 8080 not in candidates:
            candidates.append(8080)
    for port in candidates:
        health = gateway_health(port)
        if health:
            return port, health
    return None


def _studio_banner(base: str, health: dict) -> None:
    print(f"  Studio UI:        {fmt.BOLD}{base}/{fmt.RST}")
    print(f"  OpenAI endpoint:  {fmt.BOLD}{base}/v1{fmt.RST}")
    print(f"  Model:            {health.get('model') or 'unknown'}")
    print(f"  Runtime:          {health.get('mode') or 'unknown'}")


def cmd_studio(args):
    """Find the already-running gateway and point the user/browser at its UI."""
    from clozn.cli import main as ctx

    found = _find_gateway(args.port or 0)
    if found is None:
        suffix = f" on port {args.port}" if args.port else ""
        raise ctx.CloznError(
            f"no Clozn runtime is ready{suffix}. Start one first: clozn serve <model>"
        )
    port, health = found
    base = f"http://127.0.0.1:{port}"
    print(f"{fmt.BOLD}Clozn Studio{fmt.RST} is attached to the running product runtime:")
    _studio_banner(base, health)
    if args.open:
        _open_browser(base + "/")
