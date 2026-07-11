"""formatting -- the CLI's terminal output layer: the color globals + every pure render helper that turns
numbers/dicts into text (confidence bars, the denoise-style truecolor heatmap, sparklines, terminal width).

DIM/BOLD/RST/COLOR are set once by `_setup_console()` (called from main() at startup) and read LIVE by
every function below -- they live here, and only here. Any other module that needs the raw value (for its
own f-strings, not through one of these functions) MUST do `from clozn.cli import formatting as fmt` and
read `fmt.DIM` etc. at the point of use -- never `from clozn.cli.formatting import DIM`, which would bind
a stale copy immune to a later _setup_console() call or a test's monkeypatch. The functions defined here
are safe to import directly (`from clozn.cli.formatting import _paint`): a function always reads its OWN
module's globals, so `_paint` still sees live updates to this module's COLOR/RST no matter who calls it.
"""
from __future__ import annotations

import os
import shutil
import sys

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


def _oneline(text) -> str:
    """Collapse embedded newlines/tabs to their escaped literal form ('\\n' / '\\t') so a chunk of model
    text can be printed on the single, correctly-indented terminal line its caller built for it -- a raw
    '\n' inside a reconstructed reply (e.g. `clozn branch`'s "original:"/"branch:" lines) would otherwise
    split into multiple unindented lines and read as garbled/misaligned output. Mirrors the per-token
    escaping trace_io._render_trace and explain._format_confidence already do at the single-token level."""
    return str(text if text is not None else "").replace("\n", "\\n").replace("\t", "\\t")


def _num(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


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


# --- sparklines: one glyph per token position, height/color-scaled by confidence -- used by `clozn explain`'s
#     confidence panel (a compact, never-synthesized shape of where the model wavered across the reply).
_SPARK = "▁▂▃▄▅▆▇█"      # 8 heights for a compact per-token confidence shape (never a synthesized aggregate)
_SPARK_MAX = 400          # a defensive cap so a pathologically long run can't blow up one decorative line


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
