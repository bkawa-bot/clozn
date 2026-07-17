"""The shareable receipt card: `render_card(bundle) -> str` turns ONE export bundle
(`clozn.receipts.bundle.build(run, explain=...)` — the exact object GET /runs/<id>/export returns,
optionally + a `lineage` key from `clozn.runs.store.lineage`) into ONE self-contained HTML document a
person can save or post anywhere. Pure function: dict in, string out — zero model calls, zero file IO,
zero network. The card is a RENDERING of what the bundle already carries, never a new computation: a
receipt that was never computed renders as its honest absence, not as a blank or a guess.

Visual language: heavnOS (studio/heavn/theme.css is the color reference — values are inlined here, the
file is never imported): pearly sky washes on an off-white ground, frosted light panels, uppercase
micro-labels, square status LEDs, and ONE dark panel — the twilight-indigo CRT (#2B3160 -> #1E2447,
never black) whose pale-mint phosphor text carries the reply with per-token confidence shading.

Injection-proof by construction: every string that originated outside this module (prompt, reply,
tokens, card texts, ids, lens labels) passes through html.escape before it touches the document, and
the document ships no <script> at all — a reply containing `<script>` renders inert. Self-contained by
construction: no src/href attributes, no webfonts, no images, no JS; system font stacks only.
"""
from __future__ import annotations

import html


# Cap the phosphor token stream so a pathological trace can't blow the ~150KB budget. Typical replies
# are <=256 tokens; the cap only ever bites on something abnormal, and it says so on the card.
MAX_TOKENS = 4000
LOW_CONF = 0.5   # matches clozn.receipts.explain.LOW_CONF — ONE "unsure" convention

_ABSENT_RECEIPTS = "no receipts computed for this run — receipts are measured on demand, never assumed"
_ABSENT_LENS = ("no lens readout recorded on this run — the lens reads on demand from the engine "
                "substrate, and none was captured here")
_FOOTER_CREDO = ("measured, not asserted — every number above comes from a recorded run or an "
                 "explicit computation")
# Mirrors clozn.server.app._JLENS_NOTE — the shipped, unskippable J-lens honesty caption.
_JLENS_CAPTION = ("fitted linear Jacobian lens, transferred to this GGUF; a per-token 'disposed to "
                  "say' read, NOT the model's literal thought — a linear lens always emits something.")


def _esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _num(v, nd: int = 3, signed: bool = False) -> str:
    f = _float(v)
    if f is None:
        return ""
    return f"{f:+.{nd}f}" if signed else f"{f:.{nd}f}"


# ------------------------------------------------------------------------------------ inline stylesheet
# heavnOS values (reference: studio/heavn/theme.css; UX doc §11.2). Inlined, never imported.
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html{background:#E4EAF8}
body{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;color:#2A3252;font-size:12.5px;
 line-height:1.55;padding:28px 14px;min-height:100vh;-webkit-font-smoothing:antialiased;
 background:
  radial-gradient(1000px 600px at 8% -5%,rgba(127,180,240,.34),transparent 60%),
  radial-gradient(900px 560px at 96% 2%,rgba(182,176,218,.36),transparent 56%),
  radial-gradient(820px 620px at 50% 108%,rgba(95,200,188,.26),transparent 60%),
  linear-gradient(165deg,#F3F5FC 0%,#EAEFF9 50%,#E9EDF8 100%)}
.card{max-width:840px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.mod{position:relative;border-radius:16px;border:1px solid rgba(255,255,255,.8);
 background:linear-gradient(180deg,rgba(255,255,255,.62),rgba(240,246,252,.42));
 box-shadow:0 10px 30px rgba(100,115,160,.14),inset 0 1px 0 rgba(255,255,255,.95)}
.mod-h{display:flex;align-items:center;gap:9px;padding:12px 16px 8px;flex-wrap:wrap}
.cap{font-family:"Segoe UI",system-ui,sans-serif;font-weight:600;letter-spacing:.22em;
 text-transform:uppercase;font-size:10px;color:#4A5878}
.led{width:7px;height:7px;background:#5FC8BC;box-shadow:0 0 8px #5FC8BC;flex:none;
 animation:heartbeat 4s ease-in-out infinite}
.led.blue{background:#4C8DF0;box-shadow:0 0 8px #4C8DF0}
.led.lilac{background:#B6B0DA;box-shadow:0 0 8px #B6B0DA}
@keyframes heartbeat{0%,100%{opacity:.55}50%{opacity:1}}
@media (prefers-reduced-motion:reduce){.led{animation:none;opacity:.85}}
.tag{font-family:"Segoe UI",system-ui,sans-serif;font-size:8.5px;letter-spacing:.12em;
 text-transform:uppercase;padding:2px 7px;border-radius:4px;white-space:nowrap}
.tag.cap-t{color:#1B7F74;background:rgba(95,200,188,.14);border:1px solid rgba(95,200,188,.5)}
.tag.der-t{color:#1B87A8;background:rgba(44,191,232,.12);border:1px solid rgba(44,191,232,.45)}
.tag.warn-t{color:#C24A31;background:rgba(242,109,79,.10);border:1px solid rgba(242,109,79,.45)}
.mod-b{padding:4px 16px 14px}
.wordmark{font-family:"Segoe UI",system-ui,sans-serif;font-weight:700;font-size:18px;
 letter-spacing:.16em;color:#4A5878}
.wordmark b{color:#36AEC4}
.mast-sub{font-family:"Segoe UI",system-ui,sans-serif;font-size:8.5px;letter-spacing:.26em;
 text-transform:uppercase;color:#8290AC}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:7px 18px;
 padding:8px 16px 6px}
.meta .k{font-family:"Segoe UI",system-ui,sans-serif;font-size:8px;letter-spacing:.18em;
 text-transform:uppercase;color:#8290AC;display:block}
.meta .v{font-size:11.5px;color:#2A3252;word-break:break-all}
.meta .v.warn{color:#C24A31;font-weight:600}
.legend{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px 16px 13px;
 font-family:"Segoe UI",system-ui,sans-serif;font-size:9.5px;color:#8290AC}
.well{border:1px solid rgba(128,142,190,.16);border-radius:9px;padding:9px 12px;
 background:rgba(255,255,255,.5);white-space:pre-wrap;word-break:break-word;color:#4A5878}
.turnnote{font-size:9px;color:#8290AC;font-style:italic;padding:4px 2px 0}
.crt{position:relative;border-radius:10px;margin:10px 0 6px;padding:30px 16px 14px;overflow:hidden;
 border:1px solid rgba(182,176,218,.6);
 background:
  radial-gradient(120% 100% at 50% 0%,rgba(95,200,188,.20),transparent 60%),
  radial-gradient(90% 80% at 85% 100%,rgba(143,168,232,.26),transparent 62%),
  linear-gradient(180deg,#2B3160,#1E2447);
 box-shadow:inset 0 0 46px rgba(22,26,64,.55),inset 0 0 70px rgba(127,180,240,.12),
  0 2px 0 rgba(255,255,255,.6)}
.crt::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:.18;
 background:repeating-linear-gradient(180deg,rgba(0,0,0,0) 0 2px,rgba(18,22,56,.6) 2px 3px)}
.crt-st{position:absolute;top:8px;left:13px;font-family:"Segoe UI",system-ui,sans-serif;font-size:8px;
 letter-spacing:.16em;text-transform:uppercase;color:#5FC8BC}
.crt-id{position:absolute;top:8px;right:13px;font-size:8.5px;letter-spacing:.12em;color:#8A94C4}
.crt-text{position:relative;font-size:15px;line-height:1.55;color:#B8F5E4;
 text-shadow:0 0 6px rgba(122,235,214,.5);white-space:pre-wrap;word-break:break-word}
.tk.lo{border-bottom:1px dotted rgba(184,245,228,.75)}
.conf-legend{font-family:"Segoe UI",system-ui,sans-serif;font-size:8.5px;letter-spacing:.08em;
 color:#8290AC;padding:6px 2px 0}
.absent{padding:10px 2px 6px;color:#8290AC;font-style:italic}
.rrow{border-top:1px solid rgba(128,142,190,.16);padding:10px 0 12px}
.rrow:first-child{border-top:none}
.r-inf{font-size:12px;color:#2A3252}
.r-chips{display:flex;gap:8px;margin:5px 0 7px;flex-wrap:wrap;align-items:center}
.chip{font-family:"Segoe UI",system-ui,sans-serif;font-size:8.5px;letter-spacing:.1em;
 text-transform:uppercase;padding:1.5px 8px;border-radius:8px;
 border:1px solid rgba(128,142,190,.32);color:#4A5878}
.chip.eff{color:#1B7F74;border-color:rgba(95,200,188,.6);background:rgba(95,200,188,.12)}
.chip.noeff{color:#8290AC}
.chip.cv{color:#1B87A8;border-color:rgba(44,191,232,.5);background:rgba(44,191,232,.08)}
.nats{font-size:11px;color:#1B87A8;font-weight:600}
.dep{display:grid;grid-template-columns:minmax(80px,150px) 58px 1fr;gap:3px 10px;align-items:center;
 max-width:480px;margin-top:3px}
.dep .p{white-space:pre;overflow:hidden;text-overflow:ellipsis;font-size:11px;color:#2A3252}
.dep .d{font-size:10px;text-align:right;color:#4A5878}
.bar{height:6px;border-radius:3px;background:rgba(90,130,155,.16);overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#4C8DF0,#5FC8BC)}
.bar.neg i{background:linear-gradient(90deg,#B6B0DA,#9A92C8)}
.small-note{font-size:9px;color:#8290AC;padding-top:5px;line-height:1.6}
.active-line{font-size:10.5px;color:#4A5878;padding:2px 0 8px}
.jgroup{padding:7px 0 3px;border-top:1px solid rgba(128,142,190,.16)}
.jgroup:first-child{border-top:none}
.jlayer{font-family:"Segoe UI",system-ui,sans-serif;font-size:8.5px;letter-spacing:.16em;
 text-transform:uppercase;color:#4A5878;padding-bottom:5px}
.jchips{display:flex;flex-wrap:wrap;gap:4px}
.jchip{font-size:9.5px;padding:1px 7px;border-radius:8px;border:1px solid rgba(154,146,200,.5);
 color:#5F5794;background:rgba(182,176,218,.16);white-space:nowrap;max-width:100%;
 overflow:hidden;text-overflow:ellipsis}
.provenance{font-size:8.5px;color:#8290AC;line-height:1.7;padding-top:8px}
.tree{font-size:11px;color:#4A5878;white-space:pre;overflow-x:auto;padding:6px 2px}
.tree b{color:#2A3252}
.foot{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:11px 16px}
.flowbar{height:7px;flex:1;min-width:160px;border-radius:4px;opacity:.85;
 background:linear-gradient(90deg,#4C8DF0,#2CBFE8 34%,#5FC8BC 62%,#B6B0DA)}
.credo{font-family:"Segoe UI",system-ui,sans-serif;font-size:8.5px;letter-spacing:.1em;
 text-transform:uppercase;color:#4A5878}
.foot .rid{font-size:9px;color:#8290AC}
"""


# ------------------------------------------------------------------------------------------- sections
def _mod(led: str, title: str, tag_html: str, body_html: str) -> str:
    return (f'<section class="mod"><div class="mod-h"><span class="led {led}"></span>'
            f'<span class="cap">{_esc(title)}</span>{tag_html}</div>'
            f'<div class="mod-b">{body_html}</div></section>')


def _duration(timing: dict) -> str:
    ms = _float(timing.get("duration_ms"))
    if ms is None:
        return ""
    return f"{ms / 1000:.1f} s" if ms >= 10_000 else f"{int(ms)} ms"


def _masthead(run: dict, rid: str) -> str:
    timing = _dict(run.get("timing"))
    finish = run.get("finish_reason")
    rows = [
        ("run id", rid, ""),
        ("timestamp", run.get("created_at") or "?", ""),
        ("model", run.get("model") or "?", ""),
        ("substrate", run.get("substrate") or "?", ""),
        ("source", " · ".join(str(x) for x in (run.get("source"), run.get("client")) if x) or "?", ""),
        ("duration", _duration(timing) or "?", ""),
    ]
    if finish:
        rows.append(("stop", str(finish) + (" — tape ran out (token cap)" if finish == "length" else ""),
                     "warn" if finish == "length" else ""))
    if run.get("error"):
        rows.append(("error", str(run.get("error")), "warn"))
    meta = "".join(f'<div><span class="k">{_esc(k)}</span>'
                   f'<span class="v{" " + cls if cls else ""}">{_esc(v)}</span></div>'
                   for k, v, cls in rows)
    return (
        '<header class="mod">'
        '<div class="mod-h"><span class="led"></span>'
        '<span class="wordmark">cloz<b>n</b></span>'
        '<span class="mast-sub">run receipt</span></div>'
        f'<div class="meta">{meta}</div>'
        '<div class="legend"><span class="tag cap-t">captured</span>'
        '<span>recorded at generation time</span>'
        '<span class="tag der-t">derived</span>'
        '<span>computed afterwards from the record — never assumed</span></div>'
        '</header>')


def _prompt_text(run: dict) -> tuple[str, int]:
    msgs = [m for m in _list(run.get("messages")) if isinstance(m, dict)]
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    prompt = str(user_msgs[-1].get("content", "")) if user_msgs else str(run.get("prompt_summary") or "")
    return prompt, max(0, len(msgs) - (2 if user_msgs else 0))


def _phosphor(trace: dict, response: str) -> str:
    """The reply as phosphor text: per-token opacity = recorded confidence; dotted underline below
    LOW_CONF. Falls back to the plain response string when the run carries no per-token trace."""
    tokens = _list(trace.get("tokens"))
    if not tokens:
        note = ('<div class="conf-legend">no per-token trace captured on this run — '
                'reply shown without confidence shading</div>')
        return f'<div class="crt-text">{_esc(response)}</div>' + note
    confidence = _list(trace.get("confidence"))
    spans, truncated = [], False
    for i, tok in enumerate(tokens):
        if i >= MAX_TOKENS:
            truncated = True
            break
        piece = str(tok)
        if piece == "":
            continue
        c = _float(confidence[i]) if i < len(confidence) else None
        if c is None:
            spans.append(f'<span class="tk" style="opacity:.8">{_esc(piece)}</span>')
            continue
        c = min(1.0, max(0.0, c))
        opacity = 0.35 + 0.65 * c
        lo = " lo" if c < LOW_CONF else ""
        spans.append(f'<span class="tk{lo}" style="opacity:{opacity:.2f}" '
                     f'title="conf {c:.2f}">{_esc(piece)}</span>')
    body = f'<div class="crt-text">{"".join(spans)}</div>'
    legend = ('<div class="conf-legend">phosphor brightness = recorded token confidence · '
              f'dotted = below {LOW_CONF:.2f}</div>')
    if truncated:
        legend += (f'<div class="conf-legend">token stream truncated for card size — '
                   f'{MAX_TOKENS} of {len(tokens)} tokens shown</div>')
    return body + legend


def _exchange(run: dict, trace: dict, rid: str) -> str:
    prompt, earlier = _prompt_text(run)
    parts = [f'<div class="well">{_esc(prompt)}</div>']
    if earlier > 0:
        parts.append(f'<div class="turnnote">… {earlier} earlier message'
                     f'{"" if earlier == 1 else "s"} not shown (full record in the export)</div>')
    crt = ('<div class="crt"><span class="crt-st">replay</span>'
           f'<span class="crt-id">{_esc(rid)}</span>'
           f'{_phosphor(trace, str(run.get("response") or ""))}</div>')
    parts.append(crt)
    tag = '<span class="tag cap-t">captured</span>'
    return _mod("blue", "the exchange", tag, "".join(parts))


def _influence_label(inf) -> str:
    inf = _dict(inf)
    txt = inf.get("text")
    if txt:
        return str(txt)
    if inf.get("card_id"):
        return f"memory card {inf['card_id']}"
    if inf.get("dial"):
        return f"dial · {inf['dial']}"
    if inf.get("memory_off"):
        return "all memory off"
    if inf.get("behavior_off"):
        return "all dials off"
    return "influence"


def _dep_bars(top_dependent: list) -> str:
    rows = [d for d in _list(top_dependent) if isinstance(d, dict)]
    if not rows:
        return ""
    max_abs = max((abs(_float(d.get("delta")) or 0.0) for d in rows), default=0.0) or 1.0
    cells = []
    for d in rows:
        delta = _float(d.get("delta")) or 0.0
        width = min(100.0, abs(delta) / max_abs * 100.0)
        neg = " neg" if delta < 0 else ""
        cells.append(f'<span class="p">{_esc(d.get("piece"))}</span>'
                     f'<span class="d">{_num(delta, 3, signed=True)}</span>'
                     f'<span class="bar{neg}"><i style="width:{width:.0f}%"></i></span>')
    return '<div class="dep">' + "".join(cells) + "</div>"


def _receipt_row(r: dict) -> str:
    inf = _influence_label(r.get("influence"))
    forced = r if r.get("mode") == "forced" else _dict(r.get("forced"))
    has_effect = bool(r.get("has_effect"))
    cv = r.get("causal_verified")
    if r.get("mode") == "forced":
        eff_chip = ('<span class="chip eff">leaning detected</span>' if has_effect
                    else '<span class="chip noeff">no measurable leaning</span>')
    else:
        eff_chip = ('<span class="chip eff">changed the answer</span>' if has_effect
                    else '<span class="chip noeff">answer unchanged</span>')
    cv_chip = ('<span class="chip cv">causal · verified</span>' if cv is True else
               '<span class="chip">causal_verified: false</span>' if cv is False else
               '<span class="chip">causal_verified: null</span>')
    sum_nats = forced.get("sum_nats", r.get("sum_nats"))
    nats = (f'<span class="nats">Σ {_num(sum_nats, 3, signed=True)} nats</span>'
            if _float(sum_nats) is not None else "")
    bars = _dep_bars(forced.get("top_dependent", r.get("top_dependent")))
    return (f'<div class="rrow"><div class="r-inf">{_esc(inf)}</div>'
            f'<div class="r-chips">{eff_chip}{cv_chip}{nats}</div>{bars}</div>')


def _receipt_rows(receipts_obj: dict) -> list:
    rows = [r for r in _list(receipts_obj.get("receipts")) if isinstance(r, dict)]
    rows += [r for r in _list(receipts_obj.get("forced_receipts")) if isinstance(r, dict)]
    return rows


def _active_influences_line(bundle: dict) -> str:
    """One CAPTURED context line: what was logged as active on the run. Active is not proof — only a
    computed receipt below may claim effect (mirrors explain.py's causal_verified:null invariant)."""
    active = _dict(_dict(bundle.get("explain")).get("influences_active"))
    cards = [c for c in _list(active.get("cards")) if isinstance(c, dict)]
    if not cards:
        cards = [{"text": t} for t in _list(_dict(bundle.get("memory")).get("cards_applied"))]
    dials = [d for d in _list(active.get("dials")) if isinstance(d, dict)]
    bits = [str(_dict(c).get("text") or c) for c in cards]
    bits += [f"dial {d.get('name')}={d.get('value')}" for d in dials if d.get("name")]
    if not bits:
        return ""
    inner = " · ".join(_esc(b) for b in bits)
    return (f'<div class="active-line"><span class="tag cap-t">captured</span> '
            f'active this run (active ≠ causal): {inner}</div>')


def _receipts_section(bundle: dict) -> str:
    receipts_obj = _dict(bundle.get("receipts"))
    rows = _receipt_rows(receipts_obj) if receipts_obj else []
    active = _active_influences_line(bundle)
    if not rows:
        tag = '<span class="tag der-t">derived · on demand</span>'
        body = active + f'<div class="absent">{_esc(_ABSENT_RECEIPTS)}</div>'
        return _mod("lilac", "influences & receipts", tag, body)
    when = receipts_obj.get("computed_at") or "on demand"
    tag = (f'<span class="tag der-t">derived — computed {_esc(when)} by leave-one-out + '
           'forced scoring</span>')
    body = active + "".join(_receipt_row(r) for r in rows)
    skipped = _list(receipts_obj.get("skipped"))
    if skipped:
        body += (f'<div class="small-note">{len(skipped)} influence'
                 f'{"" if len(skipped) == 1 else "s"} skipped — reasons in the JSON export</div>')
    return _mod("lilac", "influences & receipts", tag, body)


def _lens_section(bundle: dict) -> str:
    readouts = [r for r in _list(bundle.get("workspace_readouts")) if isinstance(r, dict)]
    if not readouts:
        tag = '<span class="tag der-t">derived · on demand</span>'
        return _mod("blue", "lens readouts", tag, f'<div class="absent">{_esc(_ABSENT_LENS)}</div>')
    by_layer: dict = {}
    for r in readouts:
        by_layer.setdefault(r.get("layer"), []).append(r)
    groups = []
    for layer in sorted(by_layer, key=lambda x: (x is None, x)):
        chips = []
        for r in by_layer[layer][:40]:
            tops = [str(_dict(t).get("label") or "") for t in _list(r.get("top_readouts"))[:2]]
            tops = [t for t in tops if t.strip()]
            if not tops:
                continue
            tok = str(r.get("token_text") or "")
            chips.append(f'<span class="jchip">{_esc(tok)} → {_esc(", ".join(tops))}</span>')
        if chips:
            label = f"layer {layer}" if layer is not None else "layer ?"
            provider = by_layer[layer][0].get("provider") or by_layer[layer][0].get("provider_type") or ""
            groups.append(f'<div class="jgroup"><div class="jlayer">{_esc(label)}'
                          f'{" · " + _esc(provider) if provider else ""}</div>'
                          f'<div class="jchips">{"".join(chips)}</div></div>')
    if not groups:
        tag = '<span class="tag der-t">derived · on demand</span>'
        return _mod("blue", "lens readouts", tag, f'<div class="absent">{_esc(_ABSENT_LENS)}</div>')
    first = readouts[0]
    provenance = str(first.get("provenance") or first.get("note") or "")
    if not provenance:
        provenance = (_JLENS_CAPTION if str(first.get("provider_type") or "") == "jacobian_lens"
                      else f"workspace readout — provider {first.get('provider_type') or 'unknown'}")
    tag = '<span class="tag der-t">derived — lens readout</span>'
    body = "".join(groups) + f'<div class="provenance">{_esc(provenance)}</div>'
    return _mod("blue", "lens readouts", tag, body)


def _tree_lines(node: dict, rid: str, depth: int, out: list) -> None:
    if depth > 10 or len(out) >= 100 or not isinstance(node, dict):
        return
    nid = str(node.get("id") or "?")
    label = str(node.get("change_label") or "")
    mark = " ◀ this run" if node.get("id") == rid or node.get("is_current") else ""
    prefix = ("  " * depth + "└ ") if depth else ""
    line = f"{prefix}<b>{_esc(nid)}</b>"
    if label:
        line += f" · {_esc(label)}"
    line += _esc(mark)
    out.append(line)
    for child in _list(node.get("children")):
        _tree_lines(child, rid, depth + 1, out)


def _lineage_section(bundle: dict, run: dict, rid: str) -> str:
    tag = '<span class="tag cap-t">captured</span>'
    lineage = _dict(bundle.get("lineage"))
    tree = _dict(lineage.get("tree"))
    if tree:
        lines: list = []
        _tree_lines(tree, rid, 0, lines)
        if len(lines) > 1 or tree.get("children"):
            return _mod("lilac", "lineage", tag, f'<div class="tree">{"<br>".join(lines)}</div>')
    parent = run.get("parent_run_id")
    if parent:
        body = (f'<div class="tree"><b>{_esc(parent)}</b> · parent<br>'
                f'└ <b>{_esc(rid)}</b> ◀ this run</div>')
        return _mod("lilac", "lineage", tag, body)
    return _mod("lilac", "lineage", tag,
                '<div class="absent">no lineage — an original run '
                '(no parent, no recorded branches)</div>')


def _footer(rid: str) -> str:
    return (f'<footer class="mod"><div class="foot">'
            f'<span class="credo">{_esc(_FOOTER_CREDO)}</span>'
            f'<span class="flowbar"></span>'
            f'<span class="wordmark" style="font-size:12px">cloz<b>n</b></span>'
            f'<span class="rid">{_esc(rid)}</span></div></footer>')


# ------------------------------------------------------------------------------------------------ API
def render_card(bundle: dict) -> str:
    """One export bundle -> one self-contained HTML receipt card (a string). Never raises on missing
    fields: every section degrades to its honest-absence copy."""
    bundle = _dict(bundle)
    run = _dict(bundle.get("run"))
    trace = _dict(bundle.get("trace")) or _dict(run.get("trace"))
    rid = str(run.get("id") or "unknown-run")
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>clozn — run receipt · {_esc(rid)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        '<main class="card">',
        _masthead(run, rid),
        _exchange(run, trace, rid),
        _receipts_section(bundle),
        _lens_section(bundle),
        _lineage_section(bundle, run, rid),
        _footer(rid),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)
