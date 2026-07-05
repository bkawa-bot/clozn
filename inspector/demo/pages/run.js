/* run.js -- the Run Inspector (issues C1-C3, C4 repair shell, F2 replay-compare + save-fix, G2 quick-repair).
 * Route #/run/<id>; app.js passes the run id as render()'s 3rd arg. Three columns:
 *   Transcript / token-timeline | Influence (memory + dials + model) | Repair (replay actions).
 * Reads GET /runs/<id>. The Repair buttons POST /runs/<id>/replay and degrade gracefully until issue F1
 * (the replay engine) exists -- so they light up automatically once it lands.
 *
 * F2 (replay compare + save-fix) and G2 (quick-repair presets) all funnel through ONE replay path
 * (doReplay) and ONE result renderer (renderCompare), so every trigger -- the classic replay buttons,
 * the quick-repair presets, anything future -- shows the same Original|Replayed diff and the same
 * "Save this fix" affordance. Nothing here ever throws: the local postJSON resolves {ok,status,data}
 * on every outcome (offline included), and every branch renders a friendly note instead.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;
  var API = location.origin.startsWith("http") ? "" : "http://127.0.0.1:8090";

  var esc = function (s) { return (s == null ? "" : String(s)).replace(/[&<>]/g, function (m) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]; }); };
  var bar = function (c) { var n = Math.round(Math.max(0, Math.min(1, c || 0)) * 8); return "█".repeat(n) + "░".repeat(8 - n); };

  // Per-axis safe caps, mirrored from research/steering.py AXES (cognitive axes degenerate past these).
  // Used only for pre-flight "after" hints in the change summary; the BACKEND is the real authority and
  // its returned behavior.active_dials always wins when present.
  var AXIS_MAX = { warm: 1.5, concise: 1.5, formal: 1.5, playful: 1.5, curious: 1.5, poetic: 1.5, technical: 1.5, candid: 0.45, confident: 1.0, concrete: 0.5 };
  var NUDGE_STEP = 0.5;   // matches replay.NUDGE_STEP; a preset bumps its dial this far toward its + pole.
  var capAxis = function (name, v) { var mx = AXIS_MAX[name] == null ? 1.5 : AXIS_MAX[name]; return Math.max(-mx, Math.min(mx, v)); };

  async function getJSON(p) { var r = await fetch(API + p); if (!r.ok) throw new Error(p + " -> " + r.status); return r.json(); }
  // Local postJSON: NEVER throws. Resolves {ok, status, data} on every outcome -- HTTP error, offline,
  // bad JSON -- so callers can distinguish 404/503 (endpoint absent) from a real failure, yet stay safe.
  async function postJSON(p, b) {
    try {
      var r = await fetch(API + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
      return { ok: r.ok, status: r.status, data: await r.json().catch(function () { return {}; }) };
    } catch (e) {
      return { ok: false, status: 0, data: {} };   // offline / network error -> status 0, handled as a friendly note.
    }
  }

  function ristyle() {
    if (document.getElementById("ri-style")) return;
    var s = document.createElement("style");
    s.id = "ri-style";
    s.textContent =
      ".ri-head{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;margin-bottom:14px}" +
      ".ri-flags{display:flex;gap:6px}" +
      ".ri-flag{font-size:10px;border:1px solid var(--line,#e3e6ef);border-radius:8px;padding:1px 6px;color:var(--soft,#5a6072)}" +
      ".ri-flag.warn{color:#c0603a;border-color:#e7c3ac}" +
      ".ri-cols{display:grid;grid-template-columns:1.3fr 1fr .9fr;gap:16px;align-items:start}" +
      "@media(max-width:1000px){.ri-cols{grid-template-columns:1fr}}" +
      ".ri-col{background:#fff;border:1px solid var(--line,#e3e6ef);border-radius:14px;padding:14px}" +
      ".ri-col h3{margin:0 0 10px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--faint,#9aa0b3)}" +
      ".ri-turn{margin:8px 0;font-size:13.5px;line-height:1.45;white-space:pre-wrap}" +
      ".ri-turn .who{font-size:11px;color:var(--faint,#9aa0b3);text-transform:uppercase;letter-spacing:.04em}" +
      ".ri-turn.assistant{background:var(--wash,#f6f8ff);border-radius:10px;padding:8px 11px}" +
      // --- C2 token timeline: the response as its stream of tokens, each tinted by confidence; the
      //     unsure ones are warm-tinted + underlined branch points you can click open for the alts. ---
      ".ri-tl{margin-top:14px}" +
      ".ri-tl-h{font-size:11px;color:var(--faint,#9aa0b3);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}" +
      ".ri-tl-legend{display:flex;flex-wrap:wrap;gap:5px 12px;align-items:center;font-size:11px;color:var(--soft,#5a6072);margin-bottom:9px}" +
      ".ri-tl-legend .sw{display:inline-flex;align-items:center;gap:4px}" +
      ".ri-tl-legend .chip{width:16px;height:11px;border-radius:3px;display:inline-block}" +
      ".ri-tl-legend .chip.sure{background:var(--ink,#1b1f2a)}" +
      ".ri-tl-legend .chip.mid{background:rgba(27,31,42,.42)}" +
      ".ri-tl-legend .chip.low{background:rgba(192,96,58,.16);box-shadow:inset 0 -2px 0 #c0603a}" +
      ".ri-tl-stream{white-space:pre-wrap;word-break:break-word;line-height:1.85;font-size:13.5px}" +
      // each token: opacity carries confidence (set inline); low ones get the warm underline + a marker.
      ".ri-tk{border-radius:3px;cursor:pointer;transition:background .12s,box-shadow .12s;padding:0 .5px}" +
      ".ri-tk:hover{background:rgba(122,167,255,.14)}" +
      ".ri-tk.low{color:#a8481f;box-shadow:inset 0 -2px 0 rgba(192,96,58,.55);background:rgba(192,96,58,.06)}" +
      ".ri-tk.low:hover{background:rgba(192,96,58,.13)}" +
      ".ri-tk.open{background:rgba(122,167,255,.18)}" +
      ".ri-tk.low.open{background:rgba(192,96,58,.16)}" +
      ".ri-tk .mk{font-size:9px;vertical-align:super;color:#c0603a;opacity:.8}" +   // the branch-point dot
      // the click-to-open detail: this token's confidence + what it almost said, as small prob bars.
      ".ri-tk-pop{display:block;margin:5px 0 7px;border:1px solid var(--line,#e3e6ef);border-left:3px solid var(--halo,#7aa7ff);" +
      "border-radius:8px;padding:7px 10px;background:#fff;font-size:12px;white-space:normal;line-height:1.5;box-shadow:0 4px 14px rgba(120,150,210,.10)}" +
      ".ri-tk-pop.low{border-left-color:#c0603a}" +
      ".ri-tk-pop .hd{color:var(--soft,#5a6072);margin-bottom:5px}" +
      ".ri-tk-pop .hd b{color:var(--ink,#1b1f2a)}" +
      ".ri-tk-pop .alts{margin-top:4px}" +
      ".ri-tk-pop .arow{display:flex;align-items:center;gap:7px;margin:3px 0}" +
      ".ri-tk-pop .apiece{min-width:74px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:var(--ink,#1b1f2a)}" +
      ".ri-tk-pop .atrack{flex:1;height:7px;border-radius:5px;background:var(--wash,#eef1fb);overflow:hidden;min-width:36px}" +
      ".ri-tk-pop .afill{height:100%;background:var(--halo,#7aa7ff);border-radius:5px}" +
      ".ri-tk-pop .aprob{min-width:34px;text-align:right;color:var(--faint,#9aa0b3);font-size:11px}" +
      ".ri-tk-pop .none{color:var(--faint,#9aa0b3)}" +
      ".ri-tl-note{margin-top:9px;font-size:11.5px;color:var(--faint,#9aa0b3);line-height:1.45}" +
      ".ri-kv{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin:5px 0}" +
      ".ri-kv .k{color:var(--faint,#9aa0b3)}" +
      ".ri-card{border:1px solid var(--line,#e3e6ef);border-radius:9px;padding:7px 9px;margin:6px 0;font-size:12.5px}" +
      ".ri-dial{display:inline-block;font-size:11px;border:1px solid var(--line,#e3e6ef);border-radius:8px;padding:1px 7px;margin:2px 3px 0 0}" +
      ".ri-act{display:block;width:100%;text-align:left;border:1px solid var(--line,#e3e6ef);background:#fff;border-radius:10px;padding:8px 11px;margin:6px 0;font:inherit;font-size:13px;cursor:pointer}" +
      ".ri-act:hover{border-color:var(--halo,#7aa7ff)}" +
      ".ri-act.busy{opacity:.6;cursor:default}" +
      ".ri-out{margin-top:10px;font-size:12.5px}" +
      ".ri-out .diff{background:var(--wash,#f6f8ff);border-radius:9px;padding:8px 10px;margin-top:6px;white-space:pre-wrap}" +
      // --- F2 compare: Original | Replayed side by side (stacks under ~560px) + a one-line change summary ---
      ".ri-cmp-sum{font-size:12px;color:var(--soft,#5a6072);margin:2px 0 8px;line-height:1.4}" +
      ".ri-cmp-sum b{color:var(--ink,#1b1f2a);font-weight:640}" +
      ".ri-cmp{display:grid;grid-template-columns:1fr 1fr;gap:9px;align-items:start}" +
      "@media(max-width:560px){.ri-cmp{grid-template-columns:1fr}}" +
      ".ri-cmp .side{min-width:0}" +
      ".ri-cmp .lbl{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3);margin-bottom:3px}" +
      ".ri-cmp .side.now .lbl{color:var(--halo,#7aa7ff)}" +
      ".ri-cmp .body{background:var(--wash,#f6f8ff);border:1px solid var(--line,#e3e6ef);border-radius:9px;padding:8px 10px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.5;color:var(--ink,#1b1f2a)}" +
      ".ri-cmp .side.now .body{border-color:rgba(122,167,255,.4)}" +
      // --- F2 save-fix: a persist affordance under the diff (turns the tried change into the new default) ---
      ".ri-save{margin-top:10px}" +
      ".ri-save-btn{display:inline-block;border:1px solid rgba(122,167,255,.4);background:rgba(122,167,255,.10);color:#3b485a;" +
      "border-radius:10px;padding:6px 12px;font:inherit;font-size:12.5px;font-weight:600;cursor:pointer}" +
      ".ri-save-btn:hover{border-color:var(--halo,#7aa7ff);background:rgba(122,167,255,.16)}" +
      ".ri-save-btn.busy{opacity:.6;cursor:default}" +
      ".ri-save-note{margin-top:7px;font-size:12px;color:var(--soft,#5a6072);line-height:1.45}" +
      ".ri-save-note.ok{color:#2f8a54}" +
      ".ri-save-note.warn{color:#a9762a}" +
      ".ri-save-note .li{display:block;margin:2px 0}" +
      // --- G2 quick-repair: a row of one-click complaint->dial presets above the manual replay buttons ---
      ".ri-qr-h{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3);margin:2px 0 6px}" +
      ".ri-qr{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}" +
      ".ri-qr-btn{font:inherit;font-size:12px;border:1px solid var(--line,#e3e6ef);background:rgba(255,255,255,.7);color:var(--soft,#5a6072);" +
      "border-radius:16px;padding:5px 11px;cursor:pointer;transition:background .14s,border-color .14s,color .14s}" +
      ".ri-qr-btn:hover{color:var(--ink,#1b1f2a);border-color:var(--halo,#7aa7ff)}" +
      ".ri-qr-btn.busy{opacity:.6;cursor:default}" +
      // --- #6 time-travel: "rewind & branch from here" -- a turn picker + optional alt-user field. ---
      ".ri-tt-h{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3);margin:2px 0 4px}" +
      ".ri-tt-sub{font-size:11.5px;color:var(--soft,#5a6072);line-height:1.4;margin-bottom:7px}" +
      ".ri-tt-turns{display:flex;flex-direction:column;gap:5px}" +
      ".ri-tt-turn{display:flex;gap:7px;align-items:baseline;border:1px solid var(--line,#e3e6ef);border-radius:10px;padding:6px 9px;background:rgba(255,255,255,.7)}" +
      ".ri-tt-turn .tn{font-size:10px;letter-spacing:.03em;color:var(--faint,#9aa0b3);text-transform:uppercase;flex:none}" +
      ".ri-tt-turn .tu{flex:1;min-width:0;font-size:12.5px;color:var(--ink,#1b1f2a);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}" +
      ".ri-tt-turn .tgo{flex:none;font:inherit;font-size:11.5px;font-weight:600;border:1px solid rgba(122,167,255,.45);color:#2f4a7a;" +
      "background:rgba(122,167,255,.10);border-radius:14px;padding:3px 10px;cursor:pointer}" +
      ".ri-tt-turn .tgo:hover:not(:disabled){background:rgba(122,167,255,.18);border-color:var(--halo,#7aa7ff)}" +
      ".ri-tt-turn .tgo.busy,.ri-tt-turn .tgo:disabled{opacity:.6;cursor:default}" +
      ".ri-tt-alt{margin:8px 0 2px;display:flex;flex-direction:column;gap:5px}" +
      ".ri-tt-alt label{font-size:11px;color:var(--soft,#5a6072)}" +
      ".ri-tt-alt textarea{font:inherit;font-size:12.5px;border:1px solid var(--line,#e3e6ef);border-radius:9px;padding:6px 8px;resize:vertical;min-height:34px;color:var(--ink,#1b1f2a)}" +
      ".ri-tt-alt textarea:focus{outline:none;border-color:var(--halo,#7aa7ff)}" +
      ".ri-tt-note{font-size:11px;color:var(--faint,#9aa0b3);line-height:1.4;margin-top:6px}" +
      // "propose a memory from this run" -- its own action + result block, kept distinct from replay.
      ".ri-sep{margin:12px 0 2px;border:0;border-top:1px dashed var(--line,#e3e6ef)}" +
      ".ri-prop{margin-top:8px;font-size:12.5px}" +
      ".ri-prop .card{background:var(--wash,#f6f8ff);border:1px solid var(--line,#e3e6ef);border-radius:10px;padding:9px 11px;margin-top:6px}" +
      ".ri-prop .card .txt{color:var(--ink,#1b1f2a);font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word}" +
      ".ri-prop .chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}" +
      ".ri-prop .chip{font-size:10.5px;letter-spacing:.02em;padding:2px 8px;border-radius:9px;white-space:nowrap;color:var(--soft,#5a6072);background:#fff;border:1px solid var(--line,#e3e6ef)}" +
      ".ri-prop .chip.pending{color:#a9762a;background:rgba(230,196,120,.16);border-color:rgba(230,196,120,.42)}" +
      ".ri-prop .chip.risk{color:#c0504a;background:rgba(231,120,120,.12);border-color:rgba(231,120,120,.4)}" +
      ".ri-prop .warn{display:flex;gap:7px;align-items:flex-start;margin-top:8px;padding:7px 10px;border-radius:9px;font-size:11.5px;line-height:1.4;color:#a33;background:rgba(231,120,120,.12);border:1px solid rgba(231,120,120,.4)}" +
      ".ri-prop .warn b{color:#8f2f2f}" +
      ".ri-prop .added{margin-top:8px;font-size:12px;color:var(--soft,#5a6072)}" +
      ".ri-prop .added a{color:var(--halo,#7aa7ff);text-decoration:none;font-weight:600}" +
      ".ri-prop .added a:hover{text-decoration:underline}" +
      ".ri-prop .sub{color:var(--faint,#9aa0b3)}" +
      // --- "set the dial instead" suggestion: a proposed style preference works better as a tone DIAL. ---
      ".ri-dial{margin-top:9px;padding:10px 12px;border-radius:10px;border:1px solid rgba(122,167,255,.5);" +
      "background:linear-gradient(180deg,rgba(122,167,255,.12),rgba(231,168,196,.08))}" +
      ".ri-dial .dt{display:flex;gap:7px;align-items:flex-start;font-size:12px;line-height:1.45;color:var(--soft,#5a6072)}" +
      ".ri-dial .dt .spark{flex:none;font-size:13px;line-height:1.3}" +
      ".ri-dial .dt b{color:var(--ink,#1b1f2a);font-weight:640}" +
      ".ri-dial .drow{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}" +
      ".ri-dial .dgo{font:inherit;font-size:12px;font-weight:620;padding:6px 12px;border-radius:14px;cursor:pointer;line-height:1;" +
      "border:1px solid rgba(122,167,255,.55);color:#2f4a7a;background:linear-gradient(180deg,#dbe7ff,#cbdcff)}" +
      ".ri-dial .dgo:hover:not(:disabled){background:linear-gradient(180deg,#e3edff,#d6e3ff)}" +
      ".ri-dial .dgo:disabled{opacity:.5;cursor:default}" +
      ".ri-dial .dkeep{font:inherit;font-size:11.5px;padding:6px 11px;border-radius:14px;cursor:pointer;line-height:1;" +
      "color:var(--faint,#9aa0b3);background:none;border:1px dashed var(--line,#e3e6ef)}" +
      ".ri-dial .dkeep:hover:not(:disabled){color:var(--soft,#5a6072);border-color:var(--halo,#7aa7ff)}" +
      ".ri-dial .dkeep:disabled{opacity:.5;cursor:default}" +
      ".ri-dial .dnote{margin-top:8px;font-size:11.5px;line-height:1.45;color:var(--soft,#5a6072)}" +
      ".ri-dial .dnote.ok{color:#2f8a54}" +
      ".ri-dial .dnote.err{color:#a9481f}" +
      ".ri-dial.dismissed{border-color:var(--line,#e3e6ef);background:rgba(120,140,190,.06)}" +
      // --- receipts: the measured-delta strip under a compare + the influence-column prove-it block ---
      ".ri-rcpt{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 8px}" +
      ".ri-rcpt .m{font-size:11px;border:1px solid var(--line,#e3e6ef);border-radius:9px;padding:2px 8px;color:var(--soft,#5a6072);background:#fff}" +
      ".ri-rcpt .m b{color:var(--ink,#1b1f2a);font-weight:640}" +
      ".ri-rcpt .m.up b{color:#2f8a54}" +
      ".ri-rcpt .m.down b{color:#c0603a}" +
      ".ri-rcpt-note{font-size:11px;color:var(--faint,#9aa0b3);margin:0 0 8px;line-height:1.4}" +
      ".ri-prove-h{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3);margin:14px 0 6px}" +
      ".ri-prove-sub{font-size:11.5px;color:var(--soft,#5a6072);line-height:1.45;margin-bottom:6px}" +
      // --- M5 Explain tab: a read-only, pre-assembled summary of M1's /runs/<id>/explain -- confidence
      //     hesitations, active influences (tagged "was active", never "caused" -- causal_verified is only
      //     ever null from this endpoint; M2's on-demand ablation is what could someday flip it), and the
      //     concepts note. Additive to (never replacing) the Detail tab's per-influence "prove it" receipts
      //     above. Detail stays the default tab, so nothing about the existing view changes unasked. ---
      ".ri-tabs{display:flex;gap:8px;margin:2px 0 16px}" +
      ".ri-tab{font:inherit;font-size:12.5px;font-weight:640;border:1px solid var(--line,#e3e6ef);background:rgba(255,255,255,.7);color:var(--soft,#5a6072);border-radius:16px;padding:6px 14px;cursor:pointer;transition:background .14s,border-color .14s,color .14s}" +
      ".ri-tab:hover{color:var(--ink,#1b1f2a);border-color:var(--halo,#7aa7ff)}" +
      ".ri-tab.active{color:#2f4a7a;border-color:var(--halo,#7aa7ff);background:rgba(122,167,255,.14)}" +
      ".ri-xpl-wrap h3{margin-bottom:4px}" +
      ".ri-xpl-tagline{font-size:11.5px;color:var(--faint,#9aa0b3);line-height:1.4;margin-bottom:14px}" +
      ".ri-xpl-sec{margin:16px 0}" +
      ".ri-xpl-sec h4{margin:0 0 4px;font-size:12.5px;letter-spacing:.02em;color:var(--ink,#1b1f2a);font-weight:640}" +
      ".ri-xpl-sec h4 .sub{font-weight:400;margin-left:2px}" +
      ".ri-xpl-sub{font-size:12px;color:var(--soft,#5a6072);margin-bottom:8px}" +
      ".ri-xpl-mo{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12.5px}" +
      ".ri-xpl-mo .tok{min-width:60px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,Consolas,monospace;color:var(--ink,#1b1f2a)}" +
      ".ri-xpl-mo .track{letter-spacing:-1px;color:var(--halo,#7aa7ff)}" +
      ".ri-xpl-mo .val{color:var(--faint,#9aa0b3);min-width:32px}" +
      ".ri-xpl-alts{font-size:11.5px;color:var(--faint,#9aa0b3);margin:0 0 7px 4px}" +
      ".ri-xpl-tag{display:inline-block;font-size:10px;letter-spacing:.03em;text-transform:uppercase;color:#2f4a7a;background:rgba(122,167,255,.14);border:1px solid rgba(122,167,255,.4);border-radius:7px;padding:1px 6px;margin-right:6px;vertical-align:1px}" +
      ".ri-xpl-quote{margin-top:3px;font-size:12px;color:var(--soft,#5a6072);font-style:italic}" +
      ".ri-xpl-quote.sub{font-style:normal}" +
      ".ri-xpl-span{margin:4px 0;font-size:12.5px}" +
      ".ri-xpl-feat{display:inline-block;font-size:11px;border:1px solid var(--line,#e3e6ef);border-radius:8px;padding:1px 7px;margin:2px 4px 0 0;color:var(--soft,#5a6072)}" +
      // --- M4 narrate: "why did it say this?" -- the accountable-self narration, lazy/on-demand (a button,
      //     since /narrate generates two model calls, unlike the free M1 panels above). Flags are rendered
      //     as visually distinct warning rows -- the caught confabulations -- never folded into the "why". ---
      ".ri-nrt{margin-top:14px}" +
      ".ri-nrt-why{font-size:13.5px;line-height:1.55;color:var(--ink,#1b1f2a);white-space:pre-wrap}" +
      ".ri-nrt-flags-h{margin-top:12px;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3)}" +
      ".ri-nrt-flags{margin-top:5px}" +
      ".ri-nrt-flag{display:flex;gap:7px;align-items:flex-start;margin:5px 0;padding:7px 10px;border-radius:9px;font-size:12.5px;line-height:1.45;color:#8f2f2f;background:rgba(231,120,120,.12);border:1px solid rgba(231,120,120,.4)}" +
      ".ri-nrt-flags-none{margin-top:10px}" +
      ".ri-nrt-note{margin-top:12px;font-size:11.5px;color:var(--faint,#9aa0b3);line-height:1.45;font-style:italic}";
    document.head.appendChild(s);
  }

  var LOW_CONF = 0.5;   // below this a token is "unsure" -> a highlighted branch point (matches CLI cmd_trace).

  // Whitespace-preserving escape for a token piece rendered inline in a `white-space:pre-wrap` stream:
  // keep the real spaces/newlines (so the response reads naturally) but neutralize markup.
  function escTok(s) { return String(s == null ? "" : s).replace(/[&<>]/g, function (m) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]; }); }
  // A caret-visible label for a piece in the pop / alternatives list (spaces/newlines shown, never blank).
  function labelTok(s) {
    var t = String(s == null ? "" : s).replace(/\n/g, "\\n").replace(/\t/g, "\\t");
    return t === "" ? "∅" : (t.replace(/ /g, "·"));   // middle-dot for spaces so a lone-space token is legible
  }
  // Confidence -> text opacity in [0.5, 1]: sure tokens are full ink; less-sure ones visibly fade (but stay
  // readable). Low-confidence tokens also pick up the .low warm underline, so faintness never stands alone.
  function confOpacity(c) { return (0.5 + 0.5 * Math.max(0, Math.min(1, c))).toFixed(3); }

  function transcriptCol(run) {
    var msgs = run.messages || [], h = "<h3>Transcript</h3>";
    for (var i = 0; i < msgs.length; i++) {
      var cls = msgs[i].role === "assistant" ? "assistant" : "";
      h += '<div class="ri-turn ' + cls + '"><div class="who">' + esc(msgs[i].role) + "</div>" + esc(msgs[i].content) + "</div>";
    }
    var lastA = msgs.length && msgs[msgs.length - 1].role === "assistant";
    if (run.response && !lastA) h += '<div class="ri-turn assistant"><div class="who">assistant</div>' + esc(run.response) + "</div>";
    h += tokenTimeline(run);
    return h;
  }

  // C2: the token timeline. When run.trace carries a non-empty tokens[] array, render the response as its
  // stream of tokens -- each tinted by confidence (opacity) and, when unsure (<0.5), warm-underlined as a
  // clickable branch point that opens this token's confidence + "almost said" alternatives. When the trace
  // is empty/absent (e.g. the HF chat path), the plain transcript above stands and we add a subtle note.
  function tokenTimeline(run) {
    var tr = run.trace || {}, toks = tr.tokens || [];
    if (!toks.length) {
      return '<div class="ri-tl-note">No token trace for this run — showing the response as-is. ' +
        "(Token-by-token confidence is captured on the local engine path, not the hosted chat path.)</div>";
    }
    var conf = tr.confidence || [], lows = 0;
    var h = '<div class="ri-tl"><div class="ri-tl-h">Token timeline — where it was unsure, and what it almost said</div>';
    // compact legend: what the tint + underline mean.
    h += '<div class="ri-tl-legend">' +
      '<span class="sw"><span class="chip sure"></span>confident</span>' +
      '<span class="sw"><span class="chip mid"></span>less sure (fainter)</span>' +
      '<span class="sw"><span class="chip low"></span>unsure — click for alternatives</span>' +
      "</div>";
    h += '<div class="ri-tl-stream">';
    for (var j = 0; j < toks.length; j++) {
      var c = conf[j] == null ? 1 : +conf[j], low = c < LOW_CONF;
      if (low) lows++;
      var piece = escTok(toks[j]);
      // a lone marker so a low-confidence pure-whitespace token is still visibly clickable.
      var mk = low ? '<span class="mk">◆</span>' : "";
      var title = "confidence " + (isFinite(c) ? c.toFixed(2) : "?") + (low ? " — click to see what it almost said" : "");
      h += '<span class="ri-tk' + (low ? " low" : "") + '" data-ti="' + j + '" style="opacity:' + confOpacity(c) + '" title="' + esc(title) + '">' + piece + mk + "</span>";
    }
    h += "</div>";   // .ri-tl-stream
    // a place the click handler injects the open token's detail pop into (kept out of the flowing stream).
    h += '<div class="ri-tl-pop-host"></div>';
    var tail = lows ? " — click the underlined ones to branch" : "";
    h += '<div class="ri-tl-note">' + lows + " uncertain moment" + (lows === 1 ? "" : "s") + tail + "</div>";
    h += "</div>";   // .ri-tl
    return h;
  }

  // Build the detail-pop HTML for token `j`: its confidence bar + the "almost said" alternatives as small
  // prob bars. Safe for a missing/empty alternatives entry (many tokens, and whole runs, have none).
  function tokenPopHTML(run, j) {
    var tr = run.trace || {}, conf = tr.confidence || [], alts = tr.alternatives || [];
    var c = conf[j] == null ? 1 : +conf[j], low = c < LOW_CONF;
    var a = Array.isArray(alts[j]) ? alts[j] : [];
    var h = '<div class="ri-tk-pop' + (low ? " low" : "") + '">';
    h += '<div class="hd">token <b>' + esc(labelTok((tr.tokens || [])[j])) + "</b> · confidence <b>" +
      (isFinite(c) ? c.toFixed(2) : "?") + "</b> " + bar(c) + (low ? " · branch point" : "") + "</div>";
    if (a.length) {
      h += '<div class="hd" style="margin:6px 0 2px">almost said</div><div class="alts">';
      a.slice(0, 5).forEach(function (alt) {
        var p = Math.max(0, Math.min(1, +(alt && alt.prob) || 0));
        h += '<div class="arow"><span class="apiece">' + esc(labelTok(alt && alt.piece)) + '</span>' +
          '<span class="atrack"><span class="afill" style="width:' + (p * 100).toFixed(1) + '%"></span></span>' +
          '<span class="aprob">' + p.toFixed(2) + "</span></div>";
      });
      h += "</div>";
    } else {
      h += '<div class="none">no recorded alternatives for this token.</div>';
    }
    h += "</div>";
    return h;
  }

  // Wire the token timeline: click a token to open (toggle) its detail pop; hover already shows the title.
  // Only one pop is open at a time. No-op when there's no timeline (no-trace runs) -- selectors match nothing.
  function wireTimeline(root, run) {
    var host = root.querySelector(".ri-tl-pop-host");
    if (!host) return;
    root.querySelectorAll(".ri-tk[data-ti]").forEach(function (tk) {
      tk.onclick = function () {
        var was = tk.classList.contains("open");
        // collapse whatever was open first (only one pop open at a time).
        root.querySelectorAll(".ri-tk.open").forEach(function (o) { o.classList.remove("open"); });
        host.innerHTML = "";
        if (was) return;   // clicking the already-open token closes it.
        tk.classList.add("open");
        host.innerHTML = tokenPopHTML(run, +tk.dataset.ti);
      };
    });
  }

  // Truncated card text for a per-card receipt button label (whole texts can be a sentence long).
  function cardLabel(c) {
    var t = typeof c === "string" ? c : String((c && c.text) || "");
    return t.length > 42 ? t.slice(0, 40) + "…" : t;
  }

  function influenceCol(run) {
    var mem = run.memory || {}, beh = run.behavior || {}, h = "<h3>What influenced it</h3>";
    h += '<div class="ri-kv"><span class="k">model</span><span>' + esc(run.model || "?") + "</span></div>";
    h += '<div class="ri-kv"><span class="k">backend</span><span>' + esc(run.substrate || "?") + "</span></div>";
    h += '<div class="ri-kv"><span class="k">via</span><span>' + esc(run.source || "?") + " / " + esc(run.client || "?") + "</span></div>";
    var cards = mem.cards_applied || [], strength = mem.strength == null ? 1 : mem.strength;
    // runs recorded post-mode-swap carry memory.mode ("prompt": cards rode as context, applied per turn;
    // "internalized": the trained prefix) and, in prompt mode, applied_ids aligned with cards_applied.
    var modeTag = mem.mode ? " · " + esc(mem.mode) : "";
    var ids = Array.isArray(mem.applied_ids) ? mem.applied_ids : [];
    h += '<div class="ri-kv" style="margin-top:12px"><span class="k">memory (strength ' + (+strength).toFixed(2) + modeTag + ')</span><span>' + cards.length + " card(s)</span></div>";
    for (var i = 0; i < cards.length; i++) h += '<div class="ri-card">' + esc(typeof cards[i] === "string" ? cards[i] : cards[i].text) + "</div>";
    if (!cards.length) {
      // prompt mode records per-turn application, so "none" there means the block wasn't injected on
      // THIS turn (topic-gated out / dial at 0) -- not necessarily that no cards exist.
      h += '<div class="sub">' + (mem.mode === "prompt" ? "no memory applied this turn (block not injected)" : "no memory applied") + "</div>";
    }
    var dials = beh.active_dials || {}, keys = Object.keys(dials);
    h += '<div class="ri-kv" style="margin-top:12px"><span class="k">behavior dials</span><span>' + keys.length + "</span></div>";
    if (keys.length) h += "<div>" + keys.map(function (k) { return '<span class="ri-dial">' + esc(k) + " " + (+dials[k]).toFixed(2) + "</span>"; }).join("") + "</div>";
    else h += '<div class="sub">no dials active</div>';
    // --- RECEIPTS: don't just LIST the influences -- PROVE them. Each button re-runs this exact prompt
    //     with one influence removed (greedy, so the delta is attributable), and renders the measured
    //     difference. Measured, never self-reported: the model cannot reliably narrate what its own
    //     memory/dials do (the self-audit findings), so the ablation is the only honest receipt.
    if (cards.length || keys.length) {
      h += '<div class="ri-prove-h">Receipts — prove it</div>';
      h += '<div class="ri-prove-sub">Re-runs this prompt with one influence off (greedy). The difference <i>is</i> that influence’s contribution — measured, not self-reported.</div>';
      if (cards.length) h += "<button class=\"ri-act\" data-receipt='{\"memory_off\":true,\"greedy\":true}'>Memory receipt — what did the memory change?</button>";
      // per-card receipts (prompt mode only: applied_ids come from the card store, and the replay
      // engine can really rebuild the block minus one card there -- internalized would need a retrain).
      for (var j = 0; j < cards.length && j < ids.length; j++) {
        if (ids[j] == null) continue;
        var spec = esc(JSON.stringify({ disabled_memory_ids: [String(ids[j])], greedy: true })).replace(/'/g, "&#39;");
        h += "<button class=\"ri-act\" data-receipt='" + spec + "'>Card receipt — without “" + esc(cardLabel(cards[j])) + "”</button>";
      }
      if (keys.length) h += "<button class=\"ri-act\" data-receipt='{\"behavior_off\":true,\"greedy\":true}'>Dials receipt — what did the dials change?</button>";
      h += '<div class="ri-out" id="ri-receipt-out"></div>';
    }
    return h;
  }

  // ---------------------------------------------------------------------------------------------------
  // M5 (EXPLAIN_THIS_ANSWER_SPEC.md): a read-only "Explain" tab that pre-assembles M1's already-shipped
  // /runs/<id>/explain into one summary -- confidence hesitations, active influences, concepts -- so a
  // non-power-user gets the story without clicking every receipt button. explainSummaryHTML (+ its three
  // panel helpers) is a pure function -- the parsed /explain object in, an HTML string out -- kept
  // separate from the fetch/DOM plumbing below it, mirroring the CLI's own format_explain(). Distinct
  // from (and additive to) the Detail tab's per-influence "prove it" receipts in influenceCol above.
  // ---------------------------------------------------------------------------------------------------

  // A confidence/score/dial value -> "0.42", tolerating anything non-numeric without throwing.
  function fmt2(v) { var n = +v; return isNaN(n) ? String(v == null ? "?" : v) : n.toFixed(2); }

  function xplConfidenceHTML(conf) {
    conf = conf || {};
    var h = '<div class="ri-xpl-sec"><h4>Confidence <span class="sub">measured per token — never an overall score</span></h4>';
    if (!conf.available) {
      return h + '<div class="sub">not available — ' + esc(conf.note || "no trace on this run") + "</div></div>";
    }
    var moments = Array.isArray(conf.uncertain_moments) ? conf.uncertain_moments : [];
    h += '<div class="ri-xpl-sub">' + esc(conf.summary || "") + " of " + (conf.n_tokens || 0) +
      " tokens (threshold " + esc(conf.threshold) + ")</div>";
    if (!moments.length) return h + '<div class="sub">no hesitations below threshold.</div></div>';
    moments.forEach(function (m) {
      var c = (m && m.confidence == null) ? 0 : +(m && m.confidence);
      h += '<div class="ri-xpl-mo"><span class="tok">' + esc(labelTok(m && m.token)) + "</span>" +
        '<span class="track">' + bar(c) + '</span><span class="val">' + fmt2(c) + "</span></div>";
      var alts = (m && Array.isArray(m.alternatives)) ? m.alternatives : [];
      if (alts.length) {
        h += '<div class="ri-xpl-alts">almost: ' + alts.slice(0, 3).map(function (a) {
          return esc(labelTok(a && a.piece)) + " " + fmt2(a && a.prob);
        }).join("  ") + "</div>";
      }
    });
    return h + "</div>";
  }

  function xplInfluencesHTML(inf) {
    inf = inf || {};
    var h = '<div class="ri-xpl-sec"><h4>Influences active <span class="sub">active this turn — not yet proven causal</span></h4>';
    if (inf.gate != null || inf.mode) {
      h += '<div class="ri-xpl-sub">gate ' + (inf.gate == null ? "?" : fmt2(inf.gate)) + (inf.mode ? " · " + esc(inf.mode) : "") + "</div>";
    }
    var cards = Array.isArray(inf.cards) ? inf.cards : [];
    var dials = Array.isArray(inf.dials) ? inf.dials : [];
    // causal_verified travels with each claim, per the spec -- M1 only ever sets it null ("was active");
    // true/false can only ever come from a future M2 ablation receipt. Never render "caused" from this data.
    var tagOf = function (v) { return v === true ? "proven" : v === false ? "ruled out" : "was active"; };
    if (cards.length) {
      cards.forEach(function (c) {
        c = c || {};
        h += '<div class="ri-card"><span class="ri-xpl-tag">' + esc(tagOf(c.causal_verified)) + "</span>" + esc(c.text || "");
        if (c.quoted_span) h += '<div class="ri-xpl-quote">“' + esc(c.quoted_span) + '”</div>';
        else if (c.note) h += '<div class="ri-xpl-quote sub">' + esc(c.note) + "</div>";
        h += "</div>";
      });
    } else {
      h += '<div class="sub">' + esc(inf.note || "no memory applied") + "</div>";
    }
    if (dials.length) {
      h += '<div style="margin-top:8px">' + dials.map(function (d) {
        d = d || {};
        return '<span class="ri-dial">' + esc(d.name) + " " + fmt2(d.value) + " · " + esc(tagOf(d.causal_verified)) + "</span>";
      }).join("") + "</div>";
    } else {
      h += '<div class="sub">no dials active</div>';
    }
    return h + "</div>";
  }

  function xplConceptsHTML(conc) {
    conc = conc || {};
    var h = '<div class="ri-xpl-sec"><h4>Concepts</h4>';
    if (!conc.available) {
      return h + '<div class="sub">not available — ' + esc(conc.note || "concept readout needs the engine") + "</div></div>";
    }
    var spans = Array.isArray(conc.spans) ? conc.spans : [];
    if (!spans.length) return h + '<div class="sub">(no spans recorded)</div></div>';
    spans.forEach(function (span) {
      span = span || {};
      var feats = Array.isArray(span.features) ? span.features : [];
      h += '<div class="ri-xpl-span">' + (span.piece != null ? "<b>" + esc(labelTok(span.piece)) + "</b> " : "") +
        feats.map(function (f) {
          f = f || {};
          return '<span class="ri-xpl-feat">' + esc(f.label || f.id || "?") + " " + fmt2(f.score) + "</span>";
        }).join(" ") + "</div>";
    });
    return h + "</div>";
  }

  // The pure assembler: the parsed /explain object -> the whole summary body (all three panels).
  function explainSummaryHTML(expl) {
    expl = expl || {};
    return xplConfidenceHTML(expl.confidence) + xplInfluencesHTML(expl.influences_active) + xplConceptsHTML(expl.concepts);
  }

  // Wraps the local postJSON envelope ({ok,status,data}) around explainSummaryHTML with the same honest
  // 404/503/offline degradation language used throughout this file (doReplay et al.) -- never a raw error.
  function explainPanelHTML(r) {
    if (!r) return '<div class="sub">couldn’t load the explanation.</div>';
    if (r.status === 404) return '<div class="sub">explain isn’t available on this build yet — this lights up once /runs/&lt;id&gt;/explain exists.</div>';
    if (r.status === 503) return '<div class="sub">explain needs the studio substrate running — start it and reload.</div>';
    if (r.status === 0) return '<div class="sub">couldn’t reach the studio (offline?) — reload once it’s up.</div>';
    if (!r.ok || !r.data || r.data.error) {
      var msg = r.data && r.data.error ? " — " + esc(r.data.error) : "";
      return '<div class="sub">explain failed (' + esc(r.status) + ")" + msg + "</div>";
    }
    return explainSummaryHTML(r.data);
  }

  // ---------------------------------------------------------------------------------------------------
  // M4 (EXPLAIN_THIS_ANSWER_SPEC.md): the accountable-self narration -- a receipt-CONSTRAINED "why",
  // diffed against an independent judge and flagged wherever it overclaims. Unlike M1's /explain above
  // (free, fetched eagerly), POST /runs/<id>/narrate GENERATES: two model calls (the constrained narration
  // + the unconstrained confabulation sample it is diffed against -- the latter is never itself returned;
  // narrate.py's structural trap guard). So this is lazy: a button (wireNarrate below), never auto-fetched.
  //
  // narrateHTML is a pure assembler -- the /narrate response object in, an HTML string out (mirrors
  // explainSummaryHTML above) -- and renders ONLY the four keys narrate() returns: the constrained
  // narration prose (the "why"), `flags` as visually distinct WARNING rows (the caught confabulations --
  // each names a claim the model was about to make that no receipt backs), and `note` (which matcher ran +
  // its honesty caveat) as a muted caption. `unsupported_claims` is not re-rendered separately: every one
  // of its entries is already represented, verbatim, as a string in `flags`. There is no raw/unconstrained
  // "why" to accidentally render -- that text is never a key in this object at all, by construction.
  // ---------------------------------------------------------------------------------------------------

  function narrateHTML(obj) {
    obj = (obj && typeof obj === "object") ? obj : {};
    var cn = (obj.constrained_narration && typeof obj.constrained_narration === "object") ? obj.constrained_narration : {};
    var narration = typeof cn.narration === "string" ? cn.narration.trim() : "";
    var flags = Array.isArray(obj.flags) ? obj.flags : [];
    var note = typeof obj.note === "string" ? obj.note : "";

    var h = '<div class="ri-nrt">';
    h += '<div class="ri-nrt-why">' + (narration
      ? esc(narration)
      : '<span class="sub">no receipt-backed narration was produced for this reply — with a thin or empty record, that’s a complete and honest answer, not a failure.</span>') + "</div>";
    if (flags.length) {
      h += '<div class="ri-nrt-flags-h">Caught in the diff — claimed with no receipt to back it</div><div class="ri-nrt-flags">';
      flags.forEach(function (f) { h += '<div class="ri-nrt-flag">⚠ ' + esc(String(f)) + "</div>"; });
      h += "</div>";
    } else {
      h += '<div class="ri-nrt-flags-none sub">no unsupported claims flagged this time.</div>';
    }
    if (note) h += '<div class="ri-nrt-note">' + esc(note) + "</div>";
    h += "</div>";
    return h;
  }

  // The dials that were live on the ORIGINAL run -- the baseline a quick-repair preset nudges from.
  function runDials(run) { return ((run.behavior || {}).active_dials) || {}; }

  // --- #6 time-travel: fold a run's flat messages[] into conversational TURNS the UI can rewind to.
  // Mirrors research/timetravel.message_turns EXACTLY (a turn = a user msg + the assistant reply that
  // followed; a leading system message rides along; a final user with no reply is a dangling turn).
  // The BACKEND is the authority (it re-folds server-side); this is only to render the branch buttons.
  function messageTurns(run) {
    var msgs = (run && run.messages) || [], turns = [], cur = null;
    for (var i = 0; i < msgs.length; i++) {
      var role = msgs[i] && msgs[i].role, content = (msgs[i] && msgs[i].content) || "";
      if (role === "user") {
        if (cur) turns.push(cur);
        cur = { turn: turns.length, user: content, assistant: null };
      } else if (role === "assistant" && cur && cur.assistant == null) {
        cur.assistant = content;
      }
    }
    if (cur) turns.push(cur);
    return turns;
  }

  // A short, caret-safe preview of a turn's user text for the branch-button label.
  function turnLabel(t) {
    var u = String((t && t.user) || "").replace(/\s+/g, " ").trim();
    return u.length > 46 ? u.slice(0, 44) + "…" : (u || "(empty)");
  }

  function repairCol(run) {
    var h = "<h3>Repair / replay</h3>";
    // --- G2: quick-repair presets. One click maps a common complaint to a dial nudge + replay, then
    //     shows the before/after via the SAME compare renderer. These carry data-qr (NOT data-rep) so
    //     wireRepair's `.ri-act[data-rep]` selector never binds them as raw replay buttons.
    h += '<div class="ri-qr-h">Quick repair &mdash; one click, then compare</div>';
    h += '<div class="ri-qr">';
    for (var i = 0; i < QUICK_REPAIRS.length; i++) {
      var q = QUICK_REPAIRS[i];
      h += '<button class="ri-qr-btn" data-qr="' + esc(q.key) + '" title="' + esc(q.title) + '">' + esc(q.label) + "</button>";
    }
    h += "</div>";
    h += '<hr class="ri-sep">';
    // --- classic manual replay buttons (F1). Unchanged selector (data-rep) so they keep working. ---
    h += "<button class=\"ri-act\" data-rep='{\"memory_off\":true}'>Replay with memory OFF</button>";
    h += "<button class=\"ri-act\" data-rep='{\"behavior_off\":true}'>Replay with dials OFF (neutral)</button>";
    h += "<button class=\"ri-act\" data-rep='{\"nudge\":\"concise\"}'>Make it more concise + replay</button>";
    h += "<button class=\"ri-act\" data-rep='{\"plain\":true}'>Replay unchanged (re-roll)</button>";
    h += '<div class="ri-out" id="ri-out"></div>';
    // --- #6: rewind & branch from here. Pick a turn to rewind to; the branch RE-GENERATES from the
    //     truncated transcript (optionally with a different question at that turn) and is recorded as a
    //     CHILD run. The KV cache makes this byte-exact + cheap (kv_timetravel_findings.md); state-surgery
    //     is Lab-only (half-life < 1 turn). Uses the SAME compare renderer as replay. ---
    var turns = messageTurns(run);
    if (turns.length) {
      h += '<hr class="ri-sep">';
      h += '<div class="ri-tt-h">Rewind &amp; branch from here</div>';
      h += '<div class="ri-tt-sub">Rewind to a turn and re-generate from there — as-is (re-roll) or with a different question. Recorded as a child run; the branched future is byte-exact for a fresh recompute.</div>';
      h += '<div class="ri-tt-alt"><label for="ri-tt-alt-in">Optional: ask something different at the branch turn</label>' +
        '<textarea id="ri-tt-alt-in" placeholder="leave blank to re-roll the turn unchanged"></textarea></div>';
      h += '<div class="ri-tt-turns">';
      for (var ti = 0; ti < turns.length; ti++) {
        h += '<div class="ri-tt-turn"><span class="tn">turn ' + turns[ti].turn + '</span>' +
          '<span class="tu" title="' + esc(turns[ti].user) + '">' + esc(turnLabel(turns[ti])) + '</span>' +
          '<button class="tgo" data-branch="' + turns[ti].turn + '">Branch</button></div>';
      }
      h += "</div>";
      h += '<div class="ri-out" id="ri-tt-out"></div>';
      h += '<div class="ri-tt-note">A branch keeps the current memory &amp; dials; toggle those with the replay buttons above.</div>';
    }
    // --- E2: propose a durable memory from this run (drops a PENDING card on the Memory page) ---
    h += '<hr class="ri-sep">';
    h += '<button class="ri-act" id="ri-propose">Propose a memory from this run</button>';
    h += '<div class="ri-prop" id="ri-prop"></div>';
    return h;
  }

  // ---------------------------------------------------------------------------------------------------
  // F2 core: ONE replay path (doReplay) + ONE result renderer (renderCompare). Everything -- the manual
  // buttons, the G2 presets -- calls doReplay, so every result gets the same Original|Replayed diff and
  // the same "Save this fix" affordance. Nothing throws.
  // ---------------------------------------------------------------------------------------------------

  // Fire a replay of `run` under `changes`, render the compare into `out`, restore the trigger button.
  // `btn` is optional (the element to disable/busy while in flight). Returns a Promise (never rejects).
  function doReplay(out, run, changes, btn) {
    if (!out) return Promise.resolve();
    var restore = null;
    if (btn) {
      var label = btn.textContent;
      btn.classList.add("busy"); btn.disabled = true;
      restore = function () { btn.classList.remove("busy"); btn.disabled = false; btn.textContent = label; };
    }
    out.innerHTML = '<span class="sub">replaying&hellip;</span>';
    return postJSON("/runs/" + run.id + "/replay", { changes_applied: changes }).then(function (r) {
      if (restore) restore();
      // Distinguish the "not wired / unreachable" cases from a real server error, and never throw.
      if (r.status === 404) {
        out.innerHTML = '<span class="sub">replay isn’t wired yet (issue F1) — this lights up once /runs/&lt;id&gt;/replay exists.</span>';
        return;
      }
      if (r.status === 503) {
        out.innerHTML = '<span class="sub">replay needs the qwen substrate running — start it (or switch to qwen) and try again.</span>';
        return;
      }
      if (r.status === 0) {
        out.innerHTML = '<span class="sub">couldn’t reach the studio (offline?) — nothing was changed. Try again once it’s up.</span>';
        return;
      }
      if (!r.ok || !r.data || !r.data.id) {
        var msg = r.data && r.data.error ? " — " + esc(r.data.error) : "";
        out.innerHTML = '<span class="sub">replay failed (' + esc(r.status) + ")" + msg + "</span>";
        return;
      }
      renderCompare(out, run, r.data);
    }, function () {
      // defensive: postJSON shouldn't reject, but never leave the trigger stuck if it somehow does.
      if (restore) restore();
      out.innerHTML = '<span class="sub">replay failed (unexpected error) — nothing was changed.</span>';
    });
  }

  // --- #6 branch: POST /runs/<id>/branch {turn, alt_user}, render the compare via the SAME renderer. --
  // The child's changes_applied carries {branch_turn, edited_user, alt_user?, kv_snapshot}; renderCompare
  // + summarizeChanges below understand it. Original = the parent run's reply; Replayed = the branched
  // reply. Never throws (postJSON resolves {ok,status,data} on every outcome).
  function doBranch(out, run, turn, altUser, btn) {
    if (!out) return Promise.resolve();
    var restore = null;
    if (btn) {
      var label = btn.textContent;
      btn.classList.add("busy"); btn.disabled = true;
      restore = function () { btn.classList.remove("busy"); btn.disabled = false; btn.textContent = label; };
    }
    out.innerHTML = '<span class="sub">rewinding to turn ' + esc(turn) + ' &amp; branching&hellip;</span>';
    var body = { turn: turn };
    if (altUser != null && String(altUser).trim()) body.alt_user = String(altUser);
    return postJSON("/runs/" + run.id + "/branch", body).then(function (r) {
      if (restore) restore();
      if (r.status === 404) { out.innerHTML = '<span class="sub">branch isn’t wired yet — this lights up once /runs/&lt;id&gt;/branch exists.</span>'; return; }
      if (r.status === 503) { out.innerHTML = '<span class="sub">branching needs the qwen substrate running — start it (or switch to qwen) and try again.</span>'; return; }
      if (r.status === 0) { out.innerHTML = '<span class="sub">couldn’t reach the studio (offline?) — nothing was changed.</span>'; return; }
      if (!r.ok || !r.data || !r.data.id) {
        var msg = r.data && r.data.error ? " — " + esc(r.data.error) : "";
        out.innerHTML = '<span class="sub">branch failed (' + esc(r.status) + ")" + msg + "</span>";
        return;
      }
      renderCompare(out, run, r.data);
    }, function () {
      if (restore) restore();
      out.innerHTML = '<span class="sub">branch failed (unexpected error) — nothing was changed.</span>';
    });
  }

  // --- the RECEIPT: transparent, client-computed deltas between original and replayed ----------------
  // Three honest numbers (no model judges anything): word count, words/sentence, and % of wording changed
  // (1 - Jaccard overlap of word sets). Every replay path gets this strip for free via renderCompare.
  function wordsOf(s) { return String(s || "").toLowerCase().match(/[a-z0-9']+/g) || []; }
  function sentCount(s) { var m = String(s || "").split(/[.!?]+/).filter(function (x) { return x.trim(); }); return Math.max(1, m.length); }
  function receiptMetrics(orig, repl) {
    var ow = wordsOf(orig), rw = wordsOf(repl);
    var oset = {}, rset = {}, inter = 0, uni = 0, k;
    ow.forEach(function (w) { oset[w] = 1; });
    rw.forEach(function (w) { rset[w] = 1; });
    for (k in oset) { uni++; if (rset[k]) inter++; }
    for (k in rset) { if (!oset[k]) uni++; }
    return {
      words: [ow.length, rw.length],
      wps: [Math.round(ow.length / sentCount(orig) * 10) / 10, Math.round(rw.length / sentCount(repl) * 10) / 10],
      changed: uni ? Math.round((1 - inter / uni) * 100) : 0
    };
  }
  function receiptStrip(orig, repl, changes) {
    var m = receiptMetrics(orig, repl);
    var d = m.words[1] - m.words[0];
    var chip = function (label, from, to, delta) {
      var cls = delta === 0 ? "" : (delta < 0 ? " down" : " up");
      var arrow = delta === 0 ? "±0" : (delta > 0 ? "+" + delta : String(delta));
      return '<span class="m' + cls + '">' + label + " <b>" + from + " → " + to + "</b> (" + arrow + ")</span>";
    };
    var h = '<div class="ri-rcpt">' +
      chip("words", m.words[0], m.words[1], d) +
      chip("words/sentence", m.wps[0], m.wps[1], Math.round((m.wps[1] - m.wps[0]) * 10) / 10) +
      '<span class="m">wording changed <b>' + m.changed + "%</b></span>" +
      "</div>";
    var greedy = changes && changes.greedy;
    h += '<div class="ri-rcpt-note">' + (greedy
      ? "Measured by ablation, greedy decode — the difference above is attributable to the change, not sampling."
      : "Sampled replay — differences include sampling noise; re-run to confirm, or use a receipt button (greedy).") +
      "</div>";
    return h;
  }

  // Render the Original vs Replayed comparison + change summary + a Save-Fix affordance into `out`.
  // `child` is the child run returned by /runs/<id>/replay (its changes_applied / behavior are authoritative).
  function renderCompare(out, run, child) {
    var changes = child.changes_applied || {};
    var orig = run.response != null && run.response !== "" ? run.response : (run.response_summary || "(no original response captured)");
    var repl = child.response != null && child.response !== "" ? child.response : (child.response_summary || "(no reply)");
    var html =
      '<div class="ri-cmp-sum">Changed: <b>' + summarizeChanges(changes, child) + "</b></div>" +
      receiptStrip(orig, repl, changes) +
      '<div class="ri-cmp">' +
        '<div class="side"><div class="lbl">Original</div><div class="body">' + esc(orig) + "</div></div>" +
        '<div class="side now"><div class="lbl">Replayed</div><div class="body">' + esc(repl) + "</div></div>" +
      "</div>" +
      '<div class="ri-save"></div>';
    out.innerHTML = html;
    renderSaveFix(out.querySelector(".ri-save"), changes, child);
  }

  // A one-line, human summary of what the replay changed, derived from changes_applied (+ the child's
  // effective dials when they pin the actual post-cap value). Kept short: "memory off", "concise 0.80", ...
  function summarizeChanges(changes, child) {
    changes = changes || {};
    var eff = ((child || {}).behavior || {}).active_dials || {};
    var parts = [];
    if (changes.memory_off) parts.push("memory off");
    if (Array.isArray(changes.disabled_memory_ids) && changes.disabled_memory_ids.length) {
      parts.push("memory disabled (" + changes.disabled_memory_ids.length + ")");
    }
    if (changes.behavior_off) parts.push("dials neutralized");
    if (changes.behavior_overrides && typeof changes.behavior_overrides === "object") {
      Object.keys(changes.behavior_overrides).forEach(function (k) {
        // prefer the effective (post-cap) value the backend actually applied; fall back to the request.
        var v = eff[k] != null ? eff[k] : changes.behavior_overrides[k];
        parts.push(esc(k) + " " + (+v).toFixed(2));
      });
    }
    if (changes.nudge) {
      var nk = String(changes.nudge);
      var nv = eff[nk] != null ? " → " + (+eff[nk]).toFixed(2) : "";
      parts.push(esc(nk) + " up" + nv);
    }
    // #6 branch: rewound to a turn, optionally with a different question there.
    if (changes.branch_turn != null) {
      var bp = "branched from turn " + changes.branch_turn;
      if (changes.edited_user) bp += " (edited question)";
      if (changes.kv_snapshot) bp += " · KV snapshot";
      parts.push(bp);
    }
    if (changes.plain) parts.push("nothing (re-roll, same settings)");
    if (!parts.length) parts.push("nothing (re-roll, same settings)");
    return parts.join(", ");
  }

  // --- F2 save-fix -----------------------------------------------------------------------------------
  // Which persist calls make this replay's change the new default? Returns a list of {endpoint, body, label}
  // ops, plus `unpersistable` reasons for parts of the change that have no matching endpoint (so we can say
  // so honestly instead of pretending). Mapping:
  //   memory_off / behavior_off  -> not a single persistable target (they suppress everything for one turn;
  //                                 there's no "turn all memory/dials off forever" endpoint) -> explained.
  //   disabled_memory_ids[]      -> POST /memory/disable {id} for each id.
  //   behavior_overrides{k:v}    -> POST /steer/set {name:k, value:v} for each dial (v = effective/post-cap).
  //   nudge:name                 -> POST /steer/set {name, value:effective}  (the bumped, capped value).
  //   plain / edited_memory      -> nothing to persist (pure re-roll / unwired edit) -> explained.
  function persistOps(changes, child) {
    changes = changes || {};
    var eff = ((child || {}).behavior || {}).active_dials || {};
    var ops = [], unpersistable = [];

    if (Array.isArray(changes.disabled_memory_ids) && changes.disabled_memory_ids.length) {
      changes.disabled_memory_ids.forEach(function (id) {
        ops.push({ endpoint: "/memory/disable", body: { id: id }, label: "disabled memory " + id });
      });
    }
    if (changes.behavior_overrides && typeof changes.behavior_overrides === "object") {
      Object.keys(changes.behavior_overrides).forEach(function (k) {
        var v = eff[k] != null ? +eff[k] : +changes.behavior_overrides[k];
        ops.push({ endpoint: "/steer/set", body: { name: k, value: v }, label: k + " set to " + v.toFixed(2) });
      });
    }
    if (changes.nudge) {
      var nk = String(changes.nudge);
      // the bumped value the backend actually landed on (post per-axis cap); needed so Save reproduces it.
      var nv = eff[nk] != null ? +eff[nk] : capAxis(nk, NUDGE_STEP);
      ops.push({ endpoint: "/steer/set", body: { name: nk, value: nv }, label: nk + " set to " + nv.toFixed(2) });
    }

    // changes that deliberately can't map to a "new default" endpoint -> report, don't offer a button.
    if (changes.branch_turn != null) unpersistable.push("A branch is a conversation fork, saved as its own child run — there’s no default to set. Open it from the Runs list to keep exploring.");
    if (changes.memory_off) unpersistable.push("Memory OFF is a one-turn suppression; to make it permanent, disable specific cards on the Memory page.");
    if (changes.behavior_off) unpersistable.push("Dials OFF is a one-turn neutral; to make it permanent, zero the dials on the Behavior page.");
    if (changes.edited_memory) unpersistable.push("Memory-card editing isn’t wired to persist yet.");
    if (changes.plain && !ops.length) unpersistable.push("This was an unchanged re-roll — there’s nothing to save.");

    return { ops: ops, unpersistable: unpersistable };
  }

  // Render the Save-Fix affordance under a compare. Shows a button only when the change maps to a persist
  // endpoint; otherwise a short note explaining why it can't be saved (never a dead button).
  function renderSaveFix(host, changes, child) {
    if (!host) return;
    var plan = persistOps(changes, child);
    if (!plan.ops.length) {
      var why = plan.unpersistable.length
        ? plan.unpersistable.join(" ")
        : "This change can’t be saved as a default (nothing persistable to apply).";
      host.innerHTML = '<div class="ri-save-note warn">' + esc(why) + "</div>";
      return;
    }
    // one button persists the whole plan; any unpersistable remainder is noted alongside.
    var extra = plan.unpersistable.length ? '<div class="ri-save-note warn" style="margin-top:6px">' + esc(plan.unpersistable.join(" ")) + "</div>" : "";
    host.innerHTML = '<button class="ri-save-btn">Save this fix</button><div class="ri-save-note"></div>' + extra;
    var btn = host.querySelector(".ri-save-btn"), note = host.querySelector(".ri-save-note");
    btn.onclick = function () {
      if (btn.disabled) return;
      btn.disabled = true; btn.classList.add("busy");
      var was = btn.textContent; btn.textContent = "saving…";
      note.className = "ri-save-note"; note.innerHTML = "";
      // fire the ops sequentially (order-independent, but simplest to reason about) and tally outcomes.
      var done = [], failed = [];
      var chain = Promise.resolve();
      plan.ops.forEach(function (op) {
        chain = chain.then(function () {
          return postJSON(op.endpoint, op.body).then(function (r) {
            var okBody = r.data && (r.data.ok === false) ? false : true;   // /memory/disable can 200 with {ok:false}
            if (r.ok && okBody && r.status !== 404 && r.status !== 503 && r.status !== 0) done.push(op.label);
            else failed.push({ op: op, r: r });
          });
        });
      });
      chain.then(function () {
        btn.classList.remove("busy"); btn.textContent = was;
        if (!failed.length) {
          note.className = "ri-save-note ok";
          note.innerHTML = "Saved: " + done.map(function (d) { return '<span class="li">✓ ' + esc(d) + "</span>"; }).join("");
        } else if (!done.length) {
          btn.disabled = false;   // total failure -> let them retry
          note.className = "ri-save-note warn";
          note.innerHTML = "Couldn’t save (studio offline or endpoint unavailable) — nothing was changed." + failNote(failed);
        } else {
          btn.disabled = false;   // partial -> allow a retry of the whole plan
          note.className = "ri-save-note warn";
          note.innerHTML = "Saved: " + done.map(function (d) { return '<span class="li">✓ ' + esc(d) + "</span>"; }).join("") +
            '<span class="li">Some parts didn’t save — try again.</span>' + failNote(failed);
        }
      }, function () {
        btn.classList.remove("busy"); btn.disabled = false; btn.textContent = was;
        note.className = "ri-save-note warn";
        note.innerHTML = "Couldn’t save (unexpected error) — nothing was changed.";
      });
    };
  }

  // A compact "(2 endpoints unreachable)"-style tail so a failure isn't totally opaque, without dumping raw errors.
  function failNote(failed) {
    if (!failed.length) return "";
    var codes = failed.map(function (f) { return f.r.status === 0 ? "offline" : f.r.status; });
    return '<span class="li sub">(' + esc(codes.join(", ")) + ")</span>";
  }

  // ---------------------------------------------------------------------------------------------------
  // G2 quick-repair presets. Each maps a plain-language complaint to a single dial pushed toward its +
  // pole by NUDGE_STEP relative to the run's CURRENT value (backend re-caps per axis). We send an explicit
  // behavior_overrides so the intent is legible in changes_applied ("concise 0.80"), and the compare shows
  // before/after. `candid` is the direct opposite of "agreeable"; `concrete` of "vague"; `warm` of "cold".
  // ---------------------------------------------------------------------------------------------------
  var QUICK_REPAIRS = [
    { key: "verbose",   label: "Too verbose",   axis: "concise",  title: "Push the concise↔verbose dial toward concise (+0.5) and replay." },
    { key: "vague",     label: "Too vague",     axis: "concrete", title: "Push the concrete↔abstract dial toward concrete (+0.5) and replay." },
    { key: "agreeable", label: "Too agreeable", axis: "candid",   title: "Push the candid↔agreeable dial toward candid (+0.5) and replay." },
    { key: "cold",      label: "Too cold",      axis: "warm",     title: "Push the warm↔detached dial toward warm (+0.5) and replay." }
  ];
  var QR_BY_KEY = {};
  QUICK_REPAIRS.forEach(function (q) { QR_BY_KEY[q.key] = q; });

  // Build the behavior_overrides for a preset: current dial value + NUDGE_STEP, capped to the axis max
  // (belt-and-suspenders; the backend caps too). Returns e.g. { concise: 0.8 }.
  function presetOverrides(preset, run) {
    var cur = +(runDials(run)[preset.axis] || 0);
    var target = capAxis(preset.axis, cur + NUDGE_STEP);
    var o = {};
    o[preset.axis] = target;
    return o;
  }

  function wireQuickRepair(root, run) {
    var out = root.querySelector("#ri-out");
    root.querySelectorAll(".ri-qr-btn[data-qr]").forEach(function (btn) {
      btn.onclick = function () {
        if (btn.disabled) return;
        var preset = QR_BY_KEY[btn.dataset.qr];
        if (!preset || !out) return;
        doReplay(out, run, { behavior_overrides: presetOverrides(preset, run) }, btn);
      };
    });
  }

  // #6: wire the "Branch" buttons. Each carries data-branch=<turn>; the optional alt-user comes from the
  // shared textarea. All funnel through doBranch -> renderCompare (the same compare + receipt strip as
  // replay). Scoped by data-branch so nothing else here binds them.
  function wireBranch(root, run) {
    var out = root.querySelector("#ri-tt-out");
    var altEl = root.querySelector("#ri-tt-alt-in");
    root.querySelectorAll(".tgo[data-branch]").forEach(function (btn) {
      btn.onclick = function () {
        if (btn.disabled || !out) return;
        var turn = parseInt(btn.dataset.branch, 10);
        if (isNaN(turn)) return;
        doBranch(out, run, turn, altEl ? altEl.value : "", btn);
      };
    });
  }

  function wireRepair(root, run) {
    // Only the manual replay buttons carry data-rep; scope by it so neither the E2 "propose" button nor
    // the G2 quick-repair chips (they carry data-qr) get a replay handler bound here. All roads lead to
    // doReplay -> renderCompare, so these buttons now show the F2 compare + Save-Fix too.
    root.querySelectorAll(".ri-act[data-rep]").forEach(function (btn) {
      btn.onclick = function () {
        var out = root.querySelector("#ri-out"), changes = {};
        try { changes = JSON.parse(btn.dataset.rep || "{}"); } catch (e) { changes = {}; }
        doReplay(out, run, changes, btn);
      };
    });
  }

  // The influence-column receipt buttons: same replay path, greedy, rendered into the influence column so
  // the proof sits next to the claim it proves.
  function wireReceipts(root, run) {
    var out = root.querySelector("#ri-receipt-out");
    root.querySelectorAll(".ri-act[data-receipt]").forEach(function (btn) {
      btn.onclick = function () {
        var changes = {};
        try { changes = JSON.parse(btn.dataset.receipt || "{}"); } catch (e) { changes = {}; }
        doReplay(out, run, changes, btn);
      };
    });
  }

  // M5: the Detail/Explain tab switch -- toggle .active + show/hide the two view containers. Detail stays
  // the default (matches querySelectorAll(...).forEach usage elsewhere in this file, e.g. wireQuickRepair).
  function wireTabs(root) {
    var tabs = root.querySelectorAll(".ri-tab[data-tab]");
    var views = { detail: root.querySelector("#ri-view-detail"), explain: root.querySelector("#ri-view-explain") };
    tabs.forEach(function (t) {
      t.onclick = function () {
        tabs.forEach(function (o) { o.classList.remove("active"); });
        t.classList.add("active");
        Object.keys(views).forEach(function (k) { if (views[k]) views[k].style.display = (k === t.dataset.tab) ? "" : "none"; });
      };
    });
  }

  // M4: wire the "Why did it say this?" button. Lazy on purpose (see the narrateHTML block above) -- fires
  // only on click, shows a "thinking…" state while the two generations run, and renders via the pure
  // narrateHTML() assembler on success. Every failure path (404/503/offline/other) gets one honest line,
  // same discipline as doReplay/wirePropose elsewhere in this file -- nothing here ever throws.
  function wireNarrate(root, run) {
    var btn = root.querySelector("#ri-narrate-btn"), out = root.querySelector("#ri-narrate-out");
    if (!btn || !out) return;
    btn.onclick = function () {
      if (btn.disabled) return;
      var label = btn.textContent;
      btn.disabled = true; btn.classList.add("busy"); btn.textContent = "thinking…";
      out.innerHTML = '<span class="sub">thinking… composing a receipt-bound narration (two model calls) — this can take a little while.</span>';
      postJSON("/runs/" + run.id + "/narrate", {}).then(function (r) {
        btn.disabled = false; btn.classList.remove("busy"); btn.textContent = label;
        if (r.status === 404) { out.innerHTML = '<span class="sub">run not found.</span>'; return; }
        if (r.status === 503) { out.innerHTML = '<span class="sub">narration needs the Studio’s qwen substrate running — start it and try again.</span>'; return; }
        if (r.status === 0) { out.innerHTML = '<span class="sub">couldn’t reach the studio (offline?) — nothing was generated.</span>'; return; }
        if (!r.ok || !r.data || r.data.error) {
          var msg = r.data && r.data.error ? " — " + esc(r.data.error) : "";
          out.innerHTML = '<span class="sub">narration failed (' + esc(r.status) + ")" + msg + "</span>";
          return;
        }
        out.innerHTML = narrateHTML(r.data);
      }, function () {
        // defensive: postJSON shouldn't reject, but never leave the button stuck if it somehow does.
        btn.disabled = false; btn.classList.remove("busy"); btn.textContent = label;
        out.innerHTML = '<span class="sub">narration failed (unexpected error) — nothing was generated.</span>';
      });
    };
  }

  // E2: "Propose a memory from this run". Asks the backend to distill a durable preference out of this
  // conversation and drop it into the PENDING memory queue (reviewed on the Memory page). Uses
  // ctx.postJSON(path, body, fallback) which NEVER throws -- on any failure (offline, 404, 500) it
  // resolves to the fallback (null here), so every branch below is safe.
  function wirePropose(root, run, ctx) {
    var btn = root.querySelector("#ri-propose");
    if (!btn) return;
    var out = root.querySelector("#ri-prop");
    btn.onclick = function () {
      if (btn.disabled) return;
      btn.disabled = true;
      btn.classList.add("busy");
      var label = btn.textContent;
      btn.textContent = "reading the conversation…";
      if (out) out.innerHTML = '<span class="sub">reading the conversation… distilling a durable preference from this run.</span>';

      var post = ctx && ctx.postJSON ? ctx.postJSON : postJSON;
      Promise.resolve(post("/runs/" + run.id + "/propose-memory", null, null)).then(function (res) {
        btn.disabled = false;
        btn.classList.remove("busy");
        btn.textContent = label;
        renderProposal(out, res, ctx);
      }, function () {
        // defensive: ctx.postJSON shouldn't reject, but never leave the button stuck if it somehow does.
        btn.disabled = false;
        btn.classList.remove("busy");
        btn.textContent = label;
        renderProposal(out, null, ctx);
      });
    };
  }

  // Render the outcome of a propose-memory call into `out`. Handles all four shapes the endpoint can
  // return, plus null (offline / non-2xx swallowed by ctx.postJSON): proposed card, no-proposal, error.
  // `ctx` (optional) carries the null-safe postJSON used by the "set the dial instead" action.
  function renderProposal(out, res, ctx) {
    if (!out) return;
    // offline / endpoint absent / any swallowed failure -> a friendly, non-alarming note.
    if (res == null || res.ok === false) {
      var reason = res && res.reason ? " (" + esc(res.reason) + ")" : "";
      out.innerHTML = '<span class="sub">Couldn’t propose a memory (endpoint offline?)' + reason +
        " — nothing was changed. This lights up once /runs/&lt;id&gt;/propose-memory exists.</span>";
      return;
    }
    // nothing durable to remember from this run.
    if (res.proposed === false || !res.card) {
      var why = res.reason ? esc(res.reason) : "no durable preference found in this run";
      out.innerHTML = '<span class="sub">No memory proposed — ' + why + ".</span>";
      return;
    }
    // proposed:true with a card -> show the card text, a "pending review" chip, its risk, and a
    // warning if it reads instruction-like (risk != "low"). Point the user to the Memory page.
    var card = res.card || {};
    var text = card.text != null ? String(card.text) : "";
    var risk = card.risk != null ? String(card.risk) : "";
    var riskLow = risk === "" || risk.toLowerCase() === "low" || risk.toLowerCase() === "none";

    var html = '<div class="card">';
    html += '<div class="txt">' + (text ? esc(text) : "(empty proposal)") + "</div>";
    html += '<div class="chips"><span class="chip pending">pending review</span>';
    if (risk) html += '<span class="chip' + (riskLow ? "" : " risk") + '">risk: ' + esc(risk) + "</span>";
    html += "</div>";
    if (!riskLow) {
      html += '<div class="warn"><span>⚠</span><span><b>Looks instruction-like.</b> ' +
        "This reads more like an embedded instruction than a lasting preference — review it carefully before approving.</span></div>";
    }
    // inline nav to #/memory (a plain hash link is enough; the router mounts the Memory page).
    html += '<div class="added">Added to pending. Review it in <a href="#/memory" data-nav="memory">Memory</a>.</div>';
    // "set the dial instead" host: filled below (as DOM) when the backend flags this as a style preference.
    html += '<div class="ri-dial-host"></div>';
    html += "</div>";
    out.innerHTML = html;

    // prefer ctx-style navigation when available, but the href alone already works.
    var link = out.querySelector('a[data-nav="memory"]');
    if (link) link.onclick = function (e) {
      e.preventDefault();
      if (S && typeof S.navigate === "function") S.navigate("memory");
      else location.hash = "#/memory";
    };

    // If the proposal is really a style preference, recommend the tone DIAL over the (weak) memory.
    var sug = dialSuggestionOf(res);
    if (sug) renderDialSuggestion(out.querySelector(".ri-dial-host"), sug, card, out, res, ctx);
  }

  // ---- "set the dial instead" (proposed style preference -> tone dial) -------------------------
  // The backend attaches `dial_suggestion:{axis,value,pole_label}` to propose-memory when the distilled
  // preference is really a tone knob (null/absent otherwise). Read it defensively; a malformed one is
  // treated as absent so the plain proposal card stands.
  function dialSuggestionOf(res) {
    if (!res || typeof res !== "object") return null;
    var d = res.dial_suggestion;
    if (!d || typeof d !== "object") return null;
    if (d.axis == null || d.value == null || d.pole_label == null) return null;
    return { axis: String(d.axis), value: +d.value, pole_label: String(d.pole_label) };
  }

  function fmtDialVal(v) { var n = +v; return isNaN(n) ? String(v) : (n === Math.round(n) ? String(n) : n.toFixed(1)); }

  // Render the recommendation + wire the two buttons into `host` (a child of the proposal `out`). `card`
  // is the just-proposed pending card ({id,...}); we reject it by id when the dial is chosen.
  function renderDialSuggestion(host, sug, card, out, res, ctx) {
    if (!host) return;
    var cardId = card && card.id != null ? card.id : null;
    var valTxt = fmtDialVal(sug.value);

    var box = document.createElement("div");
    box.className = "ri-dial";
    box.innerHTML =
      '<div class="dt"><span class="spark">✦</span><span>This reads like a style preference. The <b>' +
      esc(sug.pole_label) + "</b> dial steers this directly — memory is weak for style. " +
      "Recommended: set the dial to " + esc(valTxt) + " instead.</span></div>" +
      '<div class="drow">' +
      '<button class="dgo">Set the ' + esc(sug.pole_label) + " dial</button>" +
      '<button class="dkeep">keep it as a memory anyway</button>' +
      "</div>" +
      '<div class="dnote" style="display:none"></div>';
    host.appendChild(box);

    var setBtn = box.querySelector(".dgo");
    var keepBtn = box.querySelector(".dkeep");
    var note = box.querySelector(".dnote");

    setBtn.onclick = function () {
      if (setBtn.disabled) return;
      acceptDial(sug, cardId, valTxt, setBtn, keepBtn, note, box, ctx);
    };
    keepBtn.onclick = function () {
      if (keepBtn.disabled) return;
      // dismiss the suggestion; the pending card the propose flow created is left exactly as it is.
      box.className = "ri-dial dismissed";
      setBtn.disabled = true; keepBtn.disabled = true;
      dialNote(note, "Kept as a pending memory — review it on the Memory page.", "");
    };
  }

  function dialNote(note, msg, kind) {
    if (!note) return;
    note.textContent = msg;
    note.className = "dnote" + (kind ? " " + kind : "");
    note.style.display = "";
  }

  // Accept the dial: POST /steer/set, then discard the weak style card via /memory/reject (when it has an
  // id). Uses ctx.postJSON when available (null-safe -> null on any failure), else the local one adapted.
  function acceptDial(sug, cardId, valTxt, setBtn, keepBtn, note, box, ctx) {
    setBtn.disabled = true; keepBtn.disabled = true;
    var was = setBtn.textContent;
    setBtn.textContent = "setting…";
    dialNote(note, "Setting the " + sug.pole_label + " dial…", "");

    // Prefer ctx.postJSON(path, body, fallback) (null on failure); fall back to the local {ok,status,data}
    // postJSON, normalized to the same "null on failure" contract so the branches below are uniform.
    var post = (ctx && ctx.postJSON)
      ? function (p, b) { return ctx.postJSON(p, b, null); }
      : function (p, b) { return postJSON(p, b).then(function (r) { return (r && r.ok && r.status !== 404 && r.status !== 503 && r.status !== 0) ? (r.data || {}) : null; }); };

    Promise.resolve(post("/steer/set", { name: sug.axis, value: sug.value })).then(function (sres) {
      if (sres == null) {
        setBtn.disabled = false; keepBtn.disabled = false; setBtn.textContent = was;
        dialNote(note, "Couldn’t set the dial — is the studio server up? Nothing was changed; the memory is still pending.", "err");
        return;
      }
      if (cardId == null) {
        setBtn.textContent = was;
        box.className = "ri-dial dismissed";
        dialNote(note, "Set the " + sug.pole_label + " dial to " + valTxt + ". (No pending card to discard.)", "ok");
        return;
      }
      Promise.resolve(post("/memory/reject", { id: cardId })).then(function (rres) {
        setBtn.textContent = was;
        box.className = "ri-dial dismissed";
        if (rres == null) {
          dialNote(note, "Set the " + sug.pole_label + " dial to " + valTxt + ", but couldn’t discard the style memory — reject it on the Memory page if you like.", "ok");
        } else {
          dialNote(note, "Set " + sug.pole_label + " dial to " + valTxt + "; discarded the style memory.", "ok");
        }
      });
    });
  }

  async function render(view, ctx, runId) {
    ristyle();
    view.innerHTML = '<div class="wrap"><div class="nav" style="margin-bottom:14px"><a href="#/runs">← Runs</a></div><div id="ri-root"><div class="sub">loading run…</div></div></div>';
    var root = view.querySelector("#ri-root"), run;
    // M1 is free (zero generation -- a read + reshape of the run already being fetched): fire it alongside
    // the run fetch, not lazily after an Explain-tab click, so opening the tab is instant.
    var explainP = postJSON("/runs/" + runId + "/explain");
    try { run = await getJSON("/runs/" + runId); }
    catch (e) { root.innerHTML = '<div class="sub">run not found (' + esc(e.message) + ")</div>"; return; }
    var explainR = await explainP;
    var flags = run.flags || [];
    root.innerHTML =
      '<div class="ri-head"><b>' + esc(run.prompt_summary || "(run)") + "</b>" +
      '<span class="sub">' + esc(run.created_at) + " · " + esc(run.source) + "/" + esc(run.client) + " · " + esc(run.model) + " · " + ((run.timing || {}).duration_ms != null ? run.timing.duration_ms : "?") + "ms</span>" +
      '<span class="ri-flags">' + flags.map(function (f) { return '<span class="ri-flag ' + (["error", "pending-memory", "low-confidence"].indexOf(f) >= 0 ? "warn" : "") + '">' + esc(f) + "</span>"; }).join("") + "</span></div>" +
      '<div class="ri-tabs">' +
        '<button class="ri-tab active" data-tab="detail">Detail</button>' +
        '<button class="ri-tab" data-tab="explain">Explain</button>' +
      "</div>" +
      '<div id="ri-view-detail">' +
        '<div class="ri-cols"><section class="ri-col">' + transcriptCol(run) + '</section><section class="ri-col">' + influenceCol(run) + '</section><section class="ri-col">' + repairCol(run) + "</section></div>" +
      "</div>" +
      '<div id="ri-view-explain" style="display:none">' +
        '<section class="ri-col ri-xpl-wrap"><h3>Explain</h3>' +
        '<div class="ri-xpl-tagline">A read-only summary, assembled from the same measured signals as the Detail tab — never a self-report. Per-influence “prove it” receipts still live over in Detail.</div>' +
        explainPanelHTML(explainR) +
        '<hr class="ri-sep">' +
        '<div class="ri-prove-h">Why did it say this? <span class="sub">(generates — two model calls)</span></div>' +
        '<div class="ri-prove-sub">A receipt-constrained “why”, checked against a separate, unconstrained guess and flagged wherever that guess doesn’t hold up — never the raw self-report itself.</div>' +
        '<button class="ri-act" id="ri-narrate-btn">Why did it say this?</button>' +
        '<div class="ri-out" id="ri-narrate-out"></div>' +
        "</section>" +
      "</div>";
    wireTimeline(root, run);
    wireRepair(root, run);
    wireReceipts(root, run);
    wireQuickRepair(root, run);
    wireBranch(root, run);
    wirePropose(root, run, ctx);
    wireTabs(root);
    wireNarrate(root, run);
  }

  S.register("run", { title: "Run Inspector", render: render });
})();
