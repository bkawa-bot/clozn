/* heavnOS · EDIT — the Glass Edit (pin & resolve), now with a second edit mode. W5 + Route D.
   TWO HONEST EDIT MODES (notes/EDIT_INSTRUCTIONS_DESIGN.md) — the user picks the property they need:
     RESOLVE (diffusion): re-mask selected spans and re-solve them under FULL bidirectional attention —
       the resolve follows your pins and context; it does NOT take instructions (there is no instruction
       channel in a base diffusion model; the honest constraint is displayed). No free text, but genuine
       backward propagation.
     REWRITE (AR): type a free-text instruction; pins become "keep these phrases verbatim" constraints;
       the AR substrate REGENERATES the unpinned text following the instruction. This is regeneration,
       not a bidirectional resolve — no backward propagation, and pin survival is MEASURED afterward
       (checked verbatim against the actual output), never assumed or guaranteed.
   Engine ops:
     RESOLVE — POST /v1/revise {text, spans:[{start,end}] BYTE offsets, steps, grow, ...} — exists ONLY
       on the C++ engine (contracts §20 confirms no studio passthrough today) and ONLY in diffusion mode
       (400 on autoregressive). Transport: try a studio passthrough (POST /engine/revise) first in case
       the backend adds one; else direct fetch to the engine URL (configurable, persisted); every failure
       reported plainly (CORS/AR-mode/down).
     REWRITE — POST /engine/rewrite {text, pins:[{start,end}] CHAR offsets, instruction, max_tokens} — a
       real studio passthrough (clozn/server/routes/rewrite.py), same-origin, no CORS dance needed: it's
       a constrained AR chat call through the existing EngineSubstrate.chat plumbing, zero engine (C++)
       changes. Response includes per-pin {kept} fidelity and the binding honest `note`. */
import { html, useState, useEffect, useRef } from "../vendor/preact-standalone.mjs";
import { useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

/* ── helpers ─────────────────────────────────────────────────────────── */
const enc8 = new TextEncoder();
const byteLen = s => enc8.encode(s).length;
/* JS char offsets -> UTF-8 byte offsets (the engine's span unit) */
function toByteSpan(text, a, b){
  return { start: byteLen(text.slice(0, a)), end: byteLen(text.slice(0, b)) };
}
/* selection offsets within a container that renders `text` across several text nodes */
function selOffsets(container){
  const sel = window.getSelection();
  if(!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
  const range = sel.getRangeAt(0);
  if(!container.contains(range.startContainer) || !container.contains(range.endContainer)) return null;
  const walk = (node, off) => {
    let n = 0;
    const it = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let cur;
    while((cur = it.nextNode())){
      if(cur === node) return n + off;
      n += cur.textContent.length;
    }
    return null;
  };
  const a = walk(range.startContainer, range.startOffset);
  const b = walk(range.endContainer, range.endOffset);
  if(a == null || b == null || a === b) return null;
  return [Math.min(a,b), Math.max(a,b)];
}
const overlaps = (a, b) => a[0] < b[1] && b[0] < a[1];

const ENGINE_URL_KEY = "heavn.engineUrl";
const engineUrl = () => localStorage.getItem(ENGINE_URL_KEY) || "http://127.0.0.1:8080";

async function callRevise(body){
  /* 1) a studio passthrough, if the backend ever adds one (preferred: same-origin).
     NOTE (review finding #1): unmatched studio POST paths return 409 — not 404 — so BOTH
     statuses mean "no passthrough route; fall through to the engine directly". */
  try{
    const r = await fetch("/engine/revise", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if(r.ok) return { via: "studio passthrough", res: await r.json(), err: null };
    if(r.status !== 404 && r.status !== 409)   /* a real passthrough exists and errored */
      return { via: "studio passthrough", res: null, err: (await r.text()).slice(0, 200) };
  }catch(e){ /* fall through */ }
  /* 2) the engine directly (cross-origin — may be blocked by CORS; reported honestly) */
  try{
    const r = await fetch(engineUrl() + "/v1/revise", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return { via: "engine direct", res: r.ok ? await r.json() : null,
             err: r.ok ? null : (await r.text()).slice(0, 200) };
  }catch(e){
    return { via: "engine direct", res: null,
             err: "unreachable (" + String(e).slice(0, 80) + ") — if the engine is up, this is " +
                  "likely CORS: the browser can't cross-origin to the engine port; a studio " +
                  "passthrough route (POST /engine/revise) is the clean fix" };
  }
}

/* REWRITE (Route D): a real studio passthrough exists (unlike revise) -- same-origin, no CORS handling
   needed. Failures reported plainly, same honesty rule as callRevise. */
async function callRewrite(body){
  try{
    const r = await fetch("/engine/rewrite", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => null);
    return r.ok ? { res: j, err: null } : { res: null, err: (j && j.error) || `HTTP ${r.status}` };
  }catch(e){
    return { res: null, err: "unreachable (" + String(e).slice(0, 80) + ")" };
  }
}

/* ── the module ──────────────────────────────────────────────────────── */
export function EditModule(){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);
  const [text, setText] = useState("");
  const [seeded, setSeeded] = useState(null);            // which run seeded the canvas
  const [pins, setPins] = useState([]);                   // [charA, charB] — glass-locked (both modes)
  const [solves, setSolves] = useState([]);               // [charA, charB] — to re-solve (Resolve only)
  const [mode, setMode] = useState(null);                 // engine mode from /engine/health
  const [editMode, setEditMode] = useState("resolve");    // "resolve" (diffusion) | "rewrite" (AR)
  const [instruction, setInstruction] = useState("");     // Rewrite's free-text instruction
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [steps, setSteps] = useState(16);
  const [grow, setGrow] = useState(0);
  const canvasRef = useRef(null);

  /* engine mode gate */
  useEffect(() => { (async () => {
    if(!live){ setMode("offline"); return; }
    const h = await fetch("/engine/health").then(r => r.ok ? r.json() : null).catch(() => null);
    setMode(h && h.engine ? (h.engine.mode || "unknown") : "down");
  })(); }, [live]);

  /* seed from the current run's answer (once per run, never overwriting user edits) */
  useEffect(() => {
    if(rec && rec.response && seeded !== rec.id && !text){
      setText(String(rec.response)); setSeeded(rec.id); setPins([]); setSolves([]); setResult(null);
    }
  }, [rec && rec.id]);

  /* Resolve needs a diffusion GGUF (the bidirectional re-mask); Rewrite needs an autoregressive one
     (the constrained AR chat call) -- the two modes are gated on OPPOSITE engine modes, honestly. */
  const armed = editMode === "resolve" ? mode === "diffusion" : mode === "autoregressive";
  const grab = () => {
    const o = canvasRef.current && selOffsets(canvasRef.current);
    if(!o) toast("select some text on the canvas first");
    return o;
  };
  const addPin = () => { const o = grab(); if(!o) return;
    if(solves.some(s => overlaps(s, o))) return toast("that overlaps a resolve span — a span can't be both");
    if(pins.some(p => overlaps(p, o))) return toast("that overlaps an existing pin");
    setPins(p => [...p, o]); };
  const addSolve = () => { const o = grab(); if(!o) return;
    if(pins.some(p => overlaps(p, o))) return toast("that overlaps a pin — unpin it first");
    if(solves.some(s => overlaps(s, o))) return toast("that overlaps an existing resolve span");
    setSolves(s => [...s, o]); };
  const editAnchor = () => {
    const o = grab(); if(!o) return;
    if(pins.some(p => overlaps(p, o))) return toast("that span is pinned ('will not change') — unpin it before editing");
    const oldPiece = text.slice(o[0], o[1]);
    const next = window.prompt("Replace this anchor (the resolve will accommodate it):", oldPiece);
    if(next == null || next === oldPiece) return;
    const delta = next.length - oldPiece.length;
    const shift = ([a,b]) => [a >= o[1] ? a + delta : a, b > o[1] ? b + delta : b];
    const dropOverlapping = list => list.filter(s => !overlaps(s, o)).map(shift);
    setText(text.slice(0, o[0]) + next + text.slice(o[1]));
    setPins(dropOverlapping(pins)); setSolves(dropOverlapping(solves));
    setResult(null);
    toast("anchor edited — now mark the spans that should re-solve around it");
  };
  const clearMarks = () => { setPins([]); setSolves([]); setResult(null); };

  const doResolve = async () => {
    if(!armed) return toast("the resolve needs a diffusion GGUF on the engine — current mode: " + mode);
    if(!solves.length) return toast("mark at least one span to re-solve (pins alone keep everything)");
    setBusy(true); setResult(null);
    const body = { text, spans: solves.map(([a,b]) => toByteSpan(text, a, b)),
                   steps: +steps || 16, grow: +grow || 0 };
    const out = await callRevise(body);
    setBusy(false);
    if(!out.res){ setResult({ mode: "resolve", ok: false, via: out.via, err: out.err || "no response" }); return; }
    const newText = out.res.choices && out.res.choices[0] && out.res.choices[0].text;
    setResult({ mode: "resolve", ok: true, via: out.via, before: text, after: newText ?? "(no text in response)",
                finish: out.res.choices && out.res.choices[0] && out.res.choices[0].finish_reason,
                usage: out.res.usage || null });
  };

  /* REWRITE (Route D): pins ride as CHAR offsets directly (no byte conversion -- this endpoint is
     Python, not the C++ engine) alongside the free-text instruction. Fidelity is whatever the server
     measured, rendered as-is -- this UI never re-derives or second-guesses the {kept} verdicts. */
  const doRewrite = async () => {
    if(!armed) return toast("rewrite needs an autoregressive GGUF on the engine — current mode: " + mode);
    if(!instruction.trim()) return toast("type an instruction first");
    setBusy(true); setResult(null);
    const body = { text, instruction, pins: pins.map(([a,b]) => ({ start: a, end: b })) };
    const out = await callRewrite(body);
    setBusy(false);
    if(!out.res){ setResult({ mode: "rewrite", ok: false, err: out.err || "no response" }); return; }
    setResult({ mode: "rewrite", ok: true, before: text, after: out.res.text,
                pins: out.res.pins || [], allPinsKept: !!out.res.all_pins_kept,
                note: out.res.note, finish: out.res.finish_reason, runId: out.res.run_id });
  };

  /* render the canvas as segments (pins frosted, solves tinted -- solves only mean something in Resolve
     mode; Rewrite regenerates everything unpinned, so a leftover solve-mark from switching modes would
     be a misleading tint there, and is deliberately not rendered). */
  const segments = (() => {
    const marks = [...pins.map(p => ({ s: p, kind: "pin" })),
                   ...(editMode === "resolve" ? solves.map(s => ({ s, kind: "solve" })) : [])]
      .sort((x,y) => x.s[0] - y.s[0]);
    const out = []; let at = 0;
    for(const m of marks){
      if(m.s[0] > at) out.push({ t: text.slice(at, m.s[0]) });
      out.push({ t: text.slice(m.s[0], m.s[1]), kind: m.kind });
      at = m.s[1];
    }
    if(at < text.length) out.push({ t: text.slice(at) });
    return out.length ? out : [{ t: text }];
  })();

  return html`<div class="col">
    <div class="mod">
      <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
      <div class="mod-h"><span class="led blue"></span><span class="cap">the glass edit — pin ${"&"} ${editMode === "resolve" ? "resolve" : "rewrite"}</span>
        <span class="tail">${mode == null ? "checking engine…" : "engine mode: " + mode}</span>
        <span class=${"tag " + (armed ? "cap-t" : "smp-t")}>${armed ? "ARMED" : "STAGING"}</span></div>

      <div class="transport" style="border-bottom:none;background:none;padding:4px 13px 0">
        <span class="lbl" style="font-size:7.5px">edit mode</span>
        <button class=${"spd" + (editMode === "resolve" ? " busy" : "")}
          onClick=${() => { setEditMode("resolve"); setResult(null); }}>RESOLVE (diffusion)</button>
        <button class=${"spd" + (editMode === "rewrite" ? " busy" : "")}
          onClick=${() => { setEditMode("rewrite"); setResult(null); }}>REWRITE (AR)</button>
      </div>

      ${editMode === "resolve" ? html`<div class="cfg" style="margin:6px 13px 0">
        <span style="font-size:9.5px">the resolve follows your pins and surrounding context —
          <b>it does not take instructions</b>. Pin what must stay; mark what should re-solve;
          edit an anchor and let the change propagate — in both directions.</span>
      </div>` : html`<div class="cfg" style="margin:6px 13px 0">
        <span style="font-size:9.5px"><b>rewrite regenerates the unpinned text — it does not
          bidirectionally resolve</b> (no backward propagation, unlike Resolve). Pin what must survive
          verbatim, type an instruction, and the AR model regenerates the rest. Pin survival is
          <b>checked against the actual output</b>, never assumed — a broken pin is reported, not hidden.</span>
      </div>`}

      ${!armed && mode != null && html`<div class="cfg" style="margin:6px 13px 0;border-left-color:var(--lilac)">
        <span style="font-size:9.5px">${
          mode === "offline" ? "server offline — the canvas below still works as staging"
          : mode === "down" ? "engine unreachable — staging only until it's up"
          : editMode === "resolve" && mode === "autoregressive" ? "the engine has an AR model loaded; pin-resolve needs a DIFFUSION GGUF (Dream/LLaDA) — everything below stages until then"
          : editMode === "rewrite" && mode === "diffusion" ? "the engine has a diffusion model loaded; rewrite needs an AUTOREGRESSIVE GGUF — everything below stages until then (switch to RESOLVE to use this model)"
          : "engine mode unknown — staging only"}</span></div>`}

      <div style="padding:10px 13px 4px">
        <div ref=${canvasRef} style="min-height:110px;max-height:260px;overflow-y:auto;
             border:1px solid var(--edge);border-radius:9px;padding:11px 13px;
             background:linear-gradient(180deg,#fff,#EFF6FB);font-family:var(--mono);
             font-size:11.5px;line-height:1.9;color:var(--navy);white-space:pre-wrap;word-break:break-word">
          ${text ? segments.map(seg =>
              seg.kind === "pin" ? html`<span style="background:rgba(169,214,232,.45);border-bottom:2px solid var(--blue);border-radius:2px" title="pinned — will not change">${seg.t}</span>`
            : seg.kind === "solve" ? html`<span style="background:rgba(95,200,188,.3);border-bottom:2px dashed #1B7F74;border-radius:2px" title="marked to re-solve">${seg.t}</span>`
            : html`<span>${seg.t}</span>`)
          : html`<span class="none">no text on the canvas — it seeds from the current run's answer, or paste below</span>`}
        </div>
      </div>

      <div class="transport" style="border-bottom:none;background:none;padding-top:6px">
        <button class="spd" onClick=${addPin}>◈ PIN selection</button>
        ${editMode === "resolve" && html`<span style="display:contents">
          <button class="spd" onClick=${addSolve} style="color:#1B7F74;border-color:rgba(95,200,188,.7)">◌ RESOLVE selection</button>
          <button class="spd" onClick=${editAnchor}>✎ EDIT anchor</button>
        </span>`}
        <button class="spd" onClick=${clearMarks}>clear marks</button>
        <span style="flex:1"></span>
        ${editMode === "resolve" ? html`<span style="display:contents">
          <span class="lbl" style="font-size:7.5px">steps</span>
          <input type="number" min="4" max="64" value=${steps} onInput=${e => setSteps(e.target.value)}
            style="width:46px;font-family:var(--mono);font-size:10px;border:1px solid var(--edge);border-radius:6px;padding:3px 6px;background:#fff"/>
          <span class="lbl" style="font-size:7.5px">grow</span>
          <input type="number" min="0" max="8" value=${grow} onInput=${e => setGrow(e.target.value)}
            style="width:40px;font-family:var(--mono);font-size:10px;border:1px solid var(--edge);border-radius:6px;padding:3px 6px;background:#fff"/>
          <button class=${"spd" + (busy ? " busy" : "")} onClick=${doResolve}
            style="color:${armed ? "#1B7F74" : "var(--mist)"};border-color:${armed ? "rgba(95,200,188,.8)" : "var(--edge)"};font-weight:600;letter-spacing:.18em">
            ${busy ? "RESOLVING…" : "RESOLVE"}</button>
        </span>` : html`<span style="display:contents">
          <input type="text" placeholder="instruction — e.g. 'make it more formal'" value=${instruction}
            onInput=${e => setInstruction(e.target.value)}
            style="flex:2;min-width:180px;font-family:var(--mono);font-size:10px;border:1px solid var(--edge-soft);border-radius:7px;padding:5px 9px;background:rgba(255,255,255,.6)"/>
          <button class=${"spd" + (busy ? " busy" : "")} onClick=${doRewrite}
            style="color:${armed ? "#1B7F74" : "var(--mist)"};border-color:${armed ? "rgba(95,200,188,.8)" : "var(--edge)"};font-weight:600;letter-spacing:.18em">
            ${busy ? "REWRITING…" : "REWRITE"}</button>
        </span>`}
      </div>

      <div style="padding:0 13px 12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span class="lbl" style="font-size:7.5px">canvas</span>
        <input type="text" placeholder="…or paste/replace the canvas text here and press ⏎"
          style="flex:1;min-width:200px;font-family:var(--mono);font-size:10px;border:1px solid var(--edge-soft);border-radius:7px;padding:5px 9px;background:rgba(255,255,255,.6)"
          onKeyDown=${e => { if(e.key === "Enter" && e.target.value.trim()){
            setText(e.target.value); setPins([]); setSolves([]); setResult(null); e.target.value = ""; } }}/>
        <span class="lbl" style="font-size:7.5px">engine url</span>
        <input type="text" value=${engineUrl()}
          style="width:170px;font-family:var(--mono);font-size:10px;border:1px solid var(--edge-soft);border-radius:7px;padding:5px 9px;background:rgba(255,255,255,.6)"
          onChange=${e => { localStorage.setItem(ENGINE_URL_KEY, e.target.value.trim()); toast("engine url saved"); }}/>
      </div>
    </div>

    ${result && result.mode === "resolve" && html`<div class="mod">
      <div class="mod-h"><span class=${"led " + (result.ok ? "" : "lilac")}></span>
        <span class="cap">resolve — ${result.ok ? "returned" : "failed"}</span>
        <span class="tail">via ${result.via}</span>
        ${result.ok && result.finish && html`<span class="tag der-t">finish · ${result.finish}</span>`}</div>
      ${result.ok
        ? html`<div style="padding:4px 14px 12px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div><div class="lbl" style="padding:4px 0">before</div>
              <div class="leader-body" style="border:1px solid var(--edge-soft);border-radius:8px;max-height:220px">${result.before}</div></div>
            <div><div class="lbl" style="padding:4px 0">after — re-solved bidirectionally</div>
              <div class="leader-body" style="border:1px solid rgba(95,200,188,.5);border-radius:8px;max-height:220px;background:rgba(95,200,188,.06)">${result.after}</div></div>
            <div class="none" style="grid-column:1/-1;font-size:8.5px">
              only the marked spans were re-opened; pins and unmarked text were held by the engine's
              pin invariant. A score-delta receipt for resolves lands when /score gains a passthrough.</div>
          </div>`
        : html`<div style="padding:4px 14px 12px" class="none">${result.err}</div>`}
    </div>`}

    ${result && result.mode === "rewrite" && html`<div class="mod">
      <div class="mod-h"><span class=${"led " + (result.ok ? "" : "lilac")}></span>
        <span class="cap">rewrite — ${result.ok ? "returned" : "failed"}</span>
        ${result.ok && result.finish && html`<span class="tag der-t">finish · ${result.finish}</span>`}
        ${result.ok && html`<span class=${"tag " + (result.allPinsKept ? "cap-t" : "der-t")}>${
          result.allPinsKept ? "all pins kept" : "a pin broke"}</span>`}</div>
      ${result.ok
        ? html`<div style="padding:4px 14px 12px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div><div class="lbl" style="padding:4px 0">before</div>
              <div class="leader-body" style="border:1px solid var(--edge-soft);border-radius:8px;max-height:220px">${result.before}</div></div>
            <div><div class="lbl" style="padding:4px 0">after — ${result.note}</div>
              <div class="leader-body" style="border:1px solid ${result.allPinsKept ? "rgba(95,200,188,.5)" : "var(--lilac)"};border-radius:8px;max-height:220px;background:${result.allPinsKept ? "rgba(95,200,188,.06)" : "rgba(200,150,200,.06)"}">${result.after}</div></div>
            ${result.pins.length > 0 && html`<div style="grid-column:1/-1;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
              <span class="lbl" style="font-size:7.5px">pin fidelity — measured, not assumed</span>
              ${result.pins.map(pn => html`<span class=${"tag " + (pn.kept ? "cap-t" : "der-t")}
                title=${pn.kept ? "survived verbatim" : "did NOT survive verbatim in the output"}>
                ${pn.kept ? "✓" : "✗"} "${pn.text.length > 24 ? pn.text.slice(0, 24) + "…" : pn.text}"</span>`)}
            </div>`}
            <div class="none" style="grid-column:1/-1;font-size:8.5px">
              ${result.note} — pins are prompt-level constraints the model was ASKED to honor, not an
              engine-enforced invariant; the fidelity tags above are a post-hoc verbatim check on the
              actual output, not a guarantee made in advance.</div>
          </div>`
        : html`<div style="padding:4px 14px 12px" class="none">${result.err}</div>`}
    </div>`}
  </div>`;
}
