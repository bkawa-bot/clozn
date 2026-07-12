/* heavnOS · Atlas — the model-diff view (F8): where two runs diverge, token by token.
   OBSERVATIONAL by design and labeled so: post-split disagreement is downstream consequence, not
   independent evidence; char similarity is surface (wording, not meaning). The rigorous upgrade
   path is quant receipts (teacher-forced same-continuation scoring), named in the caveat. */
import { html, useState } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast, shortTime } from "../state.mjs";
import { api } from "../api.mjs";

export function AtlasModule(){
  const runs = useStore(x => x.runs);
  const live = useStore(x => x.live);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);

  const diff = async () => {
    if(!live){ toast("Atlas — live server only"); return; }
    if(!a || !b || a === b){ toast("pick two different runs"); return; }
    setBusy(true); setOut(null);
    const r = await api.diffRuns(a, b);
    setBusy(false);
    setOut(r || { error: "diff didn't answer" });
  };

  const opt = r => html`<option value=${r.id}>
    ${(r.prompt_summary || r.id).slice(0, 46)} · ${shortTime(r.created_at)}</option>`;
  const sum = (out && out.summary) || {};
  const fd = out && out.first_divergence;
  const positions = (out && out.positions) || [];

  return html`<div class="replay-grid"><div class="col">
    <div class="mod">
      <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
      <div class="mod-h"><span class="led blue"></span><span class="cap">model-diff atlas</span>
        <span class="tail">same prompt, two runs — where do they part ways?</span>
        <span class="tag der-t">DERIVED</span></div>
      <div style="padding:4px 13px 12px;display:flex;flex-direction:column;gap:8px">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <span class="cap" style="font-size:9px">A</span>
          <select value=${a} onChange=${e => setA(e.target.value)}
            style="flex:1;min-width:200px;font-size:10px;padding:4px">
            <option value="">— pick run A —</option>${runs.map(opt)}</select>
          <span class="cap" style="font-size:9px">B</span>
          <select value=${b} onChange=${e => setB(e.target.value)}
            style="flex:1;min-width:200px;font-size:10px;padding:4px">
            <option value="">— pick run B —</option>${runs.map(opt)}</select>
          <button class="spd" disabled=${busy} onClick=${diff}>${busy ? "…" : "DIFF"}</button>
        </div>
        <span class="none" style="font-size:8.5px">compare a run against its replay under a different
          quant/model/config — replay first if you don't have a pair yet</span>

        ${out && out.error && html`<span class="none">${out.error}</span>`}
        ${out && !out.error && html`
          ${out.warn && html`<div class="cfg" style="border-left-color:var(--coral)">
            <span class="cap" style="color:#C24A31">warn</span><span>${out.warn}</span></div>`}
          <div class="cfg">
            <span class="cap">A</span><b>${(out.a || {}).model || "?"}</b>
            ${(out.a || {}).quant && html`<span class="dot">·</span><span>${out.a.quant}</span>`}
            <span class="dot">·</span>
            <span class="cap">B</span><b>${(out.b || {}).model || "?"}</b>
            ${(out.b || {}).quant && html`<span class="dot">·</span><span>${out.b.quant}</span>`}
            <span style="margin-left:auto" class=${sum.identical ? "tag cap-t" : "tag der-t"}>
              ${sum.identical ? "IDENTICAL" : "DIVERGED"}</span>
          </div>

          ${fd && html`<div class="receipt-row">
            <span>first divergence at token <b>${fd.index}</b>
              (${out.common_prefix_len} shared): “${(fd.a_piece || "").trim() || "·"}” vs
              “${(fd.b_piece || "").trim() || "·"}”</span>
            <span class="sub">
              <span>A conf ${fd.a_conf != null ? (+fd.a_conf).toFixed(2) : "—"}</span>
              <span>B conf ${fd.b_conf != null ? (+fd.b_conf).toFixed(2) : "—"}</span>
              ${sum.b_was_alternative_in_a && sum.b_was_alternative_in_a.found
                ? html`<span class="yes">B's token was A's recorded alternative
                    (rank ${sum.b_was_alternative_in_a.rank}${sum.b_was_alternative_in_a.prob != null
                    ? ", p " + (+sum.b_was_alternative_in_a.prob).toFixed(2) : ""}) — it almost said that</span>`
                : sum.b_was_alternative_in_a && sum.b_was_alternative_in_a.checked
                ? html`<span class="no">B's token was NOT among A's recorded alternatives</span>` : null}
            </span>
          </div>`}

          ${positions.length ? html`<div style="overflow-x:auto">
            <div style="display:flex;gap:1px;padding:4px 0">
              ${positions.map(p => html`<div title=${"i " + p.i + " · A “" + (p.a_piece ?? "∅") + "” · B “" + (p.b_piece ?? "∅") + "”"}
                style=${"width:7px;height:26px;border-radius:2px;flex:none;background:"
                  + (p.same ? "rgba(95,200,188,.35)" : "rgba(242,109,79,.75)")}></div>`)}
            </div>
            <span class="none" style="font-size:8px">teal = same token · coral = differs
              ${out.positions_truncated ? " · first 200 positions shown" : ""}</span>
          </div>` : out.trace_available === false
            ? html`<span class="none">no per-token trace on one side — text-only comparison</span>` : null}

          <div class="cfg">
            <span>len A <b>${sum.a_reply_tokens ?? "—"}</b> / B <b>${sum.b_reply_tokens ?? "—"}</b></span>
            <span class="dot">·</span>
            <span>mean conf A <b>${sum.a_mean_confidence != null ? (+sum.a_mean_confidence).toFixed(2) : "—"}</b>
              / B <b>${sum.b_mean_confidence != null ? (+sum.b_mean_confidence).toFixed(2) : "—"}</b></span>
            <span class="dot">·</span>
            <span>${sum.char_similarity != null ? (sum.char_similarity * 100).toFixed(0) + "% " : ""}
              ${sum.char_similarity_label || "surface similarity — wording, not meaning"}</span>
          </div>
          ${out.caveat && html`<div class="provenance">${out.caveat}</div>`}`}
      </div>
    </div>
  </div><div class="col"></div></div>`;
}
