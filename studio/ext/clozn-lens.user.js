// ==UserScript==
// @name         clozn lens
// @namespace    clozn
// @version      0.1.0
// @description  In-context glass box (AMBIENT_DELIVERY.md channel 3): a floating clozn panel on ANY tool
//               you point at clozn — shows the latest reply's shaky spans + one click to the receipt.
// @match        http://localhost/*
// @match        http://127.0.0.1/*
// @match        http://192.168.*/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==
/* Load in Tampermonkey/Violentmonkey. Edit CLOZN below to your clozn base URL, and add @match lines for
   the tool whose pages you want the panel on (Open WebUI, LibreChat, a local chat page — anything you've
   pointed at clozn's /v1). It reads the journal cross-origin (clozn already sends CORS on GET), matches
   the NEWEST run to the reply you're looking at (correct for a single-user local clozn), and surfaces its
   confidence spans. HONEST: raw uncalibrated confidence — a "worth a look", never "wrong". This v1 shows
   the PANEL (robust); inline text-shading of the host page's DOM is the documented next step (fragile,
   per-tool). */
(function () {
  "use strict";
  const CLOZN = "http://127.0.0.1:8090";   // <-- your clozn base URL
  const POLL_MS = 4000;

  // ---- shadow-DOM panel (isolated from the host page's CSS) ----
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647;";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      :host{all:initial}
      .p{width:270px;font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#2A3252;
        border-radius:13px;border:1px solid rgba(255,255,255,.85);
        background:linear-gradient(180deg,rgba(255,255,255,.92),rgba(238,244,250,.86));
        box-shadow:0 12px 34px rgba(80,100,150,.28), inset 0 1px 0 #fff;backdrop-filter:blur(8px)}
      .h{display:flex;align-items:center;gap:7px;padding:9px 12px;cursor:pointer}
      .dot{width:7px;height:7px;border-radius:50%;background:#5FC8BC;box-shadow:0 0 7px #5FC8BC}
      .dot.warn{background:#E8A24A;box-shadow:0 0 7px #E8A24A}
      .dot.off{background:#B8C0D4;box-shadow:none}
      .t{font-weight:700;letter-spacing:.18em;text-transform:uppercase;font-size:9px;color:#4A5878}
      .col{margin-left:auto;font-size:10px;color:#8290AC}
      .b{padding:2px 12px 12px;display:none}
      .b.open{display:block}
      .q{font-size:10px;color:#8290AC;margin:2px 0 7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      .stat{font-size:10px;color:#4A5878;margin-bottom:6px}
      .stat b{color:#2A3252}
      .chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
      .chip{font-size:10px;color:#8A5A12;background:rgba(232,162,74,.15);border:1px solid rgba(232,162,74,.5);
        border-radius:5px;padding:1px 6px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      a.r{display:inline-block;font-size:10px;color:#1B7F74;text-decoration:none;border-bottom:1px dotted #1B7F74}
      .note{font-size:8px;color:#8290AC;margin-top:7px;line-height:1.5}
    </style>
    <div class="p">
      <div class="h" id="hd"><span class="dot off" id="dot"></span><span class="t">clozn · glass box</span>
        <span class="col" id="col">▸</span></div>
      <div class="b" id="body"><div class="q" id="q">waiting for a reply…</div>
        <div class="stat" id="stat"></div><div class="chips" id="chips"></div>
        <a class="r" id="link" href="#" target="_blank" rel="noopener" style="display:none">open the receipt →</a>
        <div class="note">raw model confidence — a prompt to look, not a verdict. Matches your newest clozn run.</div>
      </div>
    </div>`;
  document.documentElement.appendChild(host);

  const $ = id => root.getElementById(id);
  let open = false, lastId = null;
  $("hd").addEventListener("click", () => {
    open = !open; $("body").classList.toggle("open", open); $("col").textContent = open ? "▾" : "▸";
  });

  async function jget(path) {
    try { const r = await fetch(CLOZN + path, { credentials: "omit" }); return r.ok ? r.json() : null; }
    catch (e) { return null; }
  }

  function render(run, spans) {
    const shaky = (spans || []).filter(s => s.band === "shaky");
    $("dot").className = "dot" + (shaky.length ? " warn" : "");
    $("q").textContent = "“" + (run.prompt_summary || run.id) + "”";
    const confs = (spans || []).flatMap(s => Array(s.n_tokens || 1).fill(s.mean_conf)).filter(x => x != null);
    const mean = confs.length ? (confs.reduce((a, b) => a + b, 0) / confs.length) : null;
    $("stat").innerHTML = (mean != null ? "mean conf <b>" + mean.toFixed(2) + "</b> · " : "") +
      (shaky.length ? "<b>" + shaky.length + "</b> span" + (shaky.length > 1 ? "s" : "") + " worth a look"
                    : "confident throughout");
    $("chips").innerHTML = "";
    shaky.slice(0, 6).forEach(s => {
      const c = document.createElement("span"); c.className = "chip";
      c.textContent = (s.text || "").trim().slice(0, 24) || "·";
      c.title = "conf " + (s.mean_conf != null ? s.mean_conf.toFixed(2) : "?");
      $("chips").appendChild(c);
    });
    const a = $("link"); a.href = CLOZN + "/r/" + run.id; a.style.display = "inline-block";
  }

  async function poll() {
    const list = await jget("/runs");
    if (!list || !Array.isArray(list.runs) || !list.runs.length) {
      $("dot").className = "dot off"; return;
    }
    const run = list.runs[0];
    if (run.id === lastId) return;                 // nothing new
    lastId = run.id;
    const sp = await jget("/runs/" + encodeURIComponent(run.id) + "/spans");
    render(run, (sp && Array.isArray(sp.spans)) ? sp.spans : (Array.isArray(sp) ? sp : []));
  }

  poll();
  setInterval(poll, POLL_MS);
})();
