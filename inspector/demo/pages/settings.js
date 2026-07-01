/* settings.js -- the Settings page.  Issue I1.
 *
 * Deliberately boring + safe local config, over ONLY endpoints that already exist (no invented calls):
 *   GET  /substrate        -> {active, available:[...]}      (SAFE read -- see the note below)
 *   POST /substrate {name}  -> {active, switched}            (the explicit switch; re-execs + reloads ~30s)
 *   GET  /runs             -> {runs:[...]}                   (count)
 *   POST /memory/cards {}  -> {cards:[...], has_prefix}      (count)
 *   POST /steer/axes  {}   -> {axes:[{name,value,...}]}      (active-dial count, |value|>=0.05)
 *   POST /reset {keep_prefix:false}                          (clears learned memory -- guarded/confirmed)
 *
 * IMPORTANT (why reads use GET, not POST, on /substrate): the backend's POST /substrate defaults the
 * body's `name` to "qwen" and TRIGGERS A SWITCH whenever name != the active substrate. So POSTing {} from
 * a Dream session would silently re-exec into Qwen. The read here is therefore GET /substrate (which safely
 * returns {active, available}); POST /substrate is used ONLY for a deliberate switch with an explicit name.
 *
 * Retention / export-import / privacy have NO backend yet -> rendered as a labelled "coming soon" section
 * (IA visible, no calls). Pure consumer of the backend; every fetch is guarded so the page renders offline.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  var STYLE_ID = "settings-page-style";
  var CSS =
    ".set-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;margin-top:22px;align-items:start}" +
    "@media(max-width:900px){.set-grid{grid-template-columns:1fr}}" +
    ".set-panel{padding:0 0 16px}" +
    ".set-panel .srows{padding:6px 18px 4px}" +
    ".set-body{padding:6px 18px 4px}" +
    ".set-hint{color:var(--faint);font-size:12.5px;line-height:1.5;margin:2px 18px 12px}" +
    ".set-hint b{color:var(--soft)}" +
    // copyable endpoint row
    ".set-ep{display:flex;gap:8px;align-items:center;margin:0 18px 2px}" +
    ".set-ep code{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:8px 12px;font-size:12.5px}" +
    ".set-copy{flex:none}" +
    // substrate switch controls
    ".set-switch{display:flex;gap:8px;align-items:center;margin:8px 18px 2px;flex-wrap:wrap}" +
    ".set-select{font:inherit;font-size:13.5px;border:1px solid var(--line);background:rgba(255,255,255,.85);" +
    "color:var(--ink);border-radius:22px;padding:8px 13px;outline:none;cursor:pointer;transition:border-color .2s,box-shadow .2s}" +
    ".set-select:focus{border-color:var(--halo);box-shadow:0 0 0 3px rgba(122,167,255,.14)}" +
    ".set-select:disabled{opacity:.55;cursor:default}" +
    ".set-switchmsg{color:var(--soft);font-size:12.5px;margin:9px 18px 0;line-height:1.5;display:none}" +
    ".set-switchmsg.on{display:block}" +
    ".set-switchmsg .spin{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin-right:7px;" +
    "border-radius:50%;border:2px solid rgba(122,167,255,.35);border-top-color:var(--halo);animation:setspin .8s linear infinite}" +
    "@keyframes setspin{to{transform:rotate(360deg)}}" +
    // counts row of big numbers
    ".set-counts{display:flex;gap:10px;flex-wrap:wrap;padding:10px 18px 6px}" +
    ".set-count{flex:1;min-width:96px;border:1px solid var(--line);border-radius:14px;padding:12px 14px;" +
    "background:rgba(255,255,255,.6);text-align:center}" +
    ".set-count .n{font-size:24px;font-weight:700;color:var(--ink);line-height:1;font-variant-numeric:tabular-nums}" +
    ".set-count .n.faint{color:var(--faint);font-weight:500}" +
    ".set-count .l{font-size:11px;color:var(--faint);letter-spacing:.04em;margin-top:6px}" +
    // storage contents list
    ".set-store{margin:2px 18px 2px;list-style:none;padding:0}" +
    ".set-store li{display:flex;gap:9px;align-items:baseline;padding:5px 0;border-bottom:1px dashed var(--line);font-size:13px;color:var(--soft)}" +
    ".set-store li:last-child{border-bottom:none}" +
    ".set-store b{color:var(--ink);font-weight:600;min-width:96px;flex:none}" +
    // reset (danger) zone
    ".set-danger{margin-top:20px;padding:0 0 16px;border-color:rgba(231,120,120,.32)}" +
    ".set-danger h2{color:#b06a6a}" +
    ".set-warn{margin:10px 18px 12px;padding:11px 14px;border-radius:12px;font-size:12.5px;line-height:1.5;" +
    "color:#8a4a4a;background:rgba(231,120,120,.10);border:1px solid rgba(231,120,120,.30)}" +
    ".set-warn b{color:#a5433f}" +
    ".set-resetrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:0 18px}" +
    ".set-reset{border-color:rgba(231,120,120,.4);color:#b0504a;background:rgba(231,120,120,.08)}" +
    ".set-reset:hover:not(:disabled){background:rgba(231,120,120,.16);border-color:rgba(231,120,120,.55);color:#a5433f;box-shadow:none;transform:none}" +
    ".set-reset.confirm{background:rgba(231,120,120,.9);border-color:#c0504a;color:#fff;font-weight:640}" +
    ".set-reset.confirm:hover:not(:disabled){background:#c0504a;color:#fff}" +
    ".set-resetmsg{color:var(--soft);font-size:12.5px}" +
    ".set-resetmsg.ok{color:var(--good)}" +
    ".set-resetmsg.err{color:#c0504a}" +
    // coming-soon section
    ".set-soon{margin-top:20px;padding:16px 18px}" +
    ".set-soon h2{padding:0 0 4px}" +
    ".set-soonlist{list-style:none;padding:0;margin:12px 0 0}" +
    ".set-soonlist li{padding:9px 0;border-bottom:1px dashed var(--line)}" +
    ".set-soonlist li:last-child{border-bottom:none}" +
    ".set-soon-t{font-size:13.5px;color:var(--ink);font-weight:600}" +
    ".set-soon-d{font-size:12.5px;color:var(--faint);margin-top:3px;line-height:1.45}" +
    ".set-offline{margin:16px 0 0;padding:10px 13px;border-radius:11px;font-size:12.5px;color:var(--faint);" +
    "background:rgba(120,140,190,.06);border:1px solid var(--line)}";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  // page-local state (kept tiny; the page is mostly read-only)
  var state = { active: null, available: [], switching: false, resetArmed: false };

  function render(view, ctx) {
    ensureStyle();
    state = { active: null, available: [], switching: false, resetArmed: false };

    var endpoint = ctx.endpoint;

    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("h1", {}, [S.el("span", { class: "glow" }, ["Settings"])]),
        S.el("p", { class: "sub" }, [
          "Your local runtime, where its data lives, and a couple of safe controls. Deliberately boring — ",
          "nothing here changes how the model answers except switching the active model.",
        ]),

        // reachability banner (shown only when nothing answered)
        S.el("div", { class: "set-offline", id: "set-offline", style: "display:none" }, [
          "The studio server is not reachable — counts and the model switch are unavailable. Start ",
          S.el("code", {}, ["clozn studio"]),
          " (default port 8090), then reload.",
        ]),

        S.el("div", { class: "set-grid" }, [
          runtimeCard(ctx, endpoint),
          storageCard(),
        ]),

        countsCard(),
        dangerCard(ctx),
        comingSoonCard(),
      ])
    );

    loadAll(ctx);
  }

  // ---- Runtime card: active substrate + model + endpoint + switch --------------------------------
  function runtimeCard(ctx, endpoint) {
    // copyable endpoint
    var epCode = S.el("code", {}, [endpoint]);
    var copyBtn = S.el("button", { class: "set-copy" }, ["Copy"]);
    copyBtn.addEventListener("click", function () {
      ctx.copyText(endpoint).then(function (ok) {
        copyBtn.textContent = ok ? "Copied ✓" : "Copy failed";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1400);
      });
    });

    // substrate switch: a select of `available` + a Switch button (disabled until a *different* one is picked)
    var sel = S.el("select", { class: "set-select", id: "set-subsel", disabled: "disabled" }, [
      S.el("option", { value: "" }, ["loading…"]),
    ]);
    var switchBtn = S.el("button", { id: "set-switchbtn", disabled: "disabled" }, ["Switch model"]);
    sel.addEventListener("change", function () {
      switchBtn.disabled = state.switching || !sel.value || sel.value === state.active;
    });
    switchBtn.addEventListener("click", function () { doSwitch(ctx, sel.value); });

    return S.el("section", { class: "panel set-panel" }, [
      S.el("h2", {}, ["runtime"]),
      S.el("div", { class: "srows", id: "set-runtimerows" }, [
        srow("Local runtime", S.el("span", { class: "sval faintv" }, ["checking…"])),
      ]),
      S.el("p", { class: "set-hint", style: "margin-top:6px" }, [
        "The OpenAI-compatible endpoint your tools connect to:",
      ]),
      S.el("div", { class: "set-ep" }, [epCode, copyBtn]),
      S.el("p", { class: "set-hint", style: "margin-top:12px" }, [
        "Switch the active model. ", S.el("b", {}, ["This reloads the runtime and takes ~30s"]),
        " — one 7B fits the GPU, so the server restarts on the new one.",
      ]),
      S.el("div", { class: "set-switch" }, [sel, switchBtn]),
      S.el("div", { class: "set-switchmsg", id: "set-switchmsg" }, []),
    ]);
  }

  // fill the switch select + the runtime rows from GET /substrate (safe read; see file header).
  function fillRuntime(ctx, sub, reachable) {
    var rows = document.getElementById("set-runtimerows");
    if (rows) {
      rows.innerHTML = "";
      rows.appendChild(srow("Local runtime",
        withDot(reachable ? "running" : "not reachable", reachable ? "ok" : "off")));
      rows.appendChild(srow("Active model", S.el("span", { class: "sval" }, [sub && sub.active ? cap(sub.active) : "unknown"])));
      rows.appendChild(srow("Endpoint", S.el("span", { class: "sval mono" }, [ctx.endpoint])));
    }

    var sel = document.getElementById("set-subsel");
    var btn = document.getElementById("set-switchbtn");
    var avail = (sub && sub.available) || [];
    state.active = (sub && sub.active) || null;
    state.available = avail;
    if (sel) {
      sel.innerHTML = "";
      if (!avail.length) {
        sel.appendChild(S.el("option", { value: "" }, [reachable ? "no models listed" : "unavailable"]));
        sel.disabled = true;
      } else {
        avail.forEach(function (name) {
          var isActive = name === state.active;
          sel.appendChild(S.el("option", { value: name }, [cap(name) + (isActive ? " (active)" : "")]));
        });
        if (state.active) sel.value = state.active;
        sel.disabled = state.switching;
      }
    }
    if (btn) btn.disabled = state.switching || !sel || !sel.value || sel.value === state.active;
  }

  function doSwitch(ctx, name) {
    if (!name || name === state.active || state.switching) return;
    state.switching = true;
    var sel = document.getElementById("set-subsel");
    var btn = document.getElementById("set-switchbtn");
    var msg = document.getElementById("set-switchmsg");
    if (sel) sel.disabled = true;
    if (btn) { btn.disabled = true; btn.textContent = "Switching…"; }
    if (msg) {
      msg.className = "set-switchmsg on";
      msg.innerHTML = "";
      msg.appendChild(S.el("span", { class: "spin" }, []));
      msg.appendChild(document.createTextNode(
        "Reloading into " + cap(name) + " — the server is restarting on a clean GPU (~30s). " +
        "This page will update when it's back."));
    }

    // Explicit switch: POST /substrate with an explicit name. The server acks then re-execs itself,
    // so this request may also just drop -- either way we poll GET /substrate until `active` flips.
    ctx.postJSON("/substrate", { name: name }, null).then(function () {
      pollUntilActive(ctx, name, 0);
    });
  }

  // Poll GET /substrate until active == target (or we give up). The server is unreachable mid-reload,
  // so getJSON returns null then -- we simply keep waiting. ~60 tries * 2s ≈ 2 min ceiling.
  function pollUntilActive(ctx, target, tries) {
    if (tries > 60) {
      var m = document.getElementById("set-switchmsg");
      if (m) {
        m.innerHTML = "";
        m.appendChild(document.createTextNode(
          "Still reloading into " + cap(target) + "… taking longer than usual. Reload the page to check."));
      }
      return;
    }
    setTimeout(function () {
      ctx.getJSON("/substrate", null).then(function (sub) {
        if (sub && sub.active === target) {
          // back up on the new substrate: refresh the whole page state.
          state.switching = false;
          var m = document.getElementById("set-switchmsg");
          if (m) { m.className = "set-switchmsg on"; m.textContent = "Now running " + cap(target) + " ✓"; }
          var btn = document.getElementById("set-switchbtn");
          if (btn) btn.textContent = "Switch model";
          loadAll(ctx);
          return;
        }
        pollUntilActive(ctx, target, tries + 1);
      });
    }, 2000);
  }

  // ---- Storage card: the ~/.clozn location + what lives there (prose; no FS access from a browser) ---
  function storageCard() {
    return S.el("section", { class: "panel set-panel" }, [
      S.el("h2", {}, ["storage"]),
      S.el("p", { class: "set-hint", style: "margin-top:6px" }, [
        "Everything Clozn keeps is local, on this machine, under:",
      ]),
      S.el("div", { class: "set-ep" }, [S.el("code", {}, ["~/.clozn"])]),
      S.el("p", { class: "set-hint", style: "margin:12px 18px 8px" }, ["What lives there:"]),
      S.el("ul", { class: "set-store" }, [
        storeItem("runs", "every captured interaction, one JSON per run (the Run Log)"),
        storeItem("memory", "the learned memory prefix + trait cards"),
        storeItem("personality", "saved dial positions (tone / cognition)"),
        storeItem("custom dials", "your make-your-own steering directions"),
        storeItem("models", "downloaded model weights"),
      ]),
      S.el("p", { class: "set-hint", style: "margin:12px 18px 0" }, [
        "Shown for reference — the browser can't read your filesystem, so this is the path, not a live listing.",
      ]),
    ]);
  }
  function storeItem(name, desc) {
    return S.el("li", {}, [S.el("b", {}, [name]), S.el("span", {}, [desc])]);
  }

  // ---- Counts card: runs / memories / active dials -----------------------------------------------
  function countsCard() {
    return S.el("section", { class: "panel", style: "margin-top:20px;padding:0 0 8px" }, [
      S.el("h2", {}, ["what's here"]),
      S.el("div", { class: "set-counts", id: "set-counts" }, [
        countBox("set-runs", "runs"),
        countBox("set-mems", "memories"),
        countBox("set-dials", "active dials"),
      ]),
    ]);
  }
  function countBox(id, label) {
    return S.el("div", { class: "set-count" }, [
      S.el("div", { class: "n faint", id: id }, ["…"]),
      S.el("div", { class: "l" }, [label]),
    ]);
  }
  function setCount(id, n) {
    var el = document.getElementById(id);
    if (!el) return;
    if (n == null) { el.textContent = "—"; el.className = "n faint"; }
    else { el.textContent = String(n); el.className = "n"; }
  }

  // ---- Danger zone: reset memory (guarded -- confirm before it fires) -----------------------------
  function dangerCard(ctx) {
    var msg = S.el("span", { class: "set-resetmsg", id: "set-resetmsg" }, []);
    var btn = S.el("button", { class: "set-reset", id: "set-resetbtn" }, ["Reset memory"]);

    // Two-step guard: first click ARMS ("Click again to confirm"), second click fires. A 4s timeout
    // (and a re-render) disarms so it can never fire from a stale state.
    var disarmTimer = null;
    function disarm() {
      state.resetArmed = false;
      btn.className = "set-reset";
      btn.textContent = "Reset memory";
      if (disarmTimer) { clearTimeout(disarmTimer); disarmTimer = null; }
    }
    btn.addEventListener("click", function () {
      if (state.switching) return;
      if (!state.resetArmed) {
        state.resetArmed = true;
        btn.className = "set-reset confirm";
        btn.textContent = "Click again to confirm";
        msg.className = "set-resetmsg";
        msg.textContent = "This clears all learned memory. It cannot be undone.";
        disarmTimer = setTimeout(disarm, 4000);
        return;
      }
      // confirmed -> fire the reset that clears learned memory (endpoint exists today).
      disarm();
      btn.disabled = true; btn.textContent = "Resetting…";
      msg.className = "set-resetmsg";
      msg.textContent = "Clearing learned memory…";
      ctx.postJSON("/reset", { keep_prefix: false }, null).then(function (res) {
        btn.disabled = false; btn.textContent = "Reset memory";
        if (res == null) {
          msg.className = "set-resetmsg err";
          msg.textContent = "Couldn't reset — is the studio server up? Nothing was changed.";
          return;
        }
        msg.className = "set-resetmsg ok";
        msg.textContent = "Memory cleared ✓";
        // refresh the counts so the memory tile drops to its new value.
        refreshCounts(ctx);
      });
    });

    return S.el("section", { class: "panel set-panel set-danger" }, [
      S.el("h2", {}, ["reset"]),
      S.el("div", { class: "set-warn" }, [
        S.el("b", {}, ["Reset memory"]),
        " erases every learned trait and the memory prefix — the agent forgets what it has picked up from your ",
        "conversations. Your runs, dials, and models are untouched. ", S.el("b", {}, ["This cannot be undone."]),
      ]),
      S.el("div", { class: "set-resetrow" }, [btn, msg]),
    ]);
  }

  // ---- Coming soon: retention / export-import / privacy (NO backend yet -> IA-only placeholders) ---
  function comingSoonCard() {
    return S.el("section", { class: "panel set-soon" }, [
      S.el("h2", {}, [
        S.el("span", { class: "cs-badge" }, ["coming soon"]),
        "planned settings",
      ]),
      S.el("p", { class: "set-hint", style: "margin:8px 0 0" }, [
        "These aren't wired yet — shown so the shape of Settings is visible. No controls here do anything.",
      ]),
      S.el("ul", { class: "set-soonlist" }, [
        soonItem("Memory & trace retention",
          "keep runs and memory for a chosen window; prune older records automatically."),
        soonItem("Export / import",
          "download your ~/.clozn (runs + memory + dials) as a portable bundle, and restore it elsewhere."),
        soonItem("Privacy",
          "control what a captured run stores (e.g. redact prompts) and pause capture entirely."),
        soonItem("Server port",
          "change the port the studio serves on (currently set when you launch it)."),
      ]),
    ]);
  }
  function soonItem(title, desc) {
    return S.el("li", {}, [
      S.el("div", { class: "set-soon-t" }, [title]),
      S.el("div", { class: "set-soon-d" }, [desc]),
    ]);
  }

  // ---- data loading ------------------------------------------------------------------------------
  function loadAll(ctx) {
    // GET /substrate is the SAFE read (POST would trigger a switch); the rest are the standard POST/GET reads.
    Promise.all([
      ctx.getJSON("/substrate", null),
      ctx.getJSON("/runs", null),
      ctx.postJSON("/memory/cards", {}, null),
      ctx.postJSON("/steer/axes", {}, null),
    ]).then(function (r) {
      var sub = r[0], runsResp = r[1], cardsResp = r[2], axesResp = r[3];
      var reachable = !!(sub || runsResp || cardsResp || axesResp);

      var off = document.getElementById("set-offline");
      if (off) off.style.display = reachable ? "none" : "";

      fillRuntime(ctx, sub || {}, reachable);

      setCount("set-runs", runsResp ? ((runsResp.runs || []).length) : null);
      setCount("set-mems", cardsResp ? ((cardsResp.cards || []).length) : null);
      setCount("set-dials", axesResp ? activeDialCount(axesResp.axes || []) : null);
    });
  }

  // re-read just the counts (after a reset) without disturbing the switch UI.
  function refreshCounts(ctx) {
    Promise.all([
      ctx.getJSON("/runs", null),
      ctx.postJSON("/memory/cards", {}, null),
      ctx.postJSON("/steer/axes", {}, null),
    ]).then(function (r) {
      setCount("set-runs", r[0] ? ((r[0].runs || []).length) : null);
      setCount("set-mems", r[1] ? ((r[1].cards || []).length) : null);
      setCount("set-dials", r[2] ? activeDialCount(r[2].axes || []) : null);
    });
  }

  function activeDialCount(axes) {
    var n = 0;
    for (var i = 0; i < axes.length; i++) {
      if (Math.abs(+(axes[i] && axes[i].value) || 0) >= 0.05) n++;
    }
    return n;
  }

  // ---- small helpers -----------------------------------------------------------------------------
  function srow(label, valueNode) {
    return S.el("div", { class: "srow" }, [
      S.el("span", { class: "slabel" }, [label]),
      S.el("span", { class: "sval-wrap" }, [valueNode]),
    ]);
  }
  function withDot(text, dot) {
    return S.el("span", { class: "sval-wrap" }, [
      S.el("span", { class: "sdot " + dot }, []),
      S.el("span", { class: "sval" }, [text]),
    ]);
  }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  S.register("settings", { title: "Settings", render: render });
})();
