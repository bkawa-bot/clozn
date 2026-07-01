/* memory.js -- the Memory page.  Issue D3 (with room for D1/D2's structured cards + E's review states).
 *
 * Makes the agent's memory inspectable + editable against the CURRENT (string-based) endpoints:
 *   POST /memory/cards    {}            -> {cards:[<string|object>,...], has_prefix}
 *   POST /memory/strength {}            -> {strength:<float>, has_prefix}   (read)
 *   POST /memory/strength {value}       -> set (0=off, 1=normal, up to 2=stronger)
 *   POST /memory/add      {text}        -> adds a trait + RE-TRAINS the prefix   (SLOW: tens of s .. minutes)
 *   POST /memory/remove   {index}       -> removes card at index + rebuilds
 *
 * Forward-compat: a card is a *string* today, but D2 will return objects
 * {id,text,status,source_run_id,created_at,last_used_at,usage_count,kind,risk,strength}. cardText()/
 * cardMeta() normalize both shapes so a richer object slots in with no rewrite -- extra fields render
 * as provenance/usage/status chips automatically; unknown scalar fields degrade to labelled chips.
 *
 * Pure consumer of the backend (app.js owns the shell + fetch plumbing). Every fetch is guarded so the
 * page renders fully offline.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  // page-local styles (memory-specific; the shared look comes from clozn.css / app.html). No redesign:
  // reuses .panel/.mcard/.dot conventions + the palette variables.
  var STYLE_ID = "memory-page-style";
  var CSS =
    ".mem-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}" +
    ".mem-strength{margin:20px 0 8px;padding:16px 18px}" +
    ".mem-strength h2{padding:0 0 4px}" +
    ".mem-strength .strengthrow{display:flex;align-items:center;gap:14px;margin-top:10px}" +
    ".mem-strength input[type=range]{flex:1;min-width:120px;accent-color:var(--halo);cursor:pointer}" +
    ".mem-strength input[type=range]:disabled{cursor:default;opacity:.5}" +
    ".mem-strength .strengthval{font-size:15px;font-weight:680;color:var(--ink);min-width:2.4em;text-align:right;" +
    "font-family:ui-monospace,Consolas,monospace}" +
    ".mem-strength .strengthhint{color:var(--faint);font-size:12.5px;margin-top:9px;line-height:1.45}" +
    ".mem-strength .strengthhint b{color:var(--soft);font-weight:600}" +
    ".mem-strength .ticks{display:flex;justify-content:space-between;font-size:10.5px;color:var(--faint);" +
    "letter-spacing:.02em;margin-top:4px;padding:0 1px}" +
    ".mem-listhead{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:26px 0 6px}" +
    ".mem-listhead h2{font-size:12px;font-weight:680;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin:0}" +
    ".mem-count{font-size:12px;color:var(--faint)}" +
    ".mem-list{padding:10px 14px 14px}" +
    // a memory card: reuse .mcard's frame but allow a body column + a metadata row.
    ".mem-card{display:flex;gap:11px;align-items:flex-start;padding:11px 13px;margin:8px 0;border-radius:13px;" +
    "background:rgba(255,255,255,.72);border:1px solid var(--line);transition:box-shadow .2s,opacity .3s}" +
    ".mem-card.busy{opacity:.55}" +
    ".mem-card .dot{width:9px;height:9px;border-radius:50%;margin-top:6px;flex:none;" +
    "background:radial-gradient(circle at 35% 30%,#fff,var(--halo));box-shadow:0 0 10px var(--halo)}" +
    ".mem-card:nth-child(3n+2) .dot{background:radial-gradient(circle at 35% 30%,#fff,var(--warm));box-shadow:0 0 10px var(--warm)}" +
    ".mem-card:nth-child(3n) .dot{background:radial-gradient(circle at 35% 30%,#fff,var(--gold));box-shadow:0 0 10px var(--gold)}" +
    ".mem-card.disabled{opacity:.6}.mem-card.disabled .dot{background:var(--mask);box-shadow:none}" +
    ".mem-card.pending .dot{background:radial-gradient(circle at 35% 30%,#fff,var(--gold));box-shadow:0 0 10px var(--gold)}" +
    ".mem-card-body{flex:1;min-width:0}" +
    ".mem-card-text{color:var(--ink);font-size:14px;line-height:1.5;word-break:break-word;white-space:pre-wrap}" +
    ".mem-card-meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}" +
    ".mem-meta-chip{font-size:10.5px;letter-spacing:.02em;padding:2px 8px;border-radius:9px;white-space:nowrap;" +
    "color:var(--faint);background:rgba(120,140,190,.08);border:1px solid var(--line)}" +
    ".mem-meta-chip b{color:var(--soft);font-weight:600}" +
    ".mem-meta-chip.status-active{color:#2f97a8;background:rgba(79,195,214,.12);border-color:rgba(79,195,214,.34)}" +
    ".mem-meta-chip.status-pending{color:#a9762a;background:rgba(230,196,120,.16);border-color:rgba(230,196,120,.42)}" +
    ".mem-meta-chip.status-disabled{color:var(--faint);background:rgba(120,140,190,.10)}" +
    ".mem-meta-chip.status-rejected{color:#c0504a;background:rgba(231,120,120,.12);border-color:rgba(231,120,120,.4)}" +
    ".mem-meta-chip.risk{color:#c0504a;background:rgba(231,120,120,.12);border-color:rgba(231,120,120,.4)}" +
    ".mem-card-remove{flex:none;border:1px solid var(--line);background:rgba(255,255,255,.8);color:var(--soft);" +
    "border-radius:50%;width:28px;height:28px;padding:0;font-size:14px;line-height:1;cursor:pointer;" +
    "display:flex;align-items:center;justify-content:center;transition:background .16s,color .16s,border-color .16s}" +
    ".mem-card-remove:hover:not(:disabled){background:rgba(231,120,120,.12);color:#c0504a;border-color:rgba(231,120,120,.4);transform:none;box-shadow:none}" +
    ".mem-card-remove:disabled{opacity:.4;cursor:default}" +
    ".mem-empty{padding:30px 20px;text-align:center;color:var(--faint)}" +
    ".mem-empty-t{font-size:15px;color:var(--soft);margin-bottom:6px}" +
    ".mem-empty-s{font-size:13px;max-width:520px;margin:0 auto;line-height:1.5}" +
    ".mem-add{margin-top:22px;padding:16px 18px}" +
    ".mem-add h2{padding:0 0 4px}" +
    ".mem-add .addrow{display:flex;gap:8px;margin-top:10px}" +
    ".mem-add input[type=text]{flex:1;min-width:0;font:inherit;font-size:14px;border:1px solid var(--line);" +
    "border-radius:22px;padding:10px 15px;background:rgba(255,255,255,.85);outline:none;color:var(--ink);" +
    "transition:border-color .2s,box-shadow .2s}" +
    ".mem-add input[type=text]:focus{border-color:var(--halo);box-shadow:0 0 0 3px rgba(122,167,255,.14)}" +
    ".mem-add input[type=text]:disabled{opacity:.55;cursor:default}" +
    ".mem-add .addhint{color:var(--faint);font-size:12.5px;margin-top:9px;line-height:1.45}" +
    // the SLOW-add busy banner.
    ".mem-busy{display:flex;align-items:center;gap:11px;margin-top:12px;padding:11px 14px;border-radius:12px;" +
    "background:linear-gradient(90deg,rgba(122,167,255,.12),rgba(231,168,196,.10));border:1px solid rgba(122,167,255,.3)}" +
    ".mem-busy .spin{width:15px;height:15px;flex:none;border-radius:50%;border:2px solid rgba(122,167,255,.35);" +
    "border-top-color:var(--halo);animation:memspin .8s linear infinite}" +
    ".mem-busy .busytext{color:var(--soft);font-size:13px}" +
    ".mem-busy .busytext b{color:var(--ink);font-weight:600}" +
    "@keyframes memspin{to{transform:rotate(360deg)}}" +
    ".mem-note{margin-top:8px;font-size:12.5px;padding:8px 12px;border-radius:10px}" +
    ".mem-note.err{color:#c0504a;background:rgba(231,120,120,.10);border:1px solid rgba(231,120,120,.34)}" +
    ".mem-offline{margin:18px 0 0;padding:10px 13px;border-radius:11px;font-size:12.5px;color:var(--faint);" +
    "background:rgba(120,140,190,.06);border:1px solid var(--line)}";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  // ---- card shape normalization (string today; object after D2) --------------------------------
  function isObj(c) { return c && typeof c === "object"; }
  function cardText(c) {
    if (c == null) return "";
    if (typeof c === "string") return c;
    if (isObj(c)) return c.text != null ? String(c.text) : "";
    return String(c);
  }
  // metadata fields to surface as chips, in display order. Anything else scalar on the object is
  // shown generically so a future card can add fields without a code change here.
  var META_ORDER = ["status", "kind", "risk", "source_run_id", "usage_count", "last_used_at", "created_at", "strength"];
  var META_SKIP = { id: 1, text: 1 };
  function cardMeta(c, ctx) {
    if (!isObj(c)) return [];
    var out = [];
    var seen = {};
    var push = function (key, node, cls) {
      if (node == null) return;
      seen[key] = 1;
      out.push({ cls: cls || "", node: node });
    };
    META_ORDER.forEach(function (key) {
      if (!(key in c) || c[key] == null || c[key] === "") return;
      var v = c[key];
      if (key === "status") {
        push("status", chip(S.el("b", {}, [String(v)])), "status-" + String(v).toLowerCase());
      } else if (key === "risk") {
        // truthy or a non-"none" label => flag it
        if (v === true || (typeof v === "string" && v.toLowerCase() !== "none" && v !== "")) {
          push("risk", chip([S.el("b", {}, ["risk"]), typeof v === "string" ? " " + v : ""]), "risk");
        } else { seen[key] = 1; }
      } else if (key === "kind") {
        push("kind", chip(String(v)));
      } else if (key === "source_run_id") {
        push("source_run_id", chip([S.el("b", {}, ["from"]), " " + shortId(String(v))]));
      } else if (key === "usage_count") {
        push("usage_count", chip([String(v), " use" + (Number(v) === 1 ? "" : "s")]));
      } else if (key === "last_used_at") {
        push("last_used_at", chip([S.el("b", {}, ["used"]), " " + fmtWhen(v, ctx)]));
      } else if (key === "created_at") {
        push("created_at", chip([S.el("b", {}, ["added"]), " " + fmtWhen(v, ctx)]));
      } else if (key === "strength") {
        push("strength", chip([S.el("b", {}, ["strength"]), " " + fmtNum(v)]));
      }
    });
    // generic fall-through for any *other* scalar fields a future card ships.
    Object.keys(c).forEach(function (key) {
      if (seen[key] || META_SKIP[key]) return;
      var v = c[key];
      if (v == null || v === "" || typeof v === "object") return;
      out.push({ cls: "", node: chip([S.el("b", {}, [key.replace(/_/g, " ")]), " " + String(v)]) });
    });
    return out;
  }
  function chip(kids) {
    return S.el("span", { class: "mem-meta-chip" }, Array.isArray(kids) ? kids : [kids]);
  }
  function shortId(s) { return s.length > 12 ? s.slice(0, 10) + "…" : s; }
  function fmtNum(v) { var n = Number(v); return isNaN(n) ? String(v) : (n === Math.round(n) ? String(n) : n.toFixed(1)); }
  function fmtWhen(v, ctx) {
    // reuse the shell's time formatting when it looks like a timestamp; else print as-is.
    if (typeof v === "number" || /^\d{4}-\d|Z$|:\d\d/.test(String(v))) {
      var t = ctx.fmtTime ? ctx.fmtTime(v) : String(v);
      var d = ctx.fmtDate ? ctx.fmtDate(v) : "";
      return d ? d + " " + t : t;
    }
    return String(v);
  }
  function cardStatus(c) { return isObj(c) && c.status ? String(c.status).toLowerCase() : ""; }

  // ---- page state ------------------------------------------------------------------------------
  var state = { cards: [], hasPrefix: false, offline: false, adding: false, removing: {} };

  function render(view, ctx) {
    ensureStyle();
    state = { cards: [], hasPrefix: false, offline: false, adding: false, removing: {} };

    var root = S.el("div", { class: "wrap" }, [
      S.el("div", { class: "mem-head" }, [
        S.el("div", {}, [
          S.el("h1", {}, [S.el("span", { class: "glow" }, ["Memory"])]),
          S.el("p", { class: "sub" }, [
            "What the agent has learned to carry across replies — its remembered traits. Adjust how strongly they color responses, add a new trait, or remove one.",
          ]),
        ]),
      ]),

      S.el("div", { class: "mem-offline", id: "mem-offline", style: "display:none" }, [
        "The studio server is not reachable — showing an empty memory. Start ",
        S.el("code", {}, ["research/clozn_server.py"]),
        " (or open this from ",
        S.el("code", {}, ["http://127.0.0.1:8090/app.html"]),
        ") to load and edit memory.",
      ]),

      // ---- memory strength ----
      S.el("div", { class: "mem-strength panel" }, [
        S.el("h2", {}, ["Memory strength"]),
        S.el("div", { class: "strengthrow" }, [
          S.el("input", {
            type: "range", id: "mem-strength", min: "0", max: "2", step: "0.1", value: "1",
            oninput: onStrengthInput, onchange: onStrengthChange, disabled: "disabled",
            "aria-label": "memory strength",
          }, []),
          S.el("b", { class: "strengthval", id: "mem-strength-val" }, ["1.0"]),
        ]),
        S.el("div", { class: "ticks" }, [
          S.el("span", {}, ["off"]),
          S.el("span", {}, ["normal"]),
          S.el("span", {}, ["stronger"]),
        ]),
        S.el("div", { class: "strengthhint" }, [
          "How strongly memory colors replies. ",
          S.el("b", {}, ["0"]), " turns learned traits off, ",
          S.el("b", {}, ["1"]), " is normal, up to ",
          S.el("b", {}, ["2"]), " leans on them harder (can over-bleed into unrelated answers).",
        ]),
      ]),

      // ---- card list ----
      S.el("div", { class: "mem-listhead" }, [
        S.el("h2", {}, ["Learned traits"]),
        S.el("span", { class: "mem-count", id: "mem-count" }, [""]),
      ]),
      S.el("div", { class: "mem-list panel", id: "mem-list" }, [
        S.el("div", { class: "mem-empty" }, ["Loading memory…"]),
      ]),

      // ---- add a trait ----
      S.el("div", { class: "mem-add panel" }, [
        S.el("h2", {}, ["Add a trait"]),
        S.el("div", { class: "addrow" }, [
          S.el("input", {
            type: "text", id: "mem-add-input", autocomplete: "off",
            placeholder: "e.g. prefers concise answers with concrete examples",
            onkeydown: function (e) { if (e.key === "Enter") submitAdd(ctx); },
          }, []),
          S.el("button", { class: "go", id: "mem-add-btn", onclick: function () { submitAdd(ctx); } }, ["Learn"]),
        ]),
        S.el("div", { class: "addhint" }, [
          "Describe a lasting preference or fact in a short sentence. Clozn folds it into the model's memory prefix — ",
          S.el("b", {}, ["this trains and takes a while."]),
        ]),
        // slow-add busy banner (hidden until a learn is running)
        S.el("div", { class: "mem-busy", id: "mem-busy", style: "display:none" }, [
          S.el("div", { class: "spin" }, []),
          S.el("span", { class: "busytext", id: "mem-busy-text" }, [
            S.el("b", {}, ["Learning this…"]), " baking the trait into memory (this can take a minute).",
          ]),
        ]),
        S.el("div", { class: "mem-note err", id: "mem-note", style: "display:none" }, []),
      ]),
    ]);

    view.appendChild(root);
    loadStrength(ctx);
    loadCards(ctx);
  }

  // ---- strength --------------------------------------------------------------------------------
  function onStrengthInput() {
    var val = document.getElementById("mem-strength-val");
    var sl = document.getElementById("mem-strength");
    if (val && sl) val.textContent = (+sl.value).toFixed(1);
  }
  function onStrengthChange() {
    var sl = document.getElementById("mem-strength");
    if (!sl) return;
    var v = parseFloat(sl.value);
    // fire-and-forget; guarded so a failure never throws to the page.
    S.postJSON("/memory/strength", { value: v }, null);
  }
  function loadStrength(ctx) {
    ctx.postJSON("/memory/strength", {}, null).then(function (d) {
      var sl = document.getElementById("mem-strength");
      var val = document.getElementById("mem-strength-val");
      if (!sl) return;
      if (d && d.strength != null) {
        var s = Math.max(0, Math.min(2, +d.strength));
        sl.value = String(s);
        if (val) val.textContent = s.toFixed(1);
        sl.disabled = false;
      } else {
        // offline / no data: leave the control disabled at its default so we don't imply a live value.
        markOffline(true);
      }
    });
  }

  // ---- cards -----------------------------------------------------------------------------------
  function loadCards(ctx) {
    return ctx.postJSON("/memory/cards", {}, null).then(function (d) {
      if (d == null) { state.offline = true; markOffline(true); state.cards = []; }
      else {
        state.offline = false; markOffline(false);
        state.cards = (d && d.cards) || [];
        state.hasPrefix = !!(d && d.has_prefix);
      }
      state.removing = {};
      drawCards(ctx);
    });
  }

  function markOffline(on) {
    var box = document.getElementById("mem-offline");
    if (box) box.style.display = on ? "" : "none";
    if (on) {
      var sl = document.getElementById("mem-strength");
      if (sl) sl.disabled = true;
    }
  }

  function drawCards(ctx) {
    var list = document.getElementById("mem-list");
    var count = document.getElementById("mem-count");
    if (!list) return;
    list.innerHTML = "";
    var cards = state.cards || [];

    if (count) count.textContent = cards.length ? String(cards.length) + (cards.length === 1 ? " trait" : " traits") : "";

    if (!cards.length) {
      list.appendChild(S.el("div", { class: "mem-empty" }, [
        S.el("div", { class: "mem-empty-t" }, [state.offline ? "Memory unavailable." : "No memory yet."]),
        S.el("div", { class: "mem-empty-s" }, [
          state.offline
            ? "Connect the studio server to view and edit what the agent remembers."
            : "The agent hasn't learned any traits. Add one below (or let it propose memories from your conversations) and it will remember it across replies.",
        ]),
      ]));
      return;
    }
    cards.forEach(function (c, i) { list.appendChild(cardRow(c, i, ctx)); });
  }

  function cardRow(c, i, ctx) {
    var status = cardStatus(c);
    var busy = !!state.removing[i];
    var cls = "mem-card" + (status ? " " + status : "") + (busy ? " busy" : "");

    var metas = cardMeta(c, ctx);
    var body = [S.el("div", { class: "mem-card-text" }, [cardText(c) || "(empty trait)"])];
    if (metas.length) {
      body.push(S.el("div", { class: "mem-card-meta" }, metas.map(function (m) {
        if (m.cls) m.node.className = "mem-meta-chip " + m.cls;
        return m.node;
      })));
    }

    var removeBtn = S.el("button", {
      class: "mem-card-remove", title: "remove this trait", "aria-label": "remove trait",
      onclick: function () { removeCard(i, ctx); },
    }, [busy ? "…" : "×"]);
    if (busy || state.adding) removeBtn.disabled = true;

    return S.el("div", { class: cls }, [
      S.el("span", { class: "dot" }, []),
      S.el("div", { class: "mem-card-body" }, body),
      removeBtn,
    ]);
  }

  function removeCard(i, ctx) {
    if (state.adding || state.removing[i]) return;
    state.removing[i] = true;
    drawCards(ctx); // reflect the busy state on that row + disable buttons
    ctx.postJSON("/memory/remove", { index: i }, null).then(function (res) {
      if (res == null) {
        // failed: clear busy, surface a note, keep the card.
        state.removing[i] = false;
        showNote("Couldn't remove that trait — is the studio server up?");
        drawCards(ctx);
        return;
      }
      // rebuilt: reload the authoritative card list (indices shift after removal).
      loadCards(ctx);
    });
  }

  // ---- add (SLOW: trains the prefix) -----------------------------------------------------------
  function submitAdd(ctx) {
    if (state.adding) return;
    var input = document.getElementById("mem-add-input");
    var text = input ? String(input.value || "").trim() : "";
    if (!text) { if (input) input.focus(); return; }

    hideNote();
    state.adding = true;
    setAddBusy(true);
    drawCards(ctx); // disable per-card remove buttons while training

    // staged, honest progress text: reading -> training -> still working.
    var t0 = Date.now();
    var stages = [
      "reading how it should respond…",
      "training the memory prefix…",
      "still working — larger models take longer…",
    ];
    var busyText = document.getElementById("mem-busy-text");
    var tick = setInterval(function () {
      var s = Math.round((Date.now() - t0) / 1000);
      var stage = s < 15 ? stages[0] : (s < 45 ? stages[1] : stages[2]);
      if (busyText) {
        busyText.innerHTML = "";
        busyText.appendChild(S.el("b", {}, ["Learning this…"]));
        busyText.appendChild(document.createTextNode(" " + stage + " · " + s + "s"));
      }
    }, 1000);

    ctx.postJSON("/memory/add", { text: text }, null).then(function (res) {
      clearInterval(tick);
      state.adding = false;
      setAddBusy(false);
      if (res == null) {
        showNote("Couldn't learn that trait — is the studio server up? Nothing was changed.");
        drawCards(ctx); // re-enable remove buttons
        return;
      }
      if (input) input.value = "";
      // refresh the authoritative list (the add re-trained + may normalize the text).
      loadCards(ctx).then(function () {
        var inp = document.getElementById("mem-add-input");
        if (inp) inp.focus();
      });
    });
  }

  function setAddBusy(on) {
    var btn = document.getElementById("mem-add-btn");
    var input = document.getElementById("mem-add-input");
    var busy = document.getElementById("mem-busy");
    if (btn) { btn.disabled = on; btn.textContent = on ? "Learning…" : "Learn"; }
    if (input) input.disabled = on;
    if (busy) busy.style.display = on ? "" : "none";
  }

  function showNote(msg) {
    var n = document.getElementById("mem-note");
    if (!n) return;
    n.textContent = msg;
    n.style.display = "";
  }
  function hideNote() {
    var n = document.getElementById("mem-note");
    if (n) { n.textContent = ""; n.style.display = "none"; }
  }

  S.register("memory", { title: "Memory", render: render });
})();
