/* runs.js -- the Runs page (the product centerpiece).  Issue B5.
 *
 * Lists every run newest-first (GET /runs). Each row:
 *   time (created_at) · source/client · prompt_summary · #memories (memory.cards_applied.length)
 *   · active dials (behavior.active_dials keys) · duration_ms · flag chips.
 * Filter chips: by source, and by flag (memory / low-confidence / error / replayed / steered).
 * Clicking a row navigates to #/run/<id>.
 *
 * Flags come pre-computed on each run (runlog._flags): memory, pending-memory, steered, replayed,
 * error, low-confidence, long. We render/filter a friendly subset.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  // flag key -> {label, css}. Order defines chip order.
  var FLAG_META = {
    memory: { label: "memory", css: "f-memory" },
    steered: { label: "steered", css: "f-steered" },
    "pending-memory": { label: "pending memory", css: "f-pending" },
    "low-confidence": { label: "low confidence", css: "f-low" },
    replayed: { label: "replayed", css: "f-replay" },
    error: { label: "error", css: "f-error" },
    long: { label: "long", css: "f-long" },
  };
  // which flags get a filter chip (per the issue)
  var FILTER_FLAGS = ["memory", "low-confidence", "error", "replayed", "steered"];

  var state = { runs: [], source: null, flag: null };

  function render(view, ctx) {
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("div", { class: "runshead" }, [
          S.el("div", {}, [
            S.el("h1", {}, [S.el("span", { class: "glow" }, ["Runs"])]),
            S.el("p", { class: "sub" }, [
              "Every request from any connected client, newest first. Click a run to inspect what memory, dials, and runtime state influenced it.",
            ]),
          ]),
          S.el("button", { id: "runsrefresh", title: "reload" }, ["Refresh"]),
        ]),
        S.el("div", { class: "filters", id: "filters" }, []),
        S.el("div", { class: "runlist panel", id: "runlist" }, [
          S.el("div", { class: "runloading" }, ["Loading runs…"]),
        ]),
      ])
    );

    document.getElementById("runsrefresh").addEventListener("click", function () { load(ctx); });
    load(ctx);
  }

  function load(ctx) {
    var list = document.getElementById("runlist");
    if (list) list.innerHTML = '<div class="runloading">Loading runs…</div>';
    ctx.getJSON("/runs", { runs: [] }).then(function (d) {
      state.runs = (d && d.runs) || [];
      draw(ctx);
    });
  }

  function draw(ctx) {
    drawFilters(ctx);
    drawList(ctx);
  }

  function sourcesInData() {
    var seen = {};
    state.runs.forEach(function (r) { if (r && r.source) seen[r.source] = true; });
    return Object.keys(seen).sort();
  }

  function drawFilters(ctx) {
    var wrap = document.getElementById("filters");
    if (!wrap) return;
    wrap.innerHTML = "";

    // source group
    var sources = sourcesInData();
    if (sources.length) {
      wrap.appendChild(S.el("span", { class: "filterlabel" }, ["source"]));
      wrap.appendChild(mkChip("all", state.source === null, function () { state.source = null; draw(ctx); }));
      sources.forEach(function (s) {
        wrap.appendChild(mkChip(prettySource(s), state.source === s, function () { state.source = s; draw(ctx); }));
      });
    }

    // flag group
    wrap.appendChild(S.el("span", { class: "filterlabel" }, ["flag"]));
    wrap.appendChild(mkChip("any", state.flag === null, function () { state.flag = null; draw(ctx); }));
    FILTER_FLAGS.forEach(function (f) {
      var meta = FLAG_META[f] || { label: f };
      wrap.appendChild(mkChip(meta.label, state.flag === f, function () { state.flag = f; draw(ctx); }));
    });
  }

  function mkChip(label, active, onclick) {
    return S.el("button", { class: "fchip" + (active ? " on" : ""), onclick: onclick }, [label]);
  }

  function filtered() {
    return state.runs.filter(function (r) {
      if (!r) return false;
      if (state.source && r.source !== state.source) return false;
      if (state.flag && (r.flags || []).indexOf(state.flag) < 0) return false;
      return true;
    });
  }

  function drawList(ctx) {
    var list = document.getElementById("runlist");
    if (!list) return;
    var rows = filtered();
    list.innerHTML = "";
    if (!state.runs.length) {
      list.appendChild(S.el("div", { class: "runempty" }, [
        S.el("div", { class: "runempty-t" }, ["No runs yet."]),
        S.el("div", { class: "runempty-s" }, [
          "Point an OpenAI-compatible client at the endpoint (see the Agent page) and send a request — it will appear here.",
        ]),
      ]));
      return;
    }
    if (!rows.length) {
      list.appendChild(S.el("div", { class: "runempty" }, [
        S.el("div", { class: "runempty-t" }, ["No runs match these filters."]),
      ]));
      return;
    }
    rows.forEach(function (r) { list.appendChild(runRow(r, ctx)); });
  }

  function runRow(r, ctx) {
    var mem = r.memory || {};
    var nMem = (mem.cards_applied || []).length;
    var dials = Object.keys((r.behavior || {}).active_dials || {});
    var dur = ((r.timing || {}).duration_ms);

    // flag chips (skip 'long' visual noise unless it's the only signal? keep it — cheap + honest)
    var flags = (r.flags || []).filter(function (f) { return FLAG_META[f]; });
    var flagChips = flags.map(function (f) {
      var m = FLAG_META[f];
      return S.el("span", { class: "rflag " + m.css }, [m.label]);
    });

    var dialChips = dials.slice(0, 4).map(function (d) {
      return S.el("span", { class: "rdial" }, [d]);
    });
    if (dials.length > 4) dialChips.push(S.el("span", { class: "rdial more" }, ["+" + (dials.length - 4)]));

    var row = S.el("div", { class: "runrow", tabindex: "0", role: "button" }, [
      S.el("div", { class: "rc rc-time" }, [
        S.el("span", { class: "rtime" }, [ctx.fmtTime(r.created_at)]),
        S.el("span", { class: "rdate" }, [ctx.fmtDate(r.created_at)]),
      ]),
      S.el("div", { class: "rc rc-source" }, [
        S.el("span", { class: "rsource" }, [prettySource(r.source)]),
        r.client && r.client !== "unknown" ? S.el("span", { class: "rclient" }, [r.client]) : null,
      ].filter(Boolean)),
      S.el("div", { class: "rc rc-prompt" }, [
        S.el("span", { class: "rprompt" }, [r.prompt_summary || "(no prompt)"]),
      ]),
      S.el("div", { class: "rc rc-mem" }, [
        nMem
          ? S.el("span", { class: "rmem on", title: (mem.cards_applied || []).join("\n") }, [String(nMem) + (nMem === 1 ? " memory" : " memories")])
          : S.el("span", { class: "rmem" }, ["—"]),
      ]),
      S.el("div", { class: "rc rc-dials" }, dialChips.length ? dialChips : [S.el("span", { class: "rmem" }, ["—"])]),
      S.el("div", { class: "rc rc-dur" }, [ctx.fmtDuration(dur)]),
      S.el("div", { class: "rc rc-flags" }, flagChips),
    ]);

    function open() { if (r.id) ctx.navigate("run/" + r.id); }
    row.addEventListener("click", open);
    row.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
    });
    return row;
  }

  // "openai_api" -> "OpenAI API", "studio_chat" -> "Studio", etc.
  function prettySource(s) {
    if (!s) return "unknown";
    var map = {
      openai_api: "OpenAI API",
      studio_chat: "Studio",
      engine_chat: "Engine",
      cli: "CLI",
      denoise: "Denoise",
    };
    return map[s] || s;
  }

  S.register("runs", { title: "Runs", render: render });
})();
