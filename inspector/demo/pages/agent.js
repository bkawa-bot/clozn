/* agent.js -- the Agent (home / status) page.  Issue A3.
 *
 * A calm, productized status home (NOT raw internals -- those live in Lab). Reads:
 *   GET  /substrate        -> {active, available:[...]}
 *   POST /steer/axes  {}   -> {axes:[{name,value,...}]}
 *   POST /memory/cards {}  -> {cards:[...]}   (cards are currently plain strings)
 *   GET  /runs             -> {runs:[...]}    (newest first)
 *   GET  /engine/health    -> {engine:...}    (best-effort; tolerate 502)
 * Shows: runtime running, active substrate + model, the copyable OpenAI endpoint (<base>/v1),
 * memory count, active dials, recent-runs count, engine-hook availability.
 * Buttons: Copy endpoint, Open latest run, Test prompt (inline POST /v1/chat/completions).
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  function statusRow(label, value, opts) {
    opts = opts || {};
    var v = S.el("span", { class: "sval" + (opts.mono ? " mono" : "") }, [value]);
    var dot = opts.dot ? S.el("span", { class: "sdot " + opts.dot }) : null;
    return S.el("div", { class: "srow" }, [
      S.el("span", { class: "slabel" }, [label]),
      S.el("span", { class: "sval-wrap" }, [dot, v].filter(Boolean)),
    ]);
  }

  function chip(text, cls) {
    return S.el("span", { class: "chip " + (cls || "") }, [text]);
  }

  function render(view, ctx) {
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("h1", {}, [S.el("span", { class: "glow" }, ["Agent"])]),
        S.el("p", { class: "sub" }, [
          "Your local runtime and the endpoint your tools connect to. Keep using your normal clients — Clozn runs underneath and captures every request as an inspectable run.",
        ]),
        S.el("div", { class: "agentgrid", id: "agentgrid" }, [
          statusCardShell(),
          testCardShell(ctx),
        ]),
      ])
    );

    // ---- fill status (guarded; each fetch degrades to a placeholder) --------------------------
    var endpoint = ctx.endpoint;
    var body = document.getElementById("statusbody");
    var actions = document.getElementById("statusactions");

    // endpoint is known immediately; wire Copy right away.
    var copyBtn = S.el("button", { class: "go" }, ["Copy endpoint"]);
    copyBtn.addEventListener("click", function () {
      ctx.copyText(endpoint).then(function (ok) {
        copyBtn.textContent = ok ? "Copied ✓" : "Copy failed";
        setTimeout(function () { copyBtn.textContent = "Copy endpoint"; }, 1400);
      });
    });
    var latestBtn = S.el("button", {}, ["Open latest run"]);
    latestBtn.disabled = true;
    latestBtn.addEventListener("click", function () {
      if (latestBtn.dataset.rid) ctx.navigate("run/" + latestBtn.dataset.rid);
    });
    var labBtn = S.el("button", {}, ["Open Lab"]);
    labBtn.addEventListener("click", function () { ctx.navigate("lab"); });
    actions.appendChild(copyBtn);
    actions.appendChild(latestBtn);
    actions.appendChild(labBtn);

    Promise.all([
      ctx.getJSON("/substrate", {}),
      ctx.postJSON("/memory/cards", {}, {}),
      ctx.postJSON("/steer/axes", {}, {}),
      ctx.getJSON("/runs", {}),
      // /engine/health is best-effort: 502 when the C++ engine isn't attached. getJSON returns null then.
      ctx.getJSON("/engine/health", null),
    ]).then(function (r) {
      var sub = r[0] || {};
      var cardsResp = r[1] || {};
      var axesResp = r[2] || {};
      var runsResp = r[3] || {};
      var engine = r[4];

      var reachable = !!(r[0] || r[1] || r[2] || r[3]); // any endpoint answered => server is up
      var cards = cardsResp.cards || [];
      var axes = axesResp.axes || [];
      var runs = runsResp.runs || [];

      var activeSub = sub.active || null;
      var model = modelName(sub, runs);

      // active dials = axes whose |value| is meaningfully non-zero
      var activeDials = axes.filter(function (a) { return Math.abs(+a.value || 0) >= 0.05; });
      var runsToday = runs.filter(function (x) { return ctx.isToday(x.created_at); }).length;

      // engine health: null => unreachable/502 (tolerated), object => connected
      var engineOK = !!engine && !engine.error;

      body.innerHTML = "";
      body.appendChild(statusRow("Local runtime",
        reachable ? "running" : "not reachable",
        { dot: reachable ? "ok" : "off" }));
      body.appendChild(statusRow("Active substrate", activeSub ? cap(activeSub) : "unknown"));
      body.appendChild(statusRow("Model", model || "—", { mono: true }));

      var epRow = statusRow("Endpoint", endpoint, { mono: true });
      body.appendChild(epRow);

      body.appendChild(statusRow("Memory",
        cards.length ? cards.length + (cards.length === 1 ? " memory" : " memories") : "none",
        { dot: cards.length ? "ok" : "idle" }));

      var dialWrap = S.el("span", { class: "chips" },
        activeDials.length
          ? activeDials.map(function (a) { return chip(a.name, "steer"); })
          : [S.el("span", { class: "sval faintv" }, ["none active"])]);
      body.appendChild(S.el("div", { class: "srow" }, [
        S.el("span", { class: "slabel" }, ["Active dials"]),
        dialWrap,
      ]));

      body.appendChild(statusRow("Recent runs today", String(runsToday), { dot: runsToday ? "ok" : "idle" }));
      body.appendChild(statusRow("Engine hooks",
        engineOK ? "connected" : "not attached",
        { dot: engineOK ? "ok" : "idle" }));

      // wire "Open latest run" to the newest run id
      if (runs.length && runs[0] && runs[0].id) {
        latestBtn.disabled = false;
        latestBtn.dataset.rid = runs[0].id;
      } else {
        latestBtn.textContent = "No runs yet";
      }

      if (!reachable) {
        var warn = S.el("div", { class: "notewarn" }, [
          "Couldn't reach the studio server. Start it with ",
          S.el("code", {}, ["clozn studio"]),
          " (default port 8090), then reload.",
        ]);
        body.appendChild(warn);
      }
    });
  }

  function statusCardShell() {
    return S.el("section", { class: "panel statuscard" }, [
      S.el("h2", {}, ["runtime status"]),
      S.el("div", { class: "srows", id: "statusbody" }, [
        S.el("div", { class: "srow" }, [
          S.el("span", { class: "slabel" }, ["Local runtime"]),
          S.el("span", { class: "sval faintv" }, ["checking…"]),
        ]),
      ]),
      S.el("div", { class: "cardactions", id: "statusactions" }, []),
    ]);
  }

  function testCardShell(ctx) {
    var out = S.el("div", { class: "testout", id: "testout" }, []);
    var input = S.el("textarea", {
      class: "testinput", id: "testprompt", rows: "3",
      placeholder: "Test the runtime — e.g. “Explain what a KV cache is in one sentence.”",
    }, []);
    var send = S.el("button", { class: "go", id: "testsend" }, ["Test prompt"]);

    function run() {
      var text = (input.value || "").trim();
      if (!text) { input.focus(); return; }
      send.disabled = true; send.textContent = "Running…";
      out.innerHTML = "";
      out.appendChild(S.el("div", { class: "who" }, ["Reply"]));
      var bubble = S.el("div", { class: "testreply" }, ["…"]);
      out.appendChild(bubble);
      ctx.postJSON("/v1/chat/completions", {
        messages: [{ role: "user", content: text }],
        max_tokens: 256,
      }, null).then(function (d) {
        var reply = pickReply(d);
        bubble.textContent = reply || "(no response — is the model loaded?)";
        send.disabled = false; send.textContent = "Test prompt";
      });
    }
    send.addEventListener("click", run);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
    });

    return S.el("section", { class: "panel testcard" }, [
      S.el("h2", {}, ["test prompt"]),
      S.el("p", { class: "cardhint" }, ["A quick sanity check that the runtime answers. This POSTs the same OpenAI endpoint your clients use. ⌘/Ctrl+Enter to send."]),
      input,
      S.el("div", { class: "cardactions" }, [send]),
      out,
    ]);
  }

  // OpenAI-shaped reply extractor, tolerant of shapes/errors.
  function pickReply(d) {
    if (!d) return "";
    if (d.error) return "[" + (d.error.message || d.error) + "]";
    try {
      var c = d.choices && d.choices[0];
      if (c && c.message && c.message.content != null) return c.message.content;
      if (c && c.text != null) return c.text;
    } catch (e) {}
    return "";
  }

  // best-effort model label: prefer /substrate, else the newest run's model.
  function modelName(sub, runs) {
    if (sub && sub.model) return sub.model;
    if (runs && runs.length && runs[0] && runs[0].model) return runs[0].model;
    if (sub && sub.active) return "clozn-" + sub.active;
    return "";
  }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  S.register("agent", { title: "Agent", render: render });
})();
