/* behavior.js -- the Behavior page.  Issue G1.
 *
 * "How it responds" -- plain-language dials over the model's tone/cognition, a make-your-own-dial
 * panel, a test console to feel the current settings, and a "replay the latest run with these dials"
 * before/after. NO jargon: the user never sees "activation steering"; they see concise <-> detailed.
 *
 * Backend (research/clozn_server.py, all guarded so the page renders offline):
 *   POST /steer/axes {}                     -> {axes:[{name, value, poles:[hi,lo], max, custom?}], ...}
 *   POST /steer/set {name, value}           (debounced on drag)
 *   POST /steer/custom {name, pos, neg}     -> reload sliders
 *   POST /steer/custom_delete {name}        -> reload sliders
 *   POST /v1/chat/completions {messages,max_tokens}  (test console; non-streaming)
 *   GET  /runs                              (newest id for replay)
 *   POST /runs/<id>/replay {changes_applied:{behavior_overrides:{...}}}   (F1 -- may 404 for now)
 *
 * Ports studio.html's loadSliders + custom-dial panel into a native page (no iframe). Reuses clozn.css.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  // in-page state: the axes as last fetched, keyed by name -> current value (so replay/test read live).
  var state = { axes: [], values: {} };
  var setTimer = null; // debounce handle for /steer/set

  // ---- plain-language helpers -----------------------------------------------------------------
  // poles arrive as [hi, lo] (see /steer/axes). We read them left-to-right as "lo <-> hi", e.g.
  // "concise <-> detailed", so a positive value leans toward `hi` and negative toward `lo`.
  function poleHi(a) { return (a.poles && a.poles[0]) || a.name; }
  function poleLo(a) { return (a.poles && a.poles[1]) || ""; }
  function axisLabel(a) {
    var lo = poleLo(a), hi = poleHi(a);
    return lo ? lo + " ↔ " + hi : hi;
  }
  // a friendly "leaning ___" readout for the current value.
  function leaning(a, v) {
    if (Math.abs(v) < 0.05) return "balanced";
    return "leaning " + (v > 0 ? poleHi(a) : poleLo(a));
  }

  // ---- render ---------------------------------------------------------------------------------
  function render(view, ctx) {
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.pageHead({
          kicker: "how it responds",
          kickerRight: "tune · test · replay",
          title: "behavior",
          counter: "dials over its tone and thinking — drag, test, then replay a real run to compare",
        }),

        // the dials
        S.el("div", { class: "panel", style: "margin-top:18px" }, [
          S.el("h2", {}, ["Dials"]),
          S.el("div", { class: "bx" }, [
            S.el("div", { class: "sliders", id: "bx-sliders" }, [
              S.el("div", { class: "bx-loading" }, ["Loading dials…"]),
            ]),
            mkDialMaker(ctx),
          ]),
        ]),

        // the test console
        S.el("div", { class: "panel", style: "margin-top:18px" }, [
          S.el("h2", {}, ["Try it"]),
          S.el("div", { class: "bx" }, [
            S.el("p", { class: "bx-hint" }, [
              "Send a prompt through the model with the dials above applied. Non-streaming — the whole reply arrives at once.",
            ]),
            S.el("div", { class: "composer", style: "padding:0" }, [
              S.el("input", {
                id: "bx-prompt", type: "text", autocomplete: "off",
                placeholder: "e.g. Explain how a rainbow forms.",
              }, []),
              S.el("button", { class: "go", id: "bx-testbtn" }, ["Test"]),
            ]),
            S.el("div", { class: "bx-reply", id: "bx-reply" }, []),
          ]),
        ]),

        // replay the latest real run with these dials
        S.el("div", { class: "panel", style: "margin-top:18px" }, [
          S.el("h2", {}, ["Replay your latest request"]),
          S.el("div", { class: "bx" }, [
            S.el("p", { class: "bx-hint" }, [
              "Take your most recent real request and re-run it with the current dials — see the original answer beside the new one.",
            ]),
            S.el("button", { class: "go", id: "bx-replaybtn" }, ["Replay latest with these dials"]),
            S.el("div", { class: "bx-replaymsg", id: "bx-replaymsg" }, []),
            S.el("div", { class: "cmp", id: "bx-compare" }, []),
          ]),
        ]),
      ])
    );

    wireTest(ctx);
    wireReplay(ctx);
    loadSliders(ctx);
  }

  // ---- the sliders (ported from studio.html loadSliders) --------------------------------------
  function loadSliders(ctx) {
    ctx.postJSON("/steer/axes", {}, { axes: [] }).then(function (d) {
      state.axes = (d && d.axes) || [];
      state.values = {};
      state.axes.forEach(function (a) { state.values[a.name] = +a.value || 0; });
      drawSliders(ctx);
    });
  }

  function drawSliders(ctx) {
    var host = document.getElementById("bx-sliders");
    if (!host) return;
    host.innerHTML = "";
    if (!state.axes.length) {
      host.appendChild(S.el("div", { class: "bx-empty" }, [
        "No dials available. The behavior model may still be waking up — boot research/clozn_server.py, then Refresh the page.",
      ]));
      return;
    }
    state.axes.forEach(function (a) { host.appendChild(mkSlider(a, ctx)); });
  }

  function mkSlider(a, ctx) {
    var mx = a.max || 1.5;
    var v = +a.value || 0;
    // Calibration fields (a.calibrated/a.works/a.usable_range/a.derail_point) are ALL additive -- absent
    // for any dial the server hasn't calibrated (or when /steer/axes is offline, via the postJSON
    // fallback), so `dead` is false and every branch below renders exactly as it did before this existed.
    var dead = a.calibrated === true && a.works === false;   // calibrated on THIS model, but no usable range

    var valEl = S.el("span", { class: "bx-val" }, [v.toFixed(2)]);
    var leanEl = S.el("span", { class: "bx-lean" }, [leaning(a, v)]);

    var range = S.el("input", {
      type: "range", min: String(-mx), max: String(mx), step: "0.05", value: String(v),
      "data-name": a.name, class: "bx-range", disabled: dead ? "disabled" : null,
    }, []);
    if (!dead) {
      range.addEventListener("input", function () {
        var nv = parseFloat(range.value);
        state.values[a.name] = nv;
        valEl.textContent = nv.toFixed(2);
        leanEl.textContent = leaning(a, nv);
        clearTimeout(setTimer);
        setTimer = setTimeout(function () {
          ctx.postJSON("/steer/set", { name: a.name, value: nv }, null);
        }, 160);
      });
    }

    // label line: plain-language poles, a "yours" tag + delete for custom dials, and the value.
    var labelKids = [S.el("b", {}, [axisLabel(a)])];
    if (a.custom) {
      labelKids.push(S.el("span", { class: "bx-tag" }, ["yours"]));
      labelKids.push(S.el("span", {
        class: "bx-del", title: "delete this dial",
        onclick: function () { deleteDial(a.name, ctx); },
      }, ["✕"]));
    }

    var kids = [
      S.el("div", { class: "bx-lab" }, [
        S.el("span", {}, labelKids),
        S.el("span", { class: "bx-meta" }, [leanEl, valEl]),
      ]),
      range,
      S.el("div", { class: "bx-poles" }, [
        S.el("span", {}, [poleLo(a) || "–"]),
        S.el("span", {}, [poleHi(a)]),
      ]),
    ];
    // calibration readout -- a disabled-dial note, or a working-dial hint, never both, never for an
    // uncalibrated dial (a.calibrated falsy skips this whole block, unchanged from today).
    if (dead) {
      kids.push(S.el("p", { class: "bx-calnote bx-calnote-dead" }, ["no measurable effect on this model"]));
    } else if (a.calibrated === true && a.works === true) {
      kids.push(S.el("p", { class: "bx-calnote" }, [calibrationHint(a)]));
    }

    return S.el("div", { class: "bx-slider" + (dead ? " bx-slider-dead" : "") }, kids);
  }

  // "works 0.25–1.0[ · derails past 1.5]" -- a.usable_range is [dead_below, usable_max] (server-side
  // naming; either end may in principle be null, though the server only ever sends works:true alongside a
  // complete range). a.derail_point uses `!= null` (not a truthy check) since 0 is itself a valid derail
  // point (an already-degenerate baseline) per research/dial_autocalibrate.py's own docs.
  function calibrationHint(a) {
    var r = a.usable_range || [];
    var lo = r[0], hi = r[1];
    var txt = "works " + (lo != null ? fmtDial(lo) : "?") + "–" + (hi != null ? fmtDial(hi) : "?");
    if (a.derail_point != null) txt += " · derails past " + fmtDial(a.derail_point);
    return txt;
  }
  function fmtDial(n) { return String(Math.round(+n * 100) / 100); }

  function deleteDial(name, ctx) {
    ctx.postJSON("/steer/custom_delete", { name: name }, null).then(function () {
      loadSliders(ctx);
    });
  }

  // ---- make-your-own-dial panel (ported from studio.html) -------------------------------------
  function mkDialMaker(ctx) {
    var nameI = S.el("input", { id: "bx-dname", maxlength: "24", autocomplete: "off",
      placeholder: "name — e.g. skeptical" }, []);
    var posI = S.el("input", { id: "bx-dpos", autocomplete: "off",
      placeholder: "＋ pole — responds with sharp skepticism, demands evidence" }, []);
    var negI = S.el("input", { id: "bx-dneg", autocomplete: "off",
      placeholder: "－ pole — accepts claims with credulous enthusiasm" }, []);
    var msg = S.el("span", { class: "bx-dmsg", id: "bx-dmsg" }, []);
    var createBtn = S.el("button", { class: "go", id: "bx-dcreate" }, ["Create dial"]);

    var form = S.el("div", { class: "bx-dialform", id: "bx-dialform", style: "display:none" }, [
      S.el("p", { class: "bx-hint", style: "margin:0 0 8px" }, [
        "Describe the two ends in plain language. A new slider between them appears.",
      ]),
      S.el("div", { class: "bx-addrow" }, [nameI]),
      S.el("div", { class: "bx-addrow" }, [posI]),
      S.el("div", { class: "bx-addrow" }, [negI]),
      S.el("div", { class: "bx-addrow" }, [createBtn, msg]),
    ]);

    var toggle = S.el("button", { class: "bx-newdial", id: "bx-dialtoggle" }, ["＋ make your own dial"]);
    toggle.addEventListener("click", function () {
      var open = form.style.display === "none";
      form.style.display = open ? "block" : "none";
      if (open) nameI.focus();
    });

    function create() {
      var name = nameI.value.trim(), pos = posI.value.trim(), neg = negI.value.trim();
      if (!name || !pos || !neg) { msg.textContent = "need a name + both poles"; return; }
      createBtn.disabled = true;
      msg.textContent = "computing the direction…";
      ctx.postJSON("/steer/custom", { name: name, pos: pos, neg: neg }, { error: "failed" })
        .then(function (r) {
          createBtn.disabled = false;
          if (r && r.error) { msg.textContent = r.error; return; }
          nameI.value = posI.value = negI.value = "";
          msg.textContent = "";
          form.style.display = "none";
          loadSliders(ctx);
        });
    }
    createBtn.addEventListener("click", create);
    negI.addEventListener("keydown", function (e) { if (e.key === "Enter") create(); });

    return S.el("div", { class: "bx-dialmaker" }, [toggle, form]);
  }

  // ---- test console ---------------------------------------------------------------------------
  function wireTest(ctx) {
    var btn = document.getElementById("bx-testbtn");
    var input = document.getElementById("bx-prompt");
    if (!btn || !input) return;
    function run() { runTest(ctx); }
    btn.addEventListener("click", run);
    input.addEventListener("keydown", function (e) { if (e.key === "Enter") run(); });
  }

  function runTest(ctx) {
    var input = document.getElementById("bx-prompt");
    var out = document.getElementById("bx-reply");
    var btn = document.getElementById("bx-testbtn");
    if (!input || !out) return;
    var prompt = input.value.trim();
    if (!prompt) return;
    if (btn) btn.disabled = true;
    out.className = "bx-reply on";
    out.innerHTML = "";
    out.appendChild(S.el("div", { class: "bx-spinrow" }, [
      S.el("span", { class: "bx-spin" }, []), S.el("span", {}, ["thinking with the current dials…"]),
    ]));

    ctx.postJSON("/v1/chat/completions", {
      messages: [{ role: "user", content: prompt }], max_tokens: 120,
    }, { error: "the studio server is not connected" }).then(function (d) {
      if (btn) btn.disabled = false;
      out.innerHTML = "";
      var reply = extractReply(d);
      if (reply == null) {
        var err = (d && d.error) ? String(d.error) : "no response";
        out.appendChild(S.el("div", { class: "bx-err" }, ["[" + err + "]"]));
        return;
      }
      out.appendChild(S.el("div", { class: "who" }, ["reply"]));
      out.appendChild(S.el("div", { class: "bx-replytext" }, [reply || "(empty reply)"]));
    });
  }

  // Pull the assistant text out of an OpenAI-shaped response; null if it wasn't a valid reply.
  function extractReply(d) {
    if (!d || d.error) return null;
    var ch = d.choices && d.choices[0];
    if (ch && ch.message && typeof ch.message.content === "string") return ch.message.content;
    if (ch && typeof ch.text === "string") return ch.text;   // defensive: text-style choice
    if (typeof d.reply === "string") return d.reply;         // defensive: /say-style
    return null;
  }

  // ---- replay latest run with current dials ---------------------------------------------------
  function wireReplay(ctx) {
    var btn = document.getElementById("bx-replaybtn");
    if (!btn) return;
    btn.addEventListener("click", function () { runReplay(ctx); });
  }

  // the current dial values, meaningfully-nonzero only (mirrors the run-log's |v|>=0.05 convention).
  function currentDials() {
    var out = {};
    Object.keys(state.values).forEach(function (k) {
      var v = state.values[k];
      if (typeof v === "number" && Math.abs(v) >= 0.05) out[k] = v;
    });
    return out;
  }

  function runReplay(ctx) {
    var btn = document.getElementById("bx-replaybtn");
    var msg = document.getElementById("bx-replaymsg");
    var cmp = document.getElementById("bx-compare");
    if (!msg || !cmp) return;
    cmp.innerHTML = "";
    msg.textContent = "";
    if (btn) btn.disabled = true;

    ctx.getJSON("/runs", { runs: [] }).then(function (d) {
      var runs = (d && d.runs) || [];
      var latest = runs[0]; // GET /runs is newest-first
      if (!latest || !latest.id) {
        if (btn) btn.disabled = false;
        msg.textContent = "No runs yet — send a request from any client (or use Try it above), then replay it here.";
        return;
      }
      msg.textContent = "Replaying your latest request with " + describeDials() + "…";

      // Guarded: the replay endpoint (F1) may not exist yet -> postJSON returns our fallback on 404.
      // We can't see the HTTP status through the guard, so a null/erroring result is treated as
      // "not online yet" and we show the F1 note. It starts working automatically once F1 lands.
      ctx.postJSON("/runs/" + latest.id + "/replay",
        { changes_applied: { behavior_overrides: currentDials() } }, null
      ).then(function (res) {
        if (btn) btn.disabled = false;
        if (!res || res.error) {
          msg.innerHTML = "";
          msg.appendChild(S.el("span", { class: "cs-badge" }, ["replay coming online (F1)"]));
          msg.appendChild(document.createTextNode(
            " — this button will start working automatically once the replay engine lands. " +
            "For now, use Try it above to feel the dials."));
          return;
        }
        msg.textContent = "";
        drawCompare(cmp, latest, res, ctx);
      });
    });
  }

  function describeDials() {
    var dials = currentDials();
    var keys = Object.keys(dials);
    if (!keys.length) return "no dials set (all balanced)";
    keys.sort(function (a, b) { return Math.abs(dials[b]) - Math.abs(dials[a]); });
    var top = keys.slice(0, 3).join(", ");
    return keys.length + " dial" + (keys.length === 1 ? "" : "s") + " (" + top +
      (keys.length > 3 ? "…" : "") + ")";
  }

  // original (parent run) vs replayed (child run) side by side.
  function drawCompare(cmp, parent, res, ctx) {
    cmp.innerHTML = "";
    var origText = parent.response || parent.response_summary || "(original answer unavailable)";
    var newText = extractReplayText(res);

    cmp.appendChild(S.el("div", { class: "ans before" }, [
      S.el("div", { class: "tag" }, ["original"]),
      parent.prompt_summary ? S.el("div", { class: "bx-cprompt" }, [parent.prompt_summary]) : null,
      S.el("div", { class: "bx-ctext" }, [origText]),
    ].filter(Boolean)));

    cmp.appendChild(S.el("div", { class: "ans after" }, [
      S.el("div", { class: "tag" }, ["with these dials"]),
      S.el("div", { class: "bx-cnote" }, [describeDials()]),
      S.el("div", { class: "bx-ctext" }, [newText || "(no reply)"]),
    ]));
  }

  // the replay result may be a run record ({response}), a nested {run:{...}}, or an OpenAI reply.
  function extractReplayText(res) {
    if (!res) return "";
    if (typeof res.response === "string") return res.response;
    if (res.child && typeof res.child.response === "string") return res.child.response;
    if (res.run && typeof res.run.response === "string") return res.run.response;
    var oai = extractReply(res);
    return oai || "";
  }

  // ---- page-scoped styles (kept minimal; reuses clozn.css vars + classes) ---------------------
  ensureStyle();
  function ensureStyle() {
    if (document.getElementById("bx-style")) return;
    var css =
      ".bx{padding:6px 18px 18px}" +
      ".bx-loading,.bx-empty{color:var(--faint);font-size:13px;font-style:italic;padding:6px 0}" +
      ".bx-hint{color:var(--faint);font-size:12.5px;margin:0 0 12px;line-height:1.45}" +
      ".bx-slider{margin:15px 0}" +
      ".bx-lab{display:flex;justify-content:space-between;align-items:baseline;font-size:13px;color:var(--soft);gap:10px}" +
      ".bx-lab b{color:var(--ink);font-weight:640}" +
      ".bx-meta{display:flex;align-items:baseline;gap:10px;white-space:nowrap}" +
      ".bx-lean{color:var(--faint);font-size:11px}" +
      ".bx-val{color:var(--soft);font-size:12px;min-width:38px;text-align:right;font-variant-numeric:tabular-nums}" +
      ".bx-range{width:100%;accent-color:var(--halo);margin:5px 0 2px}" +
      ".bx-poles{display:flex;justify-content:space-between;font-size:10.5px;color:var(--faint)}" +
      ".bx-calnote{color:var(--faint);font-size:11px;margin:6px 0 0;line-height:1.4}" +
      ".bx-calnote-dead{font-style:italic}" +
      ".bx-slider-dead{opacity:.55}" +
      ".bx-slider-dead .bx-range{cursor:not-allowed}" +
      ".bx-tag{font-size:9.5px;color:var(--cyan);border:1px solid var(--line);border-radius:7px;padding:0 4px;margin-left:6px;vertical-align:middle}" +
      ".bx-del{cursor:pointer;color:var(--faint);margin-left:7px;font-size:11px;vertical-align:middle}" +
      ".bx-del:hover{color:#e0607a}" +
      ".bx-dialmaker{margin-top:16px;border-top:1px solid var(--line);padding-top:14px}" +
      ".bx-newdial{background:none;border:1px dashed var(--line);color:var(--soft);border-radius:18px;padding:8px 0;width:100%}" +
      ".bx-newdial:hover{border-color:var(--halo);color:var(--halo);background:none;box-shadow:none;transform:none}" +
      ".bx-dialform{margin-top:10px}" +
      ".bx-addrow{display:flex;gap:8px;margin:8px 0;align-items:center}" +
      ".bx-addrow input{flex:1;border:1px solid var(--line);border-radius:20px;padding:8px 13px;font:inherit;font-size:13px;outline:none;background:rgba(255,255,255,.8);color:var(--ink)}" +
      ".bx-addrow input:focus{border-color:var(--halo);box-shadow:0 0 0 3px rgba(122,167,255,.14)}" +
      ".bx-dmsg{color:var(--faint);font-size:12px}" +
      ".bx-reply{display:none;margin-top:12px}" +
      ".bx-reply.on{display:block}" +
      ".bx-replytext{white-space:pre-wrap;background:rgba(255,255,255,.7);border:1px solid var(--line);border-radius:14px;padding:11px 14px;color:var(--ink)}" +
      ".bx-err{color:#c0607a;font-size:13px;background:rgba(231,168,196,.12);border:1px solid rgba(231,168,196,.3);border-radius:12px;padding:10px 13px}" +
      ".bx-spinrow{display:flex;align-items:center;gap:9px;color:var(--soft);font-size:12.5px}" +
      ".bx-spin{width:14px;height:14px;border-radius:50%;border:2px solid rgba(122,167,255,.3);border-top-color:var(--halo);animation:bxsp 1s linear infinite;flex:none}" +
      "@keyframes bxsp{to{transform:rotate(360deg)}}" +
      ".bx-replaymsg{color:var(--soft);font-size:12.5px;margin:10px 0 0;line-height:1.5}" +
      ".cmp{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}" +
      "@media(max-width:720px){.cmp{grid-template-columns:1fr}}" +
      ".bx-cprompt,.bx-cnote{color:var(--faint);font-size:11.5px;margin-bottom:8px}" +
      ".bx-ctext{white-space:pre-wrap;color:var(--ink);font-size:13.5px;line-height:1.5}";
    var st = document.createElement("style");
    st.id = "bx-style";
    st.textContent = css;
    document.head.appendChild(st);
  }

  S.register("behavior", { title: "Behavior", render: render });
})();
