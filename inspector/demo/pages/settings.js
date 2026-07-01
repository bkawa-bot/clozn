/* settings.js -- Settings page.  STUB (Issue I1).
 * Boring + safe config to come (model/backend, port, storage, retention, export/import, reset).
 * For now: a short "coming soon" note + the live endpoint/substrate for reference.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;
  function render(view, ctx) {
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("h1", {}, [S.el("span", { class: "glow" }, ["Settings"])]),
        S.el("p", { class: "sub" }, [
          "Model/backend, server port, local storage, memory & trace retention, export/import, and reset will live here — deliberately boring and safe.",
        ]),
        S.el("div", { class: "stubnote" }, [
          S.el("span", { class: "cs-badge" }, ["coming soon"]),
          " Configuration controls are not wired yet.",
        ]),
        S.el("section", { class: "panel", style: "padding:18px;max-width:560px" }, [
          S.el("h2", {}, ["current"]),
          S.el("div", { class: "srows" }, [
            row("OpenAI endpoint", ctx.endpoint, true),
            row("Server base", ctx.base || location.origin, true),
            row("Storage", "~/.clozn", true),
          ]),
        ]),
      ])
    );
  }
  function row(label, value, mono) {
    return S.el("div", { class: "srow" }, [
      S.el("span", { class: "slabel" }, [label]),
      S.el("span", { class: "sval" + (mono ? " mono" : "") }, [value]),
    ]);
  }
  S.register("settings", { title: "Settings", render: render });
})();
