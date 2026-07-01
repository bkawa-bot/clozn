/* behavior.js -- Behavior page.  STUB for the dials-tied-to-runs UI (Issue G1).
 * For now: a short "coming soon" note + the existing studio.html surface embedded so the live tone
 * dials + custom-dial creation stay fully reachable inside the shell.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;
  function render(view, ctx) {
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("h1", {}, [S.el("span", { class: "glow" }, ["Behavior"])]),
        S.el("p", { class: "sub" }, [
          "How the model responds — plain-language dials (concise ↔ detailed, warm ↔ neutral, technical ↔ plain). The test-on-a-run + before/after view is coming; for now, drag the live dials below.",
        ]),
        S.el("div", { class: "stubnote" }, [
          S.el("span", { class: "cs-badge" }, ["coming soon"]),
          " Test-on-latest-run, before/after compare, and save-as-default land here. ",
          S.el("a", { href: "studio.html", target: "_blank", rel: "noopener" }, ["Open classic studio ↗"]),
        ]),
        S.el("iframe", { class: "embed", src: "studio.html", title: "classic studio — tone dials", loading: "lazy" }, []),
      ])
    );
  }
  S.register("behavior", { title: "Behavior", render: render });
})();
