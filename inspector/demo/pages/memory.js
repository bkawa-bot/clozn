/* memory.js -- Memory page.  STUB for the new card UI (Issues D3/E3).
 * For now: a short "coming soon" note + the existing studio.html memory surface embedded so the
 * current add/remove-trait + strength controls stay fully reachable inside the shell.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;
  function render(view, ctx) {
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("h1", {}, [S.el("span", { class: "glow" }, ["Memory"])]),
        S.el("p", { class: "sub" }, [
          "What the agent thinks it knows — soon as inspectable, source-linked cards (provenance, usage, risk, approve/reject). The structured card UI is coming; for now, add and manage memory below.",
        ]),
        S.el("div", { class: "stubnote" }, [
          S.el("span", { class: "cs-badge" }, ["coming soon"]),
          " Memory Cards v2 (status, source run, usage) will replace the classic panel. ",
          S.el("a", { href: "studio.html", target: "_blank", rel: "noopener" }, ["Open classic studio ↗"]),
        ]),
        S.el("iframe", { class: "embed", src: "studio.html", title: "classic studio — memory + dials", loading: "lazy" }, []),
      ])
    );
  }
  S.register("memory", { title: "Memory", render: render });
})();
