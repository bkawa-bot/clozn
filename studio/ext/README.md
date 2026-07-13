# clozn lens — in-context glass box (AMBIENT_DELIVERY.md channel 3)

A userscript that puts a small clozn panel on **whatever tool you already use**, as long as it's pointed
at clozn's `/v1`. No build step, no extension store — paste it into Tampermonkey or Violentmonkey.

## Install
1. Install [Tampermonkey](https://www.tampermonkey.net/) (or Violentmonkey) in your browser.
2. Create a new script, paste `clozn-lens.user.js`, save.
3. Edit two things at the top:
   - `CLOZN` — your clozn base URL (default `http://127.0.0.1:8090`).
   - the `@match` lines — the pages of the tool whose replies you want annotated (Open WebUI, LibreChat,
     a local chat page… anything you've configured to call clozn instead of api.openai.com).

## What it does
A collapsible panel (bottom-right) that reads clozn's journal cross-origin (clozn already sends CORS on
GET) and, for the **newest run**, shows: the reply's confidence, how many spans are "worth a look", the
shaky snippets as chips, and one click to the full receipt (`/r/<id>` → the studio, deep-linked). It
matches the newest run to the reply you're looking at, which is correct for a single-user local clozn.

## Honesty
Raw, uncalibrated model confidence — a **prompt to look, not a verdict**; it never says "wrong". Same
signals as the studio's spans and the in-band footer, so it never disagrees with them.

## v1 scope / next
This v1 shows the **panel** — robust and tool-agnostic. Shading the reply text *inline* in the host
page's DOM (the full "color the text right in Cursor" endgame) is per-tool and fragile, so it's the
documented next step, not v1. The data it needs (`GET /runs`, `GET /runs/<id>/spans`, and the calibrated
`POST /runs/<id>/trust_spans`) is already there.
