# Clozn UI Feature Inventory

Date: 2026-07-11

Purpose: summarize the product features that are available today, especially the ones that can be shown through the Clozn Studio UI or exposed with modest UI work.

## Current Repo Orientation

- Current checkout inspected for this inventory: `main`.
- The repo is in the post-reorganization shape: `clozn.server`, `clozn.cli`, `clozn.runs`, `clozn.memory`, `clozn.receipts`, `clozn.replay`, `clozn.behavior`, `clozn.readouts`, `clozn.substrates`, and `clozn.profiles` are real packages.
- The old flat product files are gone, including `clozn_server.py`, `cli.py`, `runlog.py`, `receipts.py`, `narrate.py`, `steering.py`, and the old memory/replay/readout flat modules.
- Server routes are split by family under `clozn/server/routes/`.
- Studio is still served as a static shell with pages under `studio/pages/`.
- The current Studio navigation is: Agent, Runs, Memory, Behavior, Lab, Settings.
- The migration status doc reports the last full product test baseline as `1204 passed, 10 skipped`.

## Feature Readiness Summary

| Feature area | Current UI status | Backend/API status | Show to users? | Notes |
|---|---|---|---|---|
| Local OpenAI-compatible chat endpoint | Visible on Agent and Settings | `/v1/chat/completions`, `/v1/models` | Yes | Primary integration story for external tools. |
| Run log | Full Studio page | `/runs`, `/runs/<id>` | Yes | Central product surface. |
| Run Inspector | Full Studio page | run detail, timeline, spans, explain, receipts, replay, branch, J-lens | Yes | Best demo surface. Some panels depend on active substrate. |
| Token confidence and alternatives | Visible in Run Inspector when trace exists | trace capture and `/runs/<id>/spans` | Yes | Engine/local trace path shows strongest version. Hosted/no-trace runs degrade honestly. |
| Explain this answer | Visible in Run Inspector Explain tab | `/runs/<id>/explain` | Yes | Free read/reshape, no generation. Good default explanation surface. |
| Accountable narration | Button in Explain tab | `/runs/<id>/narrate` | Yes, with caveat | Requires generation and a chat-capable substrate. It flags unsupported self-report claims. |
| Causal receipts, regen mode | Visible in Run Inspector | `/runs/<id>/receipts`, `/runs/<id>/receipt` | Yes | Shows measured leave-one-out effects, not self-report. |
| Forced/teacher-scored receipts | Visible behind receipt mode toggle | `/runs/<id>/receipts` with `mode: forced` | Yes, with caveat | Needs engine substrate scoring. Explain as confidence-dependence, not text-change. |
| Exact re-derivation | Visible button | `/runs/<id>/rederive` | Yes, with caveat | Needs engine substrate score tokens. |
| Replay and compare | Visible in Run Inspector and Behavior | `/runs/<id>/replay` | Yes | Good repair workflow. |
| Quick repair presets | Visible in Run Inspector | replay + `/feedback` | Yes | One-click complaint to dial nudge, then compare. |
| Save fix | Visible after replay where persistable | `/steer/set`, memory endpoints | Yes | Makes replayed dial changes default when possible. |
| Branch/time travel | Visible in Run Inspector | `/runs/<id>/branch`, `/timetravel/mode`, `/timetravel/stats` | Yes, with caveat | Branching is available; live KV snapshot acceleration is gated/off by default. |
| Memory cards | Full Memory page | `/memory/*`, `/runs/<id>/propose-memory` | Yes | Strong product identity: readable, reviewable memory. |
| Memory provenance | Visible on cards | `clozn.memory.cards` approve gate | Yes | Important trust/safety story. Cards that cite a run need a quoted span. |
| Memory modes | Visible on Memory page | `/memory/mode` | Yes | Prompt mode is instant; internalized mode can retrain slowly. |
| Facts / slot memory | Visible in Memory page | `/facts/mode`, `/facts/list`, `/facts/add`, `/facts/delete`, `/facts/read` | Yes, with caveat | Separate cue-to-answer mechanism, off by default. Use for controlled demos. |
| Behavior dials | Full Behavior page | `/steer/axes`, `/steer/set`, `/steer/check` | Yes | Includes calibration hints/ranges. |
| Custom behavior dials | Visible in Behavior page | `/steer/custom`, `/steer/custom_delete` | Yes, with caveat | Requires model/backend support to compute custom directions. |
| Learned preference proposals | Visible in Behavior page | `/preferences`, `/preferences/resolve`, `/feedback` | Yes | Connects repeated quick repairs to reviewable default changes. |
| Profiles/personas | Masthead + Settings page | `/profiles/list`, `/profiles/save`, `/profiles/switch`, `/profiles/export`, `/profiles/import` | Yes | Portable bundles of memory cards, dials, custom dials, and facts. |
| Runtime switching | Settings page | `/substrate` | Yes, with caveat | Switching can reload/re-exec the runtime and take time. |
| Capture tier | Backend only | `/capture/tier` | Not yet | Useful setting, but not currently a first-class Studio control. |
| `clozn_trust` OpenAI field | API only | opt-in field on `/v1/chat/completions` | Not yet in Studio | Good for developer docs or an advanced API panel. |
| Quant receipts / quant check | CLI/API internals only | `clozn.receipts.quant_receipts`, `clozn quant-check` | Not yet | Strong future UI candidate for model comparison. |
| Swap receipts / concept steering | Dev/receipt module only | `clozn.receipts.swap_receipt`, `clozn.behavior.steering.concept_dir` | Not yet | Experimental; keep in Lab/dev until productized. |
| Retention/privacy settings | Placeholder only | no backend yet | No | Settings page labels these as coming soon. |

## Current Studio Pages

### Agent

The Agent page is the operational home screen.

What it shows:

- Whether the local Studio server/runtime is reachable.
- Active substrate and model.
- Copyable OpenAI-compatible endpoint.
- Memory card count.
- Active behavior dials.
- Recent runs count.
- Engine hook availability.
- A quick test prompt that sends a chat completion.

Primary endpoints:

- `GET /substrate`
- `GET /engine/health`
- `GET /runs`
- `POST /memory/cards`
- `POST /steer/axes`
- `POST /v1/chat/completions`

Show value:

- "Clozn is a local runtime you can point tools at."
- "The endpoint is visible and copyable."
- "The same runtime is inspectable through Runs and Run Inspector."

### Runs

The Runs page is the run-history index.

What it shows:

- Recent runs, newest first.
- Prompt/response summary.
- Source/client.
- Runtime/model metadata where available.
- Memory and dial influence summaries.
- Duration.
- Flags such as memory, steered, pending-memory, low-confidence, replayed, error, and long.
- Filters by source and flags.

Primary endpoints:

- `GET /runs`
- click-through to `#/run/<id>`

Show value:

- "Every interaction becomes an inspectable artifact."
- "Runs are not hidden logs; they are the entry point for debugging and repair."

### Run Inspector

The Run Inspector is the strongest product demo surface.

What it shows:

- Transcript and stored run metadata.
- Token timeline with confidence tinting.
- Low-confidence branch points.
- Clickable alternatives for tokens when trace alternatives exist.
- Workspace/readout events when present on the trace.
- Branch lineage tree for original, replayed, and branched runs.
- Influence panel for active memory cards, behavior dials, and model/runtime details.
- Explain tab with a read-only summary of confidence, active influences, and concepts.

Primary endpoints:

- `GET /runs/<id>`
- `GET /runs/<id>/family`
- `GET /runs/<id>/timeline`
- `GET /runs/<id>/spans`
- `POST /runs/<id>/explain`

Show value:

- "Clozn shows where the model was unsure."
- "Clozn shows what was active on the turn without claiming it caused the answer until measured."
- "A run is replayable, branchable, exportable, and explainable."

### Run Inspector: Receipts and Explanation

Receipt features currently surfaced:

- Memory receipt: re-run with memory off.
- Per-card receipt: re-run without one specific memory card.
- Dials receipt: re-run with behavior dials off.
- Forced receipt mode: teacher-force the exact answer and score how much confidence depended on each influence.
- Exact re-derivation: reproduce the exact stored answer token by token without sampling.
- J-lens readout: per-token "disposed to say" read with layer/top-k controls and provenance caption.
- Accountable narration: generate a receipt-constrained explanation and flag unsupported claims.
- Export bundle: server supports JSON or Markdown export for a run.

Primary endpoints:

- `POST /runs/<id>/receipt`
- `POST /runs/<id>/receipts`
- `POST /runs/<id>/rederive`
- `POST /runs/<id>/jlens`
- `POST /runs/<id>/narrate`
- `GET /runs/<id>/export`

Show value:

- "The model's self-report is not trusted by default."
- "Clozn measures influence by ablation or teacher-forced scoring."
- "The UI distinguishes active influence, causal text change, and confidence dependence."

Caveats:

- Regeneration receipts need a chat-capable substrate.
- Forced receipts and exact re-derivation need engine scoring.
- J-lens needs the engine substrate with a J-lens loaded.
- J-lens should be described as a fitted linear lens, not literal thought decoding.

### Run Inspector: Repair, Replay, and Branching

Repair features currently surfaced:

- Manual replay buttons, including making a reply more concise and replaying without memory/dials.
- Quick repair presets:
  - Too verbose -> nudge toward concise.
  - Too vague -> nudge toward concrete.
  - Too agreeable -> nudge toward candid.
  - Too cold -> nudge toward warm.
- Replay comparison with original vs replayed text.
- Client-computed diff/receipt strip.
- Token/probability comparison where available.
- Save this fix when a replayed change maps to a persistable default.
- Rewind and branch from a chosen conversation turn.
- Optional alternate user message at the branch point.
- Propose a durable memory from a run.
- For style-like memory proposals, offer "set the dial instead."

Primary endpoints:

- `POST /runs/<id>/replay`
- `POST /runs/<id>/branch`
- `POST /runs/<id>/propose-memory`
- `POST /feedback`
- `POST /steer/set`
- `POST /memory/reject`

Show value:

- "A bad answer is not just deleted; it becomes a repair loop."
- "Users can compare the old answer with the repaired answer before saving a default."
- "Memory and behavior are reviewable, not silently mutated."

### Memory

The Memory page is a complete management UI for durable user memory.

What it shows:

- Pending memory cards awaiting review.
- Active learned traits.
- Disabled cards.
- Optional rejected-card view.
- Add memory manually.
- Approve, reject, edit, disable/enable, and delete cards.
- Strength control.
- Prompt vs internalized memory mode.
- Retraining status for internalized mode.
- Provenance blocks with quoted user text and links back to source runs.
- Warnings for unbacked provenance claims.
- Warnings for instruction-like memory.
- Dial suggestions for style preferences that should be behavior dials instead of memory cards.

Primary endpoints:

- `POST /memory/cards`
- `POST /memory/add`
- `POST /memory/approve`
- `POST /memory/reject`
- `POST /memory/disable`
- `POST /memory/enable`
- `POST /memory/edit`
- `POST /memory/remove`
- `POST /memory/strength`
- `POST /memory/retrain-status`
- `GET /memory/mode`
- `POST /memory/mode`
- `GET /memory/<id>/runs`
- `POST /runs/<id>/propose-memory`

Show value:

- "Memory is not hidden prompt stuffing."
- "Users can inspect, approve, edit, disable, or delete what the model remembers."
- "Proposed memories need evidence before they become active."

Caveats:

- Prompt mode is the easiest user-facing mode: instant edits, no retraining.
- Internalized mode is research-heavy and can trigger slow background retraining.
- Manual memory additions are treated as self-authored and do not need run provenance.

### Memory: Facts Tier

Facts are a separate cue-to-answer slot-memory mechanism, shown inside the Memory page.

What it shows:

- Facts tier on/off.
- Stored cue -> answer entries.
- Add fact.
- Delete fact.
- Read/probe a cue.
- Honest read receipt: hit, abstention, gate value, and slot latency.

Primary endpoints:

- `POST /facts/mode`
- `POST /facts/list`
- `POST /facts/add`
- `POST /facts/delete`
- `POST /facts/read`

Show value:

- "Traits and facts are different mechanisms."
- "Facts can be surgically added, read, and deleted."
- "The system reports when it abstains instead of hallucinating a fact."

Caveats:

- Facts are off by default.
- This is best shown in a controlled demo with a known cue/answer pair.

### Behavior

The Behavior page manages steering dials.

What it shows:

- Built-in tone/cognitive behavior axes.
- Slider values for active dials.
- Calibration hints, usable ranges, derail points, and no-effect labels.
- Library/custom dial tags.
- Create custom dial from positive and negative pole text.
- Delete custom dials.
- Try It prompt using current dials.
- Replay latest run with current dials and compare.
- Learned preference proposals from feedback/quick repairs.
- Approve or dismiss preference proposals.

Primary endpoints:

- `POST /steer/axes`
- `POST /steer/set`
- `POST /steer/check`
- `POST /steer/custom`
- `POST /steer/custom_delete`
- `POST /v1/chat/completions`
- `GET /runs`
- `POST /runs/<id>/replay`
- `POST /preferences`
- `POST /preferences/resolve`

Show value:

- "Behavior is a set of inspectable controls, not prompt folklore."
- "A user can test a dial, replay a prior answer under new dials, and then save it."
- "Repeated repair behavior can become a reviewable preference proposal."

Caveats:

- Some custom dial computation depends on the active substrate/model.
- Calibration should be shown as measured operating guidance, not universal truth.

### Settings

The Settings page covers runtime, storage, profiles, and reset.

What it shows:

- Active substrate/model.
- Copyable endpoint.
- Available substrates.
- Explicit model/substrate switch.
- Local storage path and what lives there.
- Counts for runs, memories, and active dials.
- Profiles list.
- Create profile from current memory/behavior state.
- Switch profile.
- Export/import profile JSON.
- Guarded memory reset.
- Coming-soon placeholders for retention/privacy.

Primary endpoints:

- `GET /substrate`
- `POST /substrate`
- `GET /runs`
- `POST /memory/cards`
- `POST /steer/axes`
- `GET /profiles/list`
- `POST /profiles/save`
- `POST /profiles/switch`
- `POST /profiles/export`
- `POST /profiles/import`
- `POST /reset`

Show value:

- "Clozn is local-first and state is visible."
- "Personas are portable bundles, not opaque cloud accounts."
- "Users can switch runtime/model deliberately."

Caveats:

- Substrate switching can briefly take the server offline while it reloads.
- Retention and privacy controls are UI placeholders today, not backed settings.

### Lab

The Lab page collects deep glass-box surfaces.

Current Lab surfaces:

- Engine internals (`engine.html`): read/edit/write/observe on the C++ runtime, harvest residuals, apply edits, watch next-token predictions move.
- Brain concept readout (`brain.html`): concept activation graph for the Qwen substrate.
- Denoise trace (`denoise.html`): diffusion/Dream token-board trace over denoising passes.
- J-lens (`jlens.html`): paste text, choose layer/top-k, read per-token fitted linear-lens outputs.
- Combined instrument link (`instrument.html`).
- Window index link.

Primary endpoints:

- `POST /engine/harvest`
- `POST /engine/layers`
- `POST /engine/concepts`
- `POST /engine/observe`
- `POST /engine/steer/axes`
- `POST /engine/steer/check`
- `POST /engine/chat`
- `POST /say`
- `POST /denoise`
- `POST /jlens`

Show value:

- "Lab is for the deep glass-box demos that require specific substrates or engine hooks."
- "It keeps experimental surfaces available without mixing them into the core product flow."

Caveats:

- Engine internals need a GGUF engine process.
- Brain readout needs the Qwen substrate and concept assets.
- Denoise trace needs the Dream substrate.
- J-lens needs an engine substrate with a loaded J-lens.

## API Features That Could Be Surfaced More Clearly

### OpenAI-Compatible Chat

Available now:

- `GET /v1/models`
- `POST /v1/chat/completions`
- Streaming support when the substrate supports `chat_stream`.
- `clozn_run_id` returned for logged runs.
- `X-Clozn-Run-Id` response header.
- Optional `clozn_trust: true` request field.

UI opportunity:

- Add an API panel that shows an example request and response.
- Add a toggle/example for `clozn_trust`.
- Link returned run IDs directly to Run Inspector.

### Trust Field

Available now:

- If a caller sends `"clozn_trust": true`, the OpenAI-compatible response can include `clozn_spans`.
- Spans are generated by the same producer as `/runs/<id>/spans`.
- The response includes a note that spans are raw, uncalibrated model probabilities and self-confidence is not correctness.

UI opportunity:

- Add an "API trust spans" demo on Agent or Lab.
- Show the JSON field and the same spans rendered visually.

### Capture Tier

Available now:

- `GET /capture/tier`
- `POST /capture/tier`
- Supported tiers include light, standard, deep, and lab.

UI opportunity:

- Add a Settings or Lab control for capture level.
- Explain that light captures less trace detail, while deeper tiers enable richer Run Inspector views.

### Quant Receipts / Quant Check

Available in code/tests:

- `clozn.receipts.quant_receipts`
- `clozn quant-check`
- Compares two quantizations/models by scoring the same continuations.
- Reports preserved tokens, argmax flips, skipped runs, and caveats.

UI opportunity:

- Add a Lab card for "Quant check."
- Let users choose two local GGUFs and run a small fixed prompt suite or compare from existing runs.

Readiness:

- Not currently a Studio surface.
- Live two-engine path is heavier than ordinary UI actions.

### Swap Receipts and Concept-Direction Steering

Available in code/tests:

- `clozn.receipts.swap_receipt`
- `clozn.behavior.steering.concept_dir`
- Reads a disposition with J-lens, injects a target concept direction, compares baseline/swap/null arms.

UI opportunity:

- Future Lab-only experiment: "what happens if we swap this concept?"

Readiness:

- Experimental.
- Needs engine, J-lens sidecar, and unembed support.
- Do not put in the main user flow yet.

### Self-Report Reliability

Available in code/tests:

- `clozn.receipts.self_report_reliability`
- Compares self-report claims against receipt-backed premises.
- Classifies claims as faithful credit, confabulated credit, unattributed claim, missed driver, or correct silence.

UI opportunity:

- Extend the Explain tab's narration output with a fuller reliability table.

Readiness:

- The current UI already surfaces the safer/narrower narration flags.
- The fuller taxonomy is probably better as an advanced Explain/Lab panel.

## Recommended User Demo Flow

1. Start at Agent.
   - Show active local runtime and copyable `/v1` endpoint.
   - Send the quick test prompt.

2. Open Runs.
   - Show the logged run.
   - Filter or point out flags such as memory, steered, low-confidence, or replayed.

3. Open Run Inspector.
   - Show transcript and run metadata.
   - Show token confidence and alternatives.
   - Show active memory/dials in the influence column.

4. Click Explain.
   - Show the free explanation summary.
   - Use "Why did it say this?" only if the substrate is ready, because it performs generation.

5. Prove an influence.
   - Use a memory or dials receipt.
   - Explain the difference between active influence and measured causal effect.

6. Repair the answer.
   - Use a quick repair preset, compare original vs replayed, then save if persistable.

7. Propose memory.
   - Propose a memory from the run.
   - Go to Memory, show pending review and provenance.
   - Approve only when backed by a quote.

8. Adjust Behavior.
   - Move a dial.
   - Try it live.
   - Replay the latest run with the new dial.

9. Save a profile.
   - In Settings, save/export a persona bundle.
   - Show that memory and behavior are portable state.

10. Use Lab only when the substrate supports it.
    - Engine internals for GGUF engine.
    - Brain for Qwen.
    - Denoise for Dream.
    - J-lens for loaded J-lens sidecar.

## What To Show Now

Prioritize these in the product UI and demos:

- Agent status and endpoint copy.
- Runs list.
- Run Inspector token timeline.
- Explain tab.
- Receipts for memory/dials.
- Replay/quick repair/save fix.
- Memory review with provenance.
- Behavior dials and Try It.
- Profiles.
- Lab entry points with clear substrate caveats.

## What To Keep Advanced Or Hidden For Now

- Capture-tier controls, until placed in Settings/Lab with clear wording.
- `clozn_trust`, unless showing API/developer workflow.
- Quant check UI, until the two-model flow is polished.
- Swap receipts/concept-direction steering, until stabilized as a Lab experiment.
- Retention/privacy controls, because the current Settings entries are placeholders.

## Product Framing

The strongest current identity is:

> Clozn is a local glass-box AI runtime where every answer becomes inspectable, replayable, repairable, and provable.

The UI should emphasize:

- Local runtime and OpenAI-compatible endpoint.
- Runs as durable artifacts.
- Inspectable traces, confidence, and alternatives.
- Reviewable memory with provenance.
- Behavior dials instead of hidden prompt hacks.
- Receipts instead of self-report.
- Replay and branch workflows for repair.
- Lab surfaces for deeper model internals.

Avoid overclaiming:

- Do not describe J-lens as literal thought reading.
- Do not equate token confidence with correctness.
- Do not say active memory/dials caused an answer until a receipt proves an effect.
- Do not present forced receipts as "the text would have changed"; they measure confidence dependence on the exact answer.
- Do not imply privacy/retention controls exist before the backend exists.
