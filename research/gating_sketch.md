# Contextual gating of a learned preference

Today the consolidated soft-prefix (`SelfTeach.prefix`, an `[m,H]` `nn.Parameter`) is
prepended to *every* prompt at full strength in `_generate`, `_seq_loss`, and `trace`. So
a "answer in a fantasy register" preference fires on a dinner question as hard as on a book
question. We want the prefix to engage only when the prompt is in the rule's domain. Three
honest options, given a soft prefix prepended to a frozen model.

## (a) Relevance gate on injection strength

**Idea.** Scale the prefix by a scalar `g(prompt) in [0,1]` instead of always 1. Store a
*domain anchor* per rule at consolidation time: mean of the frozen model's last-hidden
state over the rule text (or over the consolidation probe prompts) -> a unit vector `d`.
At inference, embed the incoming prompt the same way, get `q`, and set
`g = relu(cos(q, d) - tau) / (1 - tau)` (a soft threshold so off-domain prompts get ~0).

**Plug-in.** One change point: wherever the prefix is concatenated, multiply it first --
`pre = g * self.prefix.detach()[None]` in `_generate` (and mirror in `trace`). `g`
computed in a tiny helper `_gate(prompt_ids)`. No retraining; `tau` hand-tuned.

**Risk.** Cosine in raw hidden space is a coarse domain detector; "what book about food"
straddles both. A single anchor can't represent a multi-topic rule, and the gate is never
trained against the actual behavioral effect, so the threshold is guesswork.

## (b) Small learned gate network

**Idea.** A tiny MLP `g_theta: pooled_prompt_hidden -> sigmoid scalar`, trained jointly with
the prefix. Build *negative* examples (off-domain probes where the target == the
**no-prefix** reply) alongside the existing positives; the gate learns to open on positives
and stay shut on negatives, because only then is the joint loss minimized.

**Plug-in.** `consolidate` adds `g_theta`'s params to the Adam group and appends negatives
to `self.examples`; `_seq_loss` multiplies the prefix by `g_theta(pool(e_p))` before the
concat. Inference multiplies by the same gate.

**Risk.** More parameters fit on the same tiny example set -> overfitting; the gate can
collapse to "always 1" if negatives are too easy, or "always 0" killing the rule. Needs
balanced, genuinely off-domain negatives, which are awkward to mine automatically.

## (c) Per-rule prefixes + router

**Idea.** Keep each rule as its own narrow prefix `P_k` (not one merged prefix). A router
scores the prompt against each rule's anchor (as in (a)) and injects a weighted sum
`sum_k w_k P_k`, with `w` a softmax/top-1 over relevance -- mixture-of-prefixes.

**Plug-in.** `self.prefix` becomes `self.prefixes: list[Parameter]` + `self.anchors`;
`consolidate` trains only the new rule's prefix; a `_route(prompt_ids)->weights` feeds all
concat sites. Cleanest separation: rules don't interfere during TTT.

**Risk.** Router misfires send the wrong prefix (or blends conflicting ones); cost and
prefix length grow linearly with rule count; calibrating cross-rule weights so two
relevant rules co-fire without overpowering the model is the hard, unsolved part.
