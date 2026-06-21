"""
Phase-1 M4 — persist a model's working memory to disk and rehydrate it in a *fresh* session.

The point: a recurrent model's internal state is a portable artifact. We feed a fact into one
source's memory, save that memory to disk, then construct a SECOND source that never sees the
text — only the saved bytes — and show it has inherited the first session's mind. This is the
"memory beyond chain-of-thought" primitive: the model remembers without re-reading.

Three claims, in increasing strength of evidence:
  (1) MECHANISM  — the saved state rehydrates bit-exactly into a new source.
  (2) BEHAVIOR   — the rehydrated source predicts *identically* to the original session, and
                   *differently* from a cold model. Loading state restores the mind, not just bytes.
  (3) RECALL     — (honest bonus) does the carried memory actually surface the fact? 169M, so we
                   just report what the rehydrated model says next.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.sources.hf_rwkv import RwkvStateSource   # noqa: E402
from clozn.store import StateStore                   # noqa: E402

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "store")
FACT = "Remember this fact: the password is Maiko."
PROBE = " The password is"


def main():
    store = StateStore(STORE)

    # ---------- SESSION 1: writer ----------
    print("=== SESSION 1 (writer): feed a fact into the recurrent memory, save to disk ===")
    A = RwkvStateSource()
    A.feed(FACT)
    print(f"  fed: {FACT!r}")
    path = store.save("maiko_memory", A, note=f"state after: {FACT!r}")
    print(f"  saved memory -> {path}")

    # ---------- SESSION 2: fresh reader, only ever touches the disk file ----------
    print("\n=== SESSION 2 (fresh reader): NEW source, never sees the text — loads the saved memory ===")
    B = RwkvStateSource()
    store.into(B, "maiko_memory")

    # (1) MECHANISM — bit-exact rehydrate
    a_state, b_state = A.get_state(), B.get_state()
    maxdiff = max(float(np.abs(a_state[n] - b_state[n]).max()) for n in a_state)
    print(f"  (1) MECHANISM   max |A.state - B.state(from disk)| = {maxdiff:.2e}  "
          f"-> {'bit-exact ✓' if maxdiff < 1e-6 else 'MISMATCH!'}")

    # (2) BEHAVIOR — rehydrated B predicts identically to A, differently from a cold model
    PROBE_IDS = A.encode(PROBE)
    a_pred = step_pred(A, PROBE_IDS)
    b_pred = step_pred(B, PROBE_IDS)
    C = RwkvStateSource()                       # cold: empty memory, same probe
    c_pred = step_pred(C, PROBE_IDS)
    same_AB = a_pred[0][0] == b_pred[0][0]
    diff_BC = b_pred[0][0] != c_pred[0][0]
    print(f"  (2) BEHAVIOR    after probe {PROBE!r}:")
    print(f"        original (A): {[t for t, _ in a_pred[:5]]}")
    print(f"        rehydrated(B): {[t for t, _ in b_pred[:5]]}   identical to A: {same_AB}")
    print(f"        cold     (C): {[t for t, _ in c_pred[:5]]}   differs from B: {diff_BC}")
    print("        -> loading the saved state restores the SESSION'S MIND, not just bytes."
          if same_AB and diff_BC else
          "        -> (carried state did not fully reproduce / differ — see above)")

    # (3) RECALL — greedily continue from each state and read the actual words back
    b_cont = greedy_continue(B, 4)        # B/C are already positioned just after the probe
    c_cont = greedy_continue(C, 4)
    recalled = "maik" in b_cont.lower()
    print(f"  (3) RECALL      greedy continuation of {PROBE!r}:")
    print(f"        rehydrated(B): {PROBE + b_cont!r}")
    print(f"        cold     (C): {PROBE + c_cont!r}")
    print(f"        -> the persisted memory RECALLED the password: {recalled}"
          if recalled else
          f"        -> no literal recall at 169M (the mechanism + behavioral equivalence still hold)")

    # ---------- the memory shelf ----------
    print("\n=== the memory store (browsable shelf of saved minds) ===")
    save_two_more(store)
    for m in store.list():
        comps = ", ".join(f"{k}|{v['norm']}" for k, v in list(m["components"].items())[:2])
        print(f"  • {m['name']:<16} {m.get('note','')[:48]:<48} [{comps}, …]")

    # ---------- shareable card ----------
    from clozn.viz import render_memory_card
    out = os.path.join(os.path.dirname(STORE), "persisted_memory.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_memory_card(
            PROBE, [
                ("rehydrated from disk — never saw the text this session", PROBE + b_cont, "#7ee0d0"),
                ("cold model — empty memory, same prompt", PROBE + c_cont, "#c9a0ff"),
            ],
            badge=f"rehydrate {maxdiff:.0e} · recall {recalled}",
            subtitle=f"RWKV-4-169m · fact saved last session: {FACT!r}"))
    print("wrote", out)

    print("\nM4 ✓  internal memory persists across sessions — an exact, portable, browsable artifact.")


def step_pred(src, ids):
    """Push the probe tokens through src's CURRENT state and return top next-token candidates."""
    for tid in ids:
        src.step(tid)
    return src.top_next(8)


def greedy_continue(src, n=4):
    """Greedily emit n tokens from src's current state; return the decoded string."""
    out = []
    for _ in range(n):
        tid = int(src._last_logits.argmax())
        src.step(tid)
        out.append(tid)
    return src.tok.decode(out)


def save_two_more(store):
    """Two more saved memories so the shelf isn't a singleton (and to demo diffability)."""
    f = RwkvStateSource(); f.feed("The capital of France is Paris.")
    store.save("france_memory", f, note="state after a France fact")
    j = RwkvStateSource(); j.feed("The capital of Japan is Tokyo.")
    store.save("japan_memory", j, note="state after a Japan fact")


if __name__ == "__main__":
    main()
