"""Memory cards, prompt/internalized mode settings, and topic gating.

The facts/slot-memory tier moved to clozn/lab/slotmem_qwen (reorg Stage B) -- it was never wired to
change the product reply (default-OFF research instrumentation, torch-dependent), so it no longer
lives under clozn.memory at all."""

from . import cards, mode, topic_gate

__all__ = ["cards", "mode", "topic_gate"]
