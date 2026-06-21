# Vendored llama.cpp patch

`third_party/llama.cpp` is a pinned-SHA submodule with one small **additive** local change (marked
`CLOZE PATCH` in-source): `llama_get_logits_tensor` + `llama_set_skip_raw_logits`. They let the §4.3
confidence-select kernel read the per-step logits on-device and skip llama's decode-time
device→host copy. Defaults are unchanged — without the patch the build just falls back to the
host-logits path, so it's optional.

The diff lives in `patches/0001-llama_get_logits_tensor.patch`; re-apply after a submodule bump:

    git -C core/third_party/llama.cpp apply core/third_party/patches/0001-llama_get_logits_tensor.patch

WIP / experiment — upstreaming these accessors as a proper PR is still TODO.
