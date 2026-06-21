// cloze/kv.hpp — the opaque KV handle the scheduler threads but never inspects
// (DESIGN invariant 4: scheduler writes tokens; model writes KV). The cache manager
// and the generate loop hold it; the model adapter produces it. seq_len() is how many
// board positions it currently covers. Shared by cache.hpp and the (later) model seam.
#pragma once

namespace cloze {

struct KVState {
    virtual ~KVState() = default;
    virtual int seq_len() const = 0;
};

}  // namespace cloze
