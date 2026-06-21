// cloze/cache.hpp — KV cache manager (DESIGN §5.5), C++ port of
// lab/cloze_lab/scheduler/cache.py. Pure scheduler logic: it never fabricates K/V
// (invariant 4) — it only decides which positions to recompute each pass and threads the
// adapter's KVState plus the cached_token "built-as" labels that reconcile board vs drawers.
//
// Tier A (prompt) / Tier B (frozen blocks) are exact and free under block-causal attention
// (frozen_prefix=true): [0, frozen_until) is never recomputed. Tier C (active block) is the
// approximate delta: recompute only positions whose token changed; reuse the rest (stale
// neighbors = the drift), bounded by a periodic full refresh and a churn trigger. mode="off"
// recomputes everything every pass (the exact baseline the divergence bench measures against).
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cloze/kv.hpp"

namespace cloze {

// Exactness knobs, exposed never hidden (DESIGN invariant 5).
struct CacheConfig {
    std::string mode = "off";    // "off" = exact every pass | "delta" = Tier C reuse
    int full_refresh_every = 4;  // force a full active-block refresh every N block-steps
    double refresh_fraction = 0.5;  // if > this fraction of the active block changed, full refresh
    void validate() const;       // throws std::invalid_argument on bad values
};

// The cache's decision for one forward.
struct ForwardPlan {
    std::shared_ptr<KVState> kv;                   // null => cold start (recompute all)
    std::optional<std::vector<int>> recompute_kv;  // nullopt => all; else sorted positions
    double cache_hit;                              // fraction of positions reused (0 when all)
};

// Threads K/V and cached_token across the passes of one generation (§5.5).
class CacheManager {
public:
    explicit CacheManager(CacheConfig cfg);

    // Decide reuse for the forward over `board` whose active block is [active.first, active.second).
    ForwardPlan plan(const std::vector<int>& board, std::pair<int, int> active, int block_step,
                     bool frozen_prefix);

    // Record the forward's drawers, refresh cached_token, advance the freeze boundary.
    // recomputed = nullopt means "all positions were recomputed".
    void observe(const std::vector<int>& board, std::shared_ptr<KVState> new_kv,
                 const std::optional<std::vector<int>>& recomputed,
                 std::optional<std::pair<int, int>> active, bool frozen_prefix);

private:
    CacheConfig config_;
    std::shared_ptr<KVState> kv_;
    std::map<int, int> cached_token_;
    int frozen_until_ = 0;  // [0, frozen_until_) is frozen-exact
};

}  // namespace cloze
