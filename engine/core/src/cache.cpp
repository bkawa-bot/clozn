// cache.cpp — implementation of cloze/cache.hpp, mirroring cache.py plan()/observe().
#include "cloze/cache.hpp"

#include <set>
#include <stdexcept>

namespace cloze {

void CacheConfig::validate() const {
    if (mode != "off" && mode != "delta")
        throw std::invalid_argument("mode must be 'off' or 'delta'");
    if (full_refresh_every < 1)
        throw std::invalid_argument("full_refresh_every must be >= 1");
    if (refresh_fraction < 0.0 || refresh_fraction > 1.0)
        throw std::invalid_argument("refresh_fraction must be in [0, 1]");
}

CacheManager::CacheManager(CacheConfig cfg) : config_(std::move(cfg)) { config_.validate(); }

ForwardPlan CacheManager::plan(const std::vector<int>& board, std::pair<int, int> active,
                               int block_step, bool frozen_prefix) {
    const int n = static_cast<int>(board.size());

    // off (the single load-bearing guard: never reuse), or cold start (no drawers yet):
    // recompute all, no kv.
    if (config_.mode == "off" || !kv_) {
        return ForwardPlan{nullptr, std::nullopt, 0.0};
    }

    const int lo = active.first, hi = active.second;
    std::set<int> frozen;
    if (frozen_prefix)
        for (int p = 0; p < frozen_until_; ++p) frozen.insert(p);

    // new = positions with no cached_token yet, minus frozen.
    // changed = cached positions whose board token differs now, minus frozen.
    std::set<int> changed;
    for (const auto& kvp : cached_token_)
        if (board[kvp.first] != kvp.second && !frozen.count(kvp.first)) changed.insert(kvp.first);
    std::set<int> new_pos;
    for (int p = 0; p < n; ++p)
        if (!cached_token_.count(p) && !frozen.count(p)) new_pos.insert(p);

    int active_count = 0, changed_in_active = 0;
    for (int p = lo; p < hi; ++p) {
        ++active_count;
        if (changed.count(p)) ++changed_in_active;
    }
    const double churn = active_count ? static_cast<double>(changed_in_active) / active_count : 0.0;
    const bool full = (block_step % config_.full_refresh_every == 0) || (churn > config_.refresh_fraction);

    // Frozen positions are excluded from BOTH branches — that exclusion is the free freeze.
    std::set<int> recompute;
    if (full) {
        for (int p = 0; p < n; ++p)
            if (!frozen.count(p)) recompute.insert(p);
    } else {
        recompute = new_pos;
        recompute.insert(changed.begin(), changed.end());
    }
    std::vector<int> ordered(recompute.begin(), recompute.end());  // std::set is sorted
    const double cache_hit = n ? 1.0 - static_cast<double>(ordered.size()) / n : 0.0;
    return ForwardPlan{kv_, std::move(ordered), cache_hit};
}

void CacheManager::observe(const std::vector<int>& board, std::shared_ptr<KVState> new_kv,
                           const std::optional<std::vector<int>>& recomputed,
                           std::optional<std::pair<int, int>> active, bool frozen_prefix) {
    kv_ = std::move(new_kv);
    if (!recomputed.has_value()) {
        cached_token_.clear();
        for (int p = 0; p < static_cast<int>(board.size()); ++p) cached_token_[p] = board[p];
    } else {
        for (int p : *recomputed) cached_token_[p] = board[p];
    }
    if (frozen_prefix && active.has_value()) frozen_until_ = active->first;
}

}  // namespace cloze
