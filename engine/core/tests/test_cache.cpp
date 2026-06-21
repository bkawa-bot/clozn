// test_cache.cpp — checks for the C++ cache port, mirroring test_cache.py behaviors:
// off recomputes all, delta reuse grows cache_hit, the refresh cadence, and the frozen
// (Tier B) exclusion.
#include "cloze/cache.hpp"

#include <cassert>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

using namespace cloze;

struct FakeKV : KVState {
    int n;
    explicit FakeKV(int n_) : n(n_) {}
    int seq_len() const override { return n; }
};
static std::shared_ptr<KVState> kv(int n) { return std::make_shared<FakeKV>(n); }

int main() {
    const std::vector<int> board = {10, 11, 12, 13, 14, 15};  // n = 6
    const std::pair<int, int> active{2, 6};                   // active block [2, 6)

    // off: always recompute all, never reuse — even after observe (the guard is in plan).
    {
        CacheManager m(CacheConfig{"off", 4, 0.5});
        m.observe(board, kv(6), std::nullopt, active, false);
        auto p = m.plan(board, active, 1, false);
        assert(p.kv == nullptr && !p.recompute_kv.has_value() && p.cache_hit == 0.0);
    }
    // delta cold start: no drawers yet -> recompute all.
    {
        CacheManager m(CacheConfig{"delta", 4, 0.5});
        auto p = m.plan(board, active, 0, false);
        assert(p.kv == nullptr && !p.recompute_kv.has_value() && p.cache_hit == 0.0);
    }
    // delta reuse: observe all, then an unchanged board at a non-refresh step reuses everything.
    {
        CacheManager m(CacheConfig{"delta", 4, 0.5});
        m.observe(board, kv(6), std::nullopt, active, false);
        auto p = m.plan(board, active, 1, false);
        assert(p.kv != nullptr);
        assert(p.recompute_kv.has_value() && p.recompute_kv->empty());
        assert(p.cache_hit == 1.0);
    }
    // delta: a changed active token is the only thing recomputed.
    {
        CacheManager m(CacheConfig{"delta", 4, 0.5});
        m.observe(board, kv(6), std::nullopt, active, false);
        auto b2 = board;
        b2[3] = 99;
        auto p = m.plan(b2, active, 1, false);
        assert(p.recompute_kv.has_value() && p.recompute_kv->size() == 1 && (*p.recompute_kv)[0] == 3);
        assert(p.cache_hit > 0.83 && p.cache_hit < 0.84);  // 1 - 1/6
    }
    // delta: a full refresh every N steps recomputes the whole (non-frozen) board.
    {
        CacheManager m(CacheConfig{"delta", 4, 0.5});
        m.observe(board, kv(6), std::nullopt, active, false);
        auto p = m.plan(board, active, 4, false);  // 4 % 4 == 0 -> full
        assert(p.recompute_kv.has_value() && p.recompute_kv->size() == 6);
    }
    // Tier B free freeze: observe(frozen_prefix) freezes [0, active.start); a later full
    // refresh recomputes only [2, 6) and never the frozen prefix.
    {
        CacheManager m(CacheConfig{"delta", 4, 0.5});
        m.observe(board, kv(6), std::nullopt, active, true);  // frozen_until = 2
        auto p = m.plan(board, active, 4, true);
        assert(p.recompute_kv.has_value());
        const auto& r = *p.recompute_kv;
        assert(r.size() == 4 && r.front() == 2 && r.back() == 5);
    }
    std::printf("test_cache: all assertions passed\n");
    return 0;
}
