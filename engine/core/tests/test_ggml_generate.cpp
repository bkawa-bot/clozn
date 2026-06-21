// test_ggml_generate.cpp — Phase 3 slice 1 end-to-end: drive the C++ generate loop with
// the real ggml adapter and cross-check it against the Python lab's golden oracle
// (lab/tests/golden/dcoder_add.json), DESIGN invariant 3.
//
// The oracle pins the open-dCoder whole-sequence run for prompt "def add(a, b):\n    return
// a +", max_new=8, steps=4 (fixed), confidence_topk quota (k=null), greedy. We replay the
// SAME prompt + config through the C++ seam (model.hpp) + sampler (sample.hpp) + loop
// (generate.hpp) and assert:
//   - prompt_ids match the lab byte-for-byte (validates the GGUF tokenizer vs HF),
//   - the committed generation region (picks) matches exactly,
//   - text / reason / new_tokens match.
//
// Caveat (honesty): the golden was recorded with the HF checkpoint in bfloat16; here we run
// an f16 GGUF through llama.cpp. Picks should match where the model is confident; any
// low-confidence slot that flips is a real numeric difference between the two backends, and
// the driver reports per-position diffs rather than hiding them. Confidences are NOT
// asserted bitwise (different quantization + reduction order).
//
// usage: test_ggml_generate <model.gguf>
#include "cloze/blocks.hpp"
#include "cloze/generate.hpp"
#include "cloze/model_ggml.hpp"
#include "cloze/sample.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ggml_generate <model.gguf>\n");
        return 1;
    }
    const int MASK = 151665;  // open-dCoder <M>
    const int EOS = 151643;
    // Exactly the lab golden's prompt (note the newline + 4-space indent).
    const char* prompt = "def add(a, b):\n    return a +";

    // --- The oracle: lab/tests/golden/dcoder_add.json ---
    const std::vector<int> golden_prompt_ids = {750, 912, 2877, 11, 293, 982, 262, 470, 264, 488};
    // Final board, generation region [10, 18): the 8 committed slots (last is EOS).
    const std::vector<int> golden_gen_region = {293, 271, 1350, 25906, 7, 16, 11, 151643};
    const std::vector<int> golden_kept = {293, 271, 1350, 25906, 7, 16, 11};  // truncated at EOS
    const char* golden_text = " b\n\nprint(add(1,";
    const char* golden_reason = "eos";
    const int golden_new_tokens = 7;
    const int golden_steps_total = 4;

    cloze::GgmlAdapter adapter(argv[1], MASK, EOS);
    const std::vector<int> prompt_ids = adapter.encode(prompt);
    const int p = static_cast<int>(prompt_ids.size());

    bool ok = true;

    // --- Check 0: tokenizer parity (GGUF gpt2/qwen2 tokenizer vs the lab's HF tokenizer).
    bool tok_ok = (prompt_ids == golden_prompt_ids);
    std::printf("prompt_len=%d (golden %d)  ids:", p, (int)golden_prompt_ids.size());
    for (int id : prompt_ids) std::printf(" %d", id);
    std::printf("\nCHECK 0 (tokenizer parity): %s\n", tok_ok ? "PASS" : "MISMATCH");
    ok = ok && tok_ok;

    // --- Check 1: cold forward through the seam -> sample_candidates -> first masked slot.
    // The lab's step-0 commits pos10 = ' b' (token 293); confirm the cold pass agrees.
    {
        std::vector<int> board = prompt_ids;
        for (int i = 0; i < 8; ++i) board.push_back(MASK);
        const int n = static_cast<int>(board.size());
        std::vector<int> masked;
        for (int i = p; i < n; ++i) masked.push_back(i);
        const cloze::Mask attn = cloze::attention_mask(n, p, 0);
        cloze::ForwardResult fwd =
            adapter.forward(board, attn, nullptr, std::nullopt, masked);
        const auto cands = cloze::sample_candidates(fwd, masked);
        const auto& first = cands.front();  // pos p (first masked)
        const std::string piece = adapter.decode({first.token_id});
        bool c1 = (first.pos == p && first.token_id == 293 && piece == " b");
        std::printf("cold first-mask pos=%d -> token %d = '%s' (conf=%.4f; golden 0.9542)\n",
                    first.pos, first.token_id, piece.c_str(), first.confidence);
        std::printf("CHECK 1 (seam reproduces forward): %s\n", c1 ? "PASS" : "MISMATCH");
        ok = ok && c1;
    }

    // --- Check 2: full generate loop vs the golden, picks exact.
    cloze::GenerateConfig cfg;
    cfg.max_new = 8;
    cfg.steps = 4;
    cfg.block_len = 0;  // whole-sequence
    cfg.topk = -1;      // confidence_topk quota (k=null in the golden)
    cloze::GenerateResult res = cloze::generate(adapter, prompt_ids, cfg);

    std::vector<int> gen_region(res.board.begin() + p, res.board.end());
    std::printf("\ngenerate: new_tokens=%d (golden %d) steps_total=%d (golden %d) reason=%s (golden %s)\n",
                res.new_tokens, golden_new_tokens, res.steps_total, golden_steps_total,
                res.reason.c_str(), golden_reason);

    std::printf("gen region:  ");
    for (int id : gen_region) std::printf("%d ", id);
    std::printf("\ngolden region: ");
    for (int id : golden_gen_region) std::printf("%d ", id);
    std::printf("\n");

    int picks_matched = 0;
    const int region_n = (int)golden_gen_region.size();
    for (int i = 0; i < region_n; ++i) {
        const bool m = (i < (int)gen_region.size() && gen_region[i] == golden_gen_region[i]);
        if (m) ++picks_matched;
        else std::printf("  diff at gen pos %d: got %d, golden %d\n",
                         p + i, i < (int)gen_region.size() ? gen_region[i] : -1,
                         golden_gen_region[i]);
    }
    bool picks_ok = (picks_matched == region_n);
    std::printf("CHECK 2 (picks vs golden): %d/%d matched -> %s\n",
                picks_matched, region_n, picks_ok ? "PASS" : "PARTIAL");

    bool text_ok = (res.text == golden_text && res.reason == golden_reason &&
                    res.new_tokens == golden_new_tokens && res.generated == golden_kept);
    std::printf("text: \"%s\"\ngolden: \"%s\"\nCHECK 3 (text/reason/kept): %s\n",
                res.text.c_str(), golden_text, text_ok ? "PASS" : "MISMATCH");

    // --- Check 4 (slice 2): BLOCK mode with exact Tier A/B KV reuse vs dcoder_add_blocks.json.
    // The one-way law (frozen prefix reused, active block recomputed) must reproduce the
    // cache=off block golden token-for-token — that exactness is the core efficiency claim.
    const std::vector<int> golden_block_region = {293, 271, 750, 526, 526, 526, 526, 526};
    const char* golden_block_text = " b\n\ndef int int int int int";
    const char* golden_block_reason = "length";
    const int golden_block_new = 8;
    const int golden_block_steps = 8;

    cloze::GenerateConfig bcfg;
    bcfg.max_new = 8;
    bcfg.steps = 4;       // per-block budget
    bcfg.block_len = 4;   // semi-AR: two blocks of 4
    bcfg.topk = -1;       // quota
    // Exact Tier A/B reuse: reuse the frozen prefix, recompute the active block every pass.
    cloze::CacheConfig bcache;
    bcache.mode = "delta";
    bcache.full_refresh_every = 1;
    cloze::GenerateResult bres = cloze::generate(adapter, prompt_ids, bcfg, bcache);

    std::vector<int> block_region(bres.board.begin() + p, bres.board.end());
    std::printf("\n[block mode, exact KV reuse] new_tokens=%d (golden %d) steps_total=%d (golden %d) reason=%s (golden %s)\n",
                bres.new_tokens, golden_block_new, bres.steps_total, golden_block_steps,
                bres.reason.c_str(), golden_block_reason);
    std::printf("block region:  ");
    for (int id : block_region) std::printf("%d ", id);
    std::printf("\ngolden region: ");
    for (int id : golden_block_region) std::printf("%d ", id);
    std::printf("\n");

    int bmatch = 0;
    for (int i = 0; i < (int)golden_block_region.size(); ++i) {
        const bool m = (i < (int)block_region.size() && block_region[i] == golden_block_region[i]);
        if (m) ++bmatch;
        else std::printf("  diff at gen pos %d: got %d, golden %d\n",
                         p + i, i < (int)block_region.size() ? block_region[i] : -1,
                         golden_block_region[i]);
    }
    bool block_ok = (bmatch == (int)golden_block_region.size() &&
                     bres.text == golden_block_text && bres.reason == golden_block_reason &&
                     bres.new_tokens == golden_block_new && bres.steps_total == golden_block_steps);
    std::printf("block text: \"%s\"\nCHECK 4 (block KV reuse vs golden): %d/%d picks -> %s\n",
                bres.text.c_str(), bmatch, (int)golden_block_region.size(),
                block_ok ? "PASS" : "MISMATCH");

    // --- Check 5 (core efficiency claim, measured transparently): block mode cache OFF vs exact reuse.
    // Same config, same board required (exactness), and the reuse path must do strictly less
    // forward work. Wall-clock ships alongside the work metric, never instead of it.
    auto run_block = [&](const cloze::CacheConfig& cc) {
        adapter.reset_decoded_tokens();
        auto t0 = std::chrono::steady_clock::now();
        cloze::GenerateResult r = cloze::generate(adapter, prompt_ids, bcfg, cc);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return std::make_pair(r, std::make_pair(adapter.decoded_tokens(), ms));
    };
    cloze::CacheConfig off_cache;  // mode "off" (default): full recompute every pass
    auto [off_res, off_stat] = run_block(off_cache);
    auto [reuse_res, reuse_stat] = run_block(bcache);  // delta, full_refresh_every=1

    bool exact_match = (off_res.board == reuse_res.board);
    std::printf("\n[efficiency] block mode, off vs exact Tier A/B reuse (identical board: %s)\n",
                exact_match ? "yes" : "NO");
    std::printf("  cache=off:   decoded %lld tokens in %.1f ms\n", off_stat.first, off_stat.second);
    std::printf("  reuse:       decoded %lld tokens in %.1f ms\n", reuse_stat.first, reuse_stat.second);
    if (reuse_stat.first > 0)
        std::printf("  work saved:  %.2fx fewer token-decodes (%.1fx wall-clock, tiny-model/CPU caveat)\n",
                    (double)off_stat.first / reuse_stat.first,
                    reuse_stat.second > 0 ? off_stat.second / reuse_stat.second : 0.0);
    bool moat_ok = exact_match && reuse_stat.first < off_stat.first;
    std::printf("CHECK 5 (reuse exact AND less work): %s\n", moat_ok ? "PASS" : "MISMATCH");

    // Picks-exact is the hard invariant. CHECK 1/text are derived from it. Tokenizer parity
    // (CHECK 0) is required for the comparison to even be meaningful.
    const bool result = tok_ok && picks_ok && text_ok && block_ok && moat_ok;
    ok = ok && result;
    std::printf("\nRESULT: %s\n", ok ? "PASS" : "MISMATCH");
    return ok ? 0 : 2;
}
